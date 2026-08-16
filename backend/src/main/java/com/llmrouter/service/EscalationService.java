package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterEmbedding;
import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.dto.EscalateRequest;
import com.llmrouter.dto.EscalateResponse;
import com.llmrouter.dto.EscalationChainState;
import com.llmrouter.dto.EscalationCompletion;
import com.llmrouter.dto.EscalationContext;
import com.llmrouter.dto.EscalationReservation;
import com.llmrouter.dto.ExistingEscalation;
import com.llmrouter.dto.LlmResult;
import com.llmrouter.dto.RequestLogEntry;
import com.llmrouter.exception.EscalationConflictException;
import com.llmrouter.exception.EscalationInProgressException;
import com.llmrouter.exception.EscalationNotFoundException;
import com.llmrouter.exception.InvalidEscalationException;
import com.llmrouter.service.RouterService.Tier;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * 사용자가 답변에 만족하지 못했을 때 상위 모델로 재요청한다 (기획서 4-1).
 *
 * 사용자가 만족한 티어가 그 질문의 실제 필요 수준이고, 그것이 라우터가
 * 과소 라우팅했다는 정정 라벨이 된다. 이 기록이 재학습 데이터의 유일한 공급원이다.
 *
 * <h3>왜 LLM 을 부르기 전에 행을 먼저 꽂는가</h3>
 * 순진하게 "호출 → 기록" 순서로 두면 중복 요청이 걸러지는 시점이 INSERT 라,
 * 두 번째 요청도 이미 Opus 요금을 내고 나서 409 를 받는다. Opus 응답이 1~3초 걸리는
 * 그 창이 정확히 사용자가 답답해서 한 번 더 누르는 타이밍이다.
 * 예약을 먼저 하면 두 번째 요청은 유니크 인덱스에서 막혀 LLM 까지 가지 못한다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EscalationService {

    private final RouterClient routerClient;
    private final CacheService cacheService;
    private final LlmService llmService;
    private final RequestLogService requestLogService;
    private final EscalationLogService escalationLogService;

    @Value("${llm.escalation.cache-refresh.enabled:true}")
    private boolean cacheRefreshEnabled;

    /**
     * 원본이 캐시 히트였을 때, 이 유사도 미만이면 그 행을 덮어쓰지 않고 새 행을 만든다.
     *
     * 캐시 히트 유래라는 것은 cache_entry.question 이 <b>다른 사용자의 질문</b>이라는 뜻이다.
     * 그 행의 답을 갈아치우면 이번 사용자의 불만 때문에 남의 질문에 달린 답이 바뀐다.
     * 기획서 4-3 은 "갱신하라"고만 하고 히트 유래와 미스 유래를 구분하지 않아 여기서 나눈다.
     */
    @Value("${llm.escalation.cache-refresh.overwrite-similarity:0.95}")
    private double overwriteSimilarity;

    /**
     * 이보다 오래된 미완료 예약은 죽은 프로세스가 남긴 것으로 보고 인수한다.
     * Anthropic SDK 요청 타임아웃(기본 10분)보다 길어야 한다 — application.yml 주석 참조.
     */
    @Value("${llm.escalation.reservation-timeout:PT15M}")
    private Duration reservationTimeout;

    public EscalateResponse escalate(EscalateRequest request) {
        long start = System.currentTimeMillis();
        String originalRequestId = request.getRequestId();

        // 1. 티어 해석과 방향 검증.
        //    ML 의 require_strict_tier_increase 를 내보내기가 아니라 입구에서 강제한다.
        Tier currentTier = parseModel(request.getCurrentModel(), "currentModel");
        Tier targetTier = parseModel(request.getTargetModel(), "targetModel");
        if (targetTier.rank() <= currentTier.rank()) {
            throw new InvalidEscalationException(
                    "더 높은 모델로만 재요청할 수 있습니다. "
                            + request.getCurrentModel() + " -> " + request.getTargetModel());
        }

        // 2. 원본 맥락은 DB 에서 읽는다. 본문의 message 는 쓰지 않는다 — EscalationContext 주석 참조.
        EscalationContext ctx = escalationLogService.loadContext(originalRequestId)
                .orElseThrow(() -> new EscalationNotFoundException(originalRequestId));

        // 3. 멱등 조회. 같은 (질문, 목표 티어) 로 이미 답을 받았다면 그것을 그대로 돌려준다.
        //
        //    체인 말단 검사(4번)보다 먼저 와야 한다. 완료된 뒤에 같은 요청이 다시 오면
        //    말단 티어가 이미 목표 티어라 4번에서 409 로 걸러지고, 정작 가장 흔한
        //    "이미 처리된 요청의 재시도"가 멱등 반환에 닿지 못한다.
        Optional<ExistingEscalation> done = escalationLogService.findByTier(
                originalRequestId, targetTier.label());
        if (done.isPresent() && done.get().completed()) {
            log.info("이미 처리된 에스컬레이션이라 저장된 답을 돌려준다 requestId={} tier={}",
                    originalRequestId, targetTier.label());
            return toResponse(originalRequestId, done.get());
        }

        // 4. 체인 상태 확인. 화면에 떠 있는 답변의 티어와 요청의 currentModel 이 맞아야 한다.
        //    loadChain 은 완결된 행만 보므로 진행 중인 예약은 여기 잡히지 않는다.
        Optional<EscalationChainState> chain = escalationLogService.loadChain(originalRequestId);
        String terminalTier = chain.map(EscalationChainState::lastFinalTier).orElse(ctx.tier());
        if (!terminalTier.equals(currentTier.label())) {
            throw new EscalationConflictException(
                    "이미 상위 모델로 답변한 요청입니다. 현재 티어=" + terminalTier
                            + " 요청한 currentModel=" + request.getCurrentModel());
        }

        List<String> path = buildPath(chain, currentTier, targetTier);
        String initialTier = chain.map(EscalationChainState::initialTier).orElse(ctx.tier());
        int seq = chain.map(EscalationChainState::lastSeq).orElse(0) + 1;

        // 5. 자리 예약. 중복 요청은 여기서 끊긴다 — LLM 요금이 나가기 전이다.
        ReservationOutcome reserved =
                reserve(ctx, originalRequestId, initialTier, path, targetTier, seq);
        if (reserved.replay() != null) {
            return reserved.replay();
        }
        long escalationLogId = reserved.escalationLogId();

        // 6. 상위 모델 호출.
        //    실패하면 예약을 되돌려야 한다. 안 풀면 그 티어로 다시 시도할 수 없게 된다.
        LlmResult result;
        try {
            result = llmService.call(targetTier, ctx.question());
        } catch (RuntimeException e) {
            escalationLogService.release(escalationLogId);
            throw e;
        }

        // ── 여기서부터는 이미 요금을 지불했다. 무슨 일이 있어도 답변은 사용자에게 돌려준다. ──

        // 7. 캐시 갱신 (기획서 4-3). 갱신하지 않으면 이후 유사 질문에 불만족스러웠던 답이 다시 나간다.
        //
        //    임베딩 호출과 캐시 갱신은 부가 작업이라 실패해도 응답을 막지 않는다.
        //    여기서 예외가 새어 나가면 지불한 Opus 답변을 버리고, 예약도 남아
        //    타임아웃 전까지 재시도가 409 로 막힌다 — 사용자는 돈만 내고 아무것도 못 받는다.
        float[] embedding = null;
        String embedModelVersion = null;
        try {
            RouterEmbedding embedded = routerClient.embed(ctx.question());
            embedding = embedded.vector();
            embedModelVersion = embedded.modelVersion();
            refreshCache(ctx, embedding, result, embedModelVersion);
        } catch (RuntimeException e) {
            log.error("에스컬레이션 후처리(임베딩/캐시 갱신) 실패 - 답변은 반환한다 requestId={}",
                    originalRequestId, e);
        }

        // 8. 예약한 행에 결과를 채우고 체인의 종결점으로 만든다.
        long latency = System.currentTimeMillis() - start;
        completeReservation(originalRequestId, escalationLogId, result, embedding, latency);

        // 9. 이 호출 자체도 과금되는 LLM 요청이므로 request_log 에 남긴다.
        //    origin 으로 구분해야 비용 집계에는 잡히고 라우터 정확도 집계에서는 빠진다.
        requestLogService.record(RequestLogEntry.builder()
                .requestId(MDC.get(RequestIdFilter.REQUEST_ID))
                .question(ctx.question())
                .conversationId(ctx.conversationId())
                .tier(result.tier())
                .modelId(result.modelId())
                .routerModelVersion(ctx.routerModelVersion())
                .embedModelVersion(embedModelVersion)
                .cacheHit(false)
                .cacheEntryId(ctx.cacheEntryId())
                .inputTokens(result.inputTokens())
                .outputTokens(result.outputTokens())
                .latencyMs(latency)
                .parentRequestId(originalRequestId)
                .origin(RequestLogEntry.ORIGIN_ESCALATE)
                .build());

        return EscalateResponse.builder()
                // 원본 requestId 를 돌려준다. 프론트가 같은 말풍선에서 한 번 더 올릴 수 있어야 한다.
                .requestId(originalRequestId)
                .response(result.text())
                .initialModel(RouterService.toModelName(initialTier))
                .finalModel(RouterService.toModelName(result.tier()))
                .escalationPath(path)
                .latency(latency)
                .tier(result.tier())
                .modelId(result.modelId())
                .inputTokens(result.inputTokens())
                .outputTokens(result.outputTokens())
                .build();
    }

    /**
     * 자리를 잡는다. 이미 누가 잡고 있으면 상황에 따라 셋 중 하나다.
     *
     * <ul>
     *   <li>이미 완료된 요청 → 저장된 답을 그대로 돌려준다 (멱등, 200). LLM 을 다시 부르지 않는다.
     *   <li>진행 중 → 409. 앞선 요청이 곧 답을 줄 것이다.
     *   <li>유령 예약(죽은 프로세스가 남김) → 치우고 한 번만 재시도한다.
     * </ul>
     *
     * @return 멱등 반환할 응답이 있으면 그것, 예약에 성공했으면 empty
     */
    /** 예약 결과. 둘 중 하나만 채워진다 — 멱등 반환할 응답이거나, 새로 잡은 행의 id 다. */
    private record ReservationOutcome(EscalateResponse replay, long escalationLogId) {}

    private ReservationOutcome reserve(EscalationContext ctx, String requestId,
                                       String initialTier, List<String> path,
                                       Tier targetTier, int seq) {
        EscalationReservation reservation = EscalationReservation.builder()
                .requestId(requestId)
                .question(ctx.question())
                .conversationId(ctx.conversationId())
                .initialTier(initialTier)
                .escalationPath(path)
                .finalTier(targetTier.label())
                .escalationSeq(seq)
                // 최초 판정을 내린 라우터 버전을 남긴다. 원본이 캐시 히트라 classify 를 안 불렀으면
                // null 이므로 현재 버전으로 채운다 — ML 이 이 값이 비면 행을 거부하기 때문이다.
                // (그런 행은 was_cache_hit=true 라 어차피 내보내기에서 빠진다)
                .routerModelVersion(ctx.routerModelVersion() != null
                        ? ctx.routerModelVersion()
                        : routerClient.routerModelVersion())
                .wasCacheHit(ctx.cacheHit())
                .cacheEntryId(ctx.cacheEntryId())
                .similarity(ctx.similarity())
                .build();

        try {
            return new ReservationOutcome(null, escalationLogService.reserve(reservation));
        } catch (DuplicateKeyException first) {
            Optional<ExistingEscalation> existing =
                    escalationLogService.findByTier(requestId, targetTier.label());

            if (existing.isPresent() && existing.get().completed()) {
                log.info("이미 처리된 에스컬레이션이라 저장된 답을 돌려준다 requestId={} tier={}",
                        requestId, targetTier.label());
                return new ReservationOutcome(toResponse(requestId, existing.get()), 0L);
            }

            boolean reclaimed = escalationLogService.deleteStaleReservation(requestId, reservationTimeout);
            if (!reclaimed) {
                throw new EscalationInProgressException(
                        "같은 요청이 처리 중입니다. requestId=" + requestId + " tier=" + targetTier.label());
            }

            // 유령 예약을 치웠으니 딱 한 번 다시 시도한다. 또 실패하면 진짜 경쟁이므로 409.
            try {
                return new ReservationOutcome(null, escalationLogService.reserve(reservation));
            } catch (DuplicateKeyException second) {
                throw new EscalationInProgressException(
                        "같은 요청이 처리 중입니다. requestId=" + requestId);
            }
        }
    }

    /** 저장된 결과로 응답을 재구성한다. 새 LLM 호출도 request_log 행도 만들지 않는다 — 요금이 없었으므로. */
    private EscalateResponse toResponse(String requestId, ExistingEscalation e) {
        return EscalateResponse.builder()
                .requestId(requestId)
                .response(e.response())
                .initialModel(RouterService.toModelName(e.initialTier()))
                .finalModel(RouterService.toModelName(e.finalTier()))
                .escalationPath(e.escalationPath())
                .latency(e.latencyMs())
                .tier(e.finalTier())
                .modelId(e.finalModelId())
                .inputTokens(e.inputTokens())
                .outputTokens(e.outputTokens())
                .build();
    }

    /**
     * 기록 실패로 응답을 실패시키지 않는다 — 이미 Opus 값을 지불했고 답변은 사용자에게 줘야 한다.
     *
     * 다만 여기서는 로그가 곧 제품이라 RequestLogService 보다 의식이 무겁다.
     * 실패하면 이벤트 내용을 한 줄로 남겨 Cloud Logging 에서 수동 복구할 수 있게 한다.
     */
    private void completeReservation(String requestId, long escalationLogId, LlmResult result,
                                     float[] embedding, long latency) {
        try {
            escalationLogService.complete(EscalationCompletion.builder()
                    .requestId(requestId)
                    .escalationLogId(escalationLogId)
                    .response(result.text())
                    .embedding(embedding)
                    .finalModelId(result.modelId())
                    .inputTokens(result.inputTokens())
                    .outputTokens(result.outputTokens())
                    .latencyMs(latency)
                    .build());
        } catch (Exception e) {
            log.error("escalation_log 완료 처리 실패 - 재학습 라벨이 유실된다. "
                            + "requestId={} id={} final={} model={} tokens={}/{}",
                    requestId, escalationLogId, result.tier(), result.modelId(),
                    result.inputTokens(), result.outputTokens(), e);
        }
    }

    private Tier parseModel(String modelName, String field) {
        return RouterService.toTier(modelName).orElseThrow(() -> new InvalidEscalationException(
                field + " 값이 올바르지 않습니다: " + modelName + " (haiku | sonnet | opus)"));
    }

    private void refreshCache(EscalationContext ctx, float[] embedding, LlmResult result,
                              String embedModelVersion) {
        if (!cacheRefreshEnabled) {
            return;
        }

        if (ctx.cacheEntryId() == null) {
            // 원본 요청이 캐시에 행을 남기지 못했다. 만족한 답으로 새로 만든다.
            cacheService.upsertForQuestion(ctx.question(), embedding, result,
                    embedModelVersion, overwriteSimilarity);
            return;
        }

        boolean ourOwnRow = !ctx.cacheHit();
        boolean effectivelySameQuestion =
                ctx.similarity() != null && ctx.similarity() >= overwriteSimilarity;

        if (ourOwnRow || effectivelySameQuestion) {
            cacheService.refresh(ctx.cacheEntryId(), result);
        } else {
            // 유사하지만 다른 질문의 행이다. 덮어쓰지 않고 이 질문 자신의 행을 만든다.
            //
            // put 이 아니라 upsert 인 이유: ctx 는 매 단계 원본 request_log 를 다시 읽어
            // cacheHit/similarity 가 변하지 않는다. 그냥 put 하면 2단계 체인에서
            // 같은 임베딩의 행이 둘 생기고, 동점 상황에서 거부당한 하위 모델 답이
            // 상위 모델 답을 제치고 나갈 수 있다.
            //
            // 원래 히트한 항목의 threshold 를 올려 false-hit 을 줄이는 것은
            // 적응형 임계값 Level 3 이며 여기서는 의도적으로 하지 않는다
            // (Decision Log 2026-08-03 은 Level 2 까지만 채택).
            log.info("캐시 히트 유래이고 유사도 {} < {} 이라 기존 행을 덮어쓰지 않는다 id={}",
                    ctx.similarity(), overwriteSimilarity, ctx.cacheEntryId());
            cacheService.upsertForQuestion(ctx.question(), embedding, result,
                    embedModelVersion, overwriteSimilarity);
        }
    }

    /**
     * 누적 경로를 만든다. 첫 단계면 [현재모델, 목표모델], 이후에는 앞 경로 + 목표모델.
     * 04 API Contract 의 예시 {@code ["haiku","sonnet"]} 처럼 출발점을 포함한다.
     */
    private List<String> buildPath(Optional<EscalationChainState> chain, Tier currentTier, Tier targetTier) {
        List<String> path = new ArrayList<>(
                chain.map(EscalationChainState::escalationPath)
                        .filter(p -> !p.isEmpty())
                        .orElseGet(() -> List.of(RouterService.toModelName(currentTier.label()))));
        path.add(RouterService.toModelName(targetTier.label()));
        return List.copyOf(path);
    }
}
