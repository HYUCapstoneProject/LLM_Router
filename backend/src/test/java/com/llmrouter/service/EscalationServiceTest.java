package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class EscalationServiceTest {

    private static final String REQ_ID = "3f2a0000-0000-4000-8000-000000000001";
    private static final String QUESTION = "저장된 원본 질문";

    @Mock private RouterClient routerClient;
    @Mock private CacheService cacheService;
    @Mock private LlmService llmService;
    @Mock private RequestLogService requestLogService;
    @Mock private EscalationLogService escalationLogService;

    @InjectMocks private EscalationService service;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(service, "cacheRefreshEnabled", true);
        ReflectionTestUtils.setField(service, "overwriteSimilarity", 0.95);
        ReflectionTestUtils.setField(service, "reservationTimeout", Duration.ofMinutes(5));

        when(routerClient.embed(anyString())).thenReturn(new float[]{0.1f, 0.2f});
        when(routerClient.embedModelVersion()).thenReturn("stub-hash-v0");
        when(routerClient.routerModelVersion()).thenReturn("stub-medium-v0");
        when(escalationLogService.loadChain(anyString())).thenReturn(Optional.empty());
        when(escalationLogService.findByTier(anyString(), anyString())).thenReturn(Optional.empty());
        when(escalationLogService.reserve(any())).thenReturn(1001L);
        when(llmService.call(any(), anyString()))
                .thenReturn(new LlmResult("상위 모델 답변", "hard", "claude-opus-4-8", 200, 900));
    }

    private EscalateRequest request(String current, String target) {
        EscalateRequest r = new EscalateRequest();
        ReflectionTestUtils.setField(r, "requestId", REQ_ID);
        ReflectionTestUtils.setField(r, "message", "클라이언트가 보낸 질문");
        ReflectionTestUtils.setField(r, "currentModel", current);
        ReflectionTestUtils.setField(r, "targetModel", target);
        return r;
    }

    /** 캐시 미스로 만들어진 원본 (우리가 만든 캐시 행이 있음) */
    private void givenMissOriginContext(String tier) {
        when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                new EscalationContext(QUESTION, "conv-1", tier, false, 42L, null,
                        "router-v1", "stub-hash-v0")));
    }

    private EscalationReservation captureReservation() {
        ArgumentCaptor<EscalationReservation> captor =
                ArgumentCaptor.forClass(EscalationReservation.class);
        verify(escalationLogService).reserve(captor.capture());
        return captor.getValue();
    }

    private EscalationCompletion captureCompletion() {
        ArgumentCaptor<EscalationCompletion> captor =
                ArgumentCaptor.forClass(EscalationCompletion.class);
        verify(escalationLogService).complete(captor.capture());
        return captor.getValue();
    }

    @Nested
    @DisplayName("티어 방향 검증")
    class TierDirection {

        @Test
        @DisplayName("더 낮은 티어로는 내려갈 수 없다")
        void rejectsDowngrade() {
            givenMissOriginContext("hard");

            assertThatThrownBy(() -> service.escalate(request("opus", "sonnet")))
                    .isInstanceOf(InvalidEscalationException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("같은 티어로도 갈 수 없다")
        void rejectsSameTier() {
            givenMissOriginContext("hard");

            assertThatThrownBy(() -> service.escalate(request("opus", "opus")))
                    .isInstanceOf(InvalidEscalationException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("모르는 모델 이름은 거부한다")
        void rejectsUnknownModel() {
            givenMissOriginContext("medium");

            assertThatThrownBy(() -> service.escalate(request("sonnet", "gpt-4")))
                    .isInstanceOf(InvalidEscalationException.class);
        }

        @Test
        @DisplayName("방향 검증이 DB 조회보다 먼저다 - 잘못된 요청에 쿼리를 낭비하지 않는다")
        void validatesBeforeLoadingContext() {
            assertThatThrownBy(() -> service.escalate(request("opus", "haiku")))
                    .isInstanceOf(InvalidEscalationException.class);
            verify(escalationLogService, never()).loadContext(anyString());
        }
    }

    @Nested
    @DisplayName("원본 맥락")
    class Context {

        @Test
        @DisplayName("원본 요청이 없으면 404 로 끊는다")
        void throwsWhenOriginMissing() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> service.escalate(request("sonnet", "opus")))
                    .isInstanceOf(EscalationNotFoundException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("LLM 에는 클라이언트가 보낸 문장이 아니라 저장된 질문을 보낸다")
        void usesStoredQuestionNotClientMessage() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            verify(llmService).call(Tier.HARD, QUESTION);
        }
    }

    @Nested
    @DisplayName("예약 (이중 과금 방지)")
    class Reservation {

        @Test
        @DisplayName("예약이 LLM 호출보다 먼저 일어난다")
        void reservesBeforeCallingLlm() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            org.mockito.InOrder order = org.mockito.Mockito.inOrder(escalationLogService, llmService);
            order.verify(escalationLogService).reserve(any());
            order.verify(llmService).call(any(), anyString());
        }

        @Test
        @DisplayName("완료된 요청의 재시도는 체인 말단 검사보다 먼저 걸러 저장된 답을 돌려준다")
        void replaysCompletedEscalationBeforeChainCheck() {
            givenMissOriginContext("medium");
            when(escalationLogService.findByTier(REQ_ID, "hard")).thenReturn(Optional.of(
                    new ExistingEscalation(1, "먼저 받은 opus 답변", "medium", "hard",
                            "claude-opus-4-8", List.of("sonnet", "opus"), 200, 900, 1500)));
            // 이미 opus 까지 올라가 있어, 멱등 조회가 없으면 체인 검사에서 409 로 걸린다
            when(escalationLogService.loadChain(REQ_ID)).thenReturn(Optional.of(
                    new EscalationChainState(1, "medium", "hard", List.of("sonnet", "opus"))));

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getResponse()).isEqualTo("먼저 받은 opus 답변");
            assertThat(response.getEscalationPath()).containsExactly("sonnet", "opus");
            assertThat(response.getInputTokens()).isEqualTo(200);
            verify(llmService, never()).call(any(), anyString());
            verify(escalationLogService, never()).reserve(any());
        }

        @Test
        @DisplayName("멱등 조회와 예약 사이에 앞선 요청이 끝나도 저장된 답으로 수렴한다")
        void replaysWhenCompletionLandsDuringReservation() {
            givenMissOriginContext("medium");
            // 멱등 조회 시점엔 아직 없었는데, 예약 시점엔 앞선 요청이 끝나 있는 경쟁
            when(escalationLogService.findByTier(REQ_ID, "hard"))
                    .thenReturn(Optional.empty())
                    .thenReturn(Optional.of(new ExistingEscalation(1, "먼저 받은 opus 답변",
                            "medium", "hard", "claude-opus-4-8",
                            List.of("sonnet", "opus"), 200, 900, 1500)));
            when(escalationLogService.reserve(any()))
                    .thenThrow(new DuplicateKeyException("uq_escalation_log_tier"));

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getResponse()).isEqualTo("먼저 받은 opus 답변");
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("멱등 반환에는 request_log 를 남기지 않는다 - 요금이 나가지 않았다")
        void idempotentReplayDoesNotDoubleCountCost() {
            givenMissOriginContext("medium");
            when(escalationLogService.reserve(any())).thenThrow(new DuplicateKeyException("dup"));
            when(escalationLogService.findByTier(REQ_ID, "hard")).thenReturn(Optional.of(
                    new ExistingEscalation(1, "답", "medium", "hard", "claude-opus-4-8",
                            List.of("sonnet", "opus"), 200, 900, 1500)));

            service.escalate(request("sonnet", "opus"));

            verify(requestLogService, never()).record(any());
            verify(cacheService, never()).refresh(anyLong(), any());
        }

        @Test
        @DisplayName("아직 진행 중인 예약이면 409 를 내고 LLM 을 부르지 않는다")
        void inFlightReservationYieldsConflict() {
            givenMissOriginContext("medium");
            when(escalationLogService.reserve(any())).thenThrow(new DuplicateKeyException("dup"));
            when(escalationLogService.findByTier(REQ_ID, "hard")).thenReturn(Optional.of(
                    new ExistingEscalation(1, null, "medium", "hard", null,
                            List.of("sonnet", "opus"), 0, 0, 0)));
            when(escalationLogService.deleteStaleReservation(anyString(), any()))
                    .thenReturn(false);

            assertThatThrownBy(() -> service.escalate(request("sonnet", "opus")))
                    .isInstanceOf(EscalationInProgressException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("죽은 프로세스가 남긴 유령 예약은 치우고 한 번 재시도한다")
        void takesOverStaleReservation() {
            givenMissOriginContext("medium");
            when(escalationLogService.findByTier(REQ_ID, "hard")).thenReturn(Optional.of(
                    new ExistingEscalation(1, null, "medium", "hard", null,
                            List.of("sonnet", "opus"), 0, 0, 0)));
            when(escalationLogService.deleteStaleReservation(eq(REQ_ID), any()))
                    .thenReturn(true);
            // 첫 예약은 유령 때문에 실패, 치운 뒤 두 번째는 성공
            when(escalationLogService.reserve(any()))
                    .thenThrow(new DuplicateKeyException("dup"))
                    .thenReturn(2002L);

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getResponse()).isEqualTo("상위 모델 답변");
            verify(escalationLogService, org.mockito.Mockito.times(2)).reserve(any());
            verify(llmService).call(any(), anyString());
        }

        @Test
        @DisplayName("유령을 치웠는데도 또 충돌하면 진짜 경쟁이므로 409")
        void realRaceAfterTakeoverYieldsConflict() {
            givenMissOriginContext("medium");
            when(escalationLogService.findByTier(REQ_ID, "hard")).thenReturn(Optional.of(
                    new ExistingEscalation(1, null, "medium", "hard", null, List.of(), 0, 0, 0)));
            when(escalationLogService.deleteStaleReservation(anyString(), any()))
                    .thenReturn(true);
            when(escalationLogService.reserve(any())).thenThrow(new DuplicateKeyException("dup"));

            assertThatThrownBy(() -> service.escalate(request("sonnet", "opus")))
                    .isInstanceOf(EscalationInProgressException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("LLM 호출이 실패하면 예약을 풀어 다시 시도할 수 있게 한다")
        void releasesReservationWhenLlmFails() {
            givenMissOriginContext("medium");
            when(llmService.call(any(), anyString())).thenThrow(new RuntimeException("Claude 500"));

            assertThatThrownBy(() -> service.escalate(request("sonnet", "opus")))
                    .isInstanceOf(RuntimeException.class);

            verify(escalationLogService).release(1001L);
            verify(escalationLogService, never()).complete(any());
        }
    }

    @Nested
    @DisplayName("체인")
    class Chain {

        @Test
        @DisplayName("첫 단계는 seq=1 이고 경로에 출발 모델이 포함된다")
        void firstStep() {
            givenMissOriginContext("medium");

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getEscalationPath()).containsExactly("sonnet", "opus");
            assertThat(response.getInitialModel()).isEqualTo("sonnet");
            assertThat(response.getFinalModel()).isEqualTo("opus");
            // 원본 chat 의 requestId 를 그대로 돌려줘야 프론트가 또 올릴 수 있다
            assertThat(response.getRequestId()).isEqualTo(REQ_ID);

            EscalationReservation reserved = captureReservation();
            assertThat(reserved.escalationSeq()).isEqualTo(1);
            assertThat(reserved.initialTier()).isEqualTo("medium");
            assertThat(reserved.finalTier()).isEqualTo("hard");
        }

        @Test
        @DisplayName("두 번째 단계는 경로를 누적하고 initial_tier 는 체인 최초값을 유지한다")
        void secondStepAccumulatesPath() {
            // 원본은 easy 로 판정됐고 이미 haiku -> sonnet 을 밟았다
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "easy", false, 42L, null,
                            "router-v1", "stub-hash-v0")));
            when(escalationLogService.loadChain(REQ_ID)).thenReturn(Optional.of(
                    new EscalationChainState(1, "easy", "medium", List.of("haiku", "sonnet"))));

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getEscalationPath()).containsExactly("haiku", "sonnet", "opus");
            // 직전 단계(sonnet)가 아니라 체인의 출발점(haiku)이어야 한다
            assertThat(response.getInitialModel()).isEqualTo("haiku");

            EscalationReservation reserved = captureReservation();
            assertThat(reserved.escalationSeq()).isEqualTo(2);
            assertThat(reserved.initialTier()).isEqualTo("easy");
        }

        @Test
        @DisplayName("화면이 낡아 currentModel 이 체인 말단과 다르면 409")
        void rejectsStaleCurrentModel() {
            givenMissOriginContext("easy");
            when(escalationLogService.loadChain(REQ_ID)).thenReturn(Optional.of(
                    new EscalationChainState(1, "easy", "medium", List.of("haiku", "sonnet"))));

            // 이미 sonnet 까지 올라갔는데 클라이언트는 아직 haiku 화면을 보고 있다
            assertThatThrownBy(() -> service.escalate(request("haiku", "sonnet")))
                    .isInstanceOf(EscalationConflictException.class);
            verify(llmService, never()).call(any(), anyString());
        }

        @Test
        @DisplayName("완료 처리가 실패해도 답변은 반환한다 - 이미 요금을 지불했다")
        void completionFailureDoesNotFailResponse() {
            givenMissOriginContext("medium");
            doThrow(new RuntimeException("DB down")).when(escalationLogService).complete(any());

            EscalateResponse response = service.escalate(request("sonnet", "opus"));

            assertThat(response.getResponse()).isEqualTo("상위 모델 답변");
        }
    }

    @Nested
    @DisplayName("캐시 갱신 (기획서 4-3)")
    class CacheRefresh {

        @Test
        @DisplayName("우리가 만든 행(캐시 미스 유래)은 그대로 덮어쓴다")
        void refreshesOwnRow() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            verify(cacheService).refresh(eq(42L), any());
            verify(cacheService, never()).upsertForQuestion(anyString(), any(), any(), anyDouble());
        }

        @Test
        @DisplayName("캐시 히트 유래여도 유사도가 임계값 이상이면 사실상 같은 질문이라 덮어쓴다")
        void refreshesHighSimilarityHit() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "medium", true, 42L, 0.97,
                            "router-v1", "stub-hash-v0")));

            service.escalate(request("sonnet", "opus"));

            verify(cacheService).refresh(eq(42L), any());
        }

        @Test
        @DisplayName("캐시 히트 유래이고 유사도가 낮으면 남의 질문이라 덮어쓰지 않고 새 행을 만든다")
        void createsNewRowForLowSimilarityHit() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "medium", true, 42L, 0.83,
                            "router-v1", "stub-hash-v0")));

            service.escalate(request("sonnet", "opus"));

            verify(cacheService, never()).refresh(anyLong(), any());
            // put 이 아니라 upsert 다. 2단계 체인에서 같은 임베딩의 행이 둘 생기면
            // 동점 상황에서 거부당한 하위 모델 답이 나갈 수 있다.
            verify(cacheService).upsertForQuestion(eq(QUESTION), any(), any(), eq(0.95));
        }

        @Test
        @DisplayName("원본이 캐시에 행을 남기지 못했으면 새로 만든다")
        void createsRowWhenOriginHadNone() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "medium", false, null, null,
                            "router-v1", "stub-hash-v0")));

            service.escalate(request("sonnet", "opus"));

            verify(cacheService).upsertForQuestion(eq(QUESTION), any(), any(), eq(0.95));
            verify(cacheService, never()).refresh(anyLong(), any());
        }

        @Test
        @DisplayName("설정으로 끄면 캐시를 건드리지 않는다")
        void disabledSkipsCache() {
            ReflectionTestUtils.setField(service, "cacheRefreshEnabled", false);
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            verify(cacheService, never()).refresh(anyLong(), any());
            verify(cacheService, never()).upsertForQuestion(anyString(), any(), any(), anyDouble());
        }
    }

    @Nested
    @DisplayName("기록")
    class Logging {

        @Test
        @DisplayName("캐시 false-hit 판별 정보를 그대로 넘긴다 (Decision Log 2026-08-03)")
        void carriesCacheProvenance() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "medium", true, 42L, 0.83,
                            "router-v1", "stub-hash-v0")));

            service.escalate(request("sonnet", "opus"));

            EscalationReservation reserved = captureReservation();
            assertThat(reserved.wasCacheHit()).isTrue();
            assertThat(reserved.cacheEntryId()).isEqualTo(42L);
            assertThat(reserved.similarity()).isEqualTo(0.83);
        }

        @Test
        @DisplayName("최초 판정을 내린 라우터 버전을 남긴다 - ML 이 빈 값이면 행을 거부한다")
        void keepsOriginalRouterVersion() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            assertThat(captureReservation().routerModelVersion()).isEqualTo("router-v1");
        }

        @Test
        @DisplayName("원본이 캐시 히트라 라우터 버전이 없으면 현재 버전으로 채운다")
        void fallsBackToCurrentRouterVersion() {
            when(escalationLogService.loadContext(REQ_ID)).thenReturn(Optional.of(
                    new EscalationContext(QUESTION, "conv-1", "medium", true, 42L, 0.99,
                            null, "stub-hash-v0")));

            service.escalate(request("sonnet", "opus"));

            assertThat(captureReservation().routerModelVersion()).isEqualTo("stub-medium-v0");
        }

        @Test
        @DisplayName("완료 시 응답과 토큰을 채운다")
        void completionCarriesResult() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            EscalationCompletion completion = captureCompletion();
            assertThat(completion.escalationLogId()).isEqualTo(1001L);
            assertThat(completion.response()).isEqualTo("상위 모델 답변");
            assertThat(completion.finalModelId()).isEqualTo("claude-opus-4-8");
            assertThat(completion.inputTokens()).isEqualTo(200);
            assertThat(completion.outputTokens()).isEqualTo(900);
            assertThat(completion.embedding()).isNotNull();
        }

        @Test
        @DisplayName("에스컬레이션 호출도 request_log 에 남겨 비용 집계에 잡히게 한다")
        void recordsCostBearingCall() {
            givenMissOriginContext("medium");

            service.escalate(request("sonnet", "opus"));

            ArgumentCaptor<RequestLogEntry> captor = ArgumentCaptor.forClass(RequestLogEntry.class);
            verify(requestLogService).record(captor.capture());
            RequestLogEntry logged = captor.getValue();

            assertThat(logged.origin()).isEqualTo(RequestLogEntry.ORIGIN_ESCALATE);
            assertThat(logged.parentRequestId()).isEqualTo(REQ_ID);
            assertThat(logged.inputTokens()).isEqualTo(200);
            assertThat(logged.outputTokens()).isEqualTo(900);
            assertThat(logged.cacheHit()).isFalse();
        }
    }
}
