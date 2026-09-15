package com.llmrouter.service;

import com.llmrouter.client.RetrainingClient;
import com.llmrouter.client.RetrainingClientException;
import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.EscalationEvent;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RouterTier;
import com.llmrouter.repository.FeedbackBatchRepository;
import com.llmrouter.repository.FeedbackBatchRepository.EscalationRow;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;

/**
 * 에스컬레이션 정정 라벨을 FastAPI 재학습에 넘긴다 (04 API Contract).
 *
 * <h2>순서가 설계의 핵심이다</h2>
 *
 * <pre>
 *   1. 대상 행을 고르고 exported_batch_id 를 찍는다  ─┐ 한 트랜잭션
 *   2. feedback_batch 행을 만든다                    ─┘
 *   3. HTTP 로 제출한다
 *   4. 202 응답을 feedback_batch 에 채운다
 * </pre>
 *
 * <p><b>왜 제출보다 먼저 찍는가.</b> 제출이 타임아웃으로 끊겨도 FastAPI 는 이미 접수했을 수
 * 있다. 그때 "실패했으니 다시 고르자"고 하면 같은 이벤트가 새 batchId 로 또 나가고,
 * 계약상 그건 <b>다른 배치</b>라서 모델 버전이 한 번 더 올라간다. 되돌릴 수 없는 종류의
 * 실수다. 먼저 찍어 두면 재시도 때 {@code exported_batch_id} 로 같은 이벤트 집합을
 * 복원해 <b>같은 batchId</b> 로 다시 보낼 수 있다. 계약이 "재시도 때문에 새 batch ID 를
 * 만들거나 payload 를 나누지 않는다"고 한 것이 이 뜻이다.
 *
 * <p>대가는 명확하다. 비재시도 오류(422 등)로 끝나면 찍어 둔 행을 되돌려야 하고,
 * 그건 {@link FeedbackBatchRepository#release} 가 한다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FeedbackBatchService {

    private static final DateTimeFormatter DAY =
            DateTimeFormatter.ofPattern("yyyyMMdd").withZone(ZoneOffset.UTC);

    private final FeedbackBatchRepository repository;
    private final RetrainingClient retrainingClient;
    private final Clock clock;

    @Value("${llm.retraining.batch-size:1000}")
    private int batchSize;

    /**
     * 이 배포가 실제 PII 제거를 요구하는가.
     *
     * <p>우리가 보내는 {@code piiRedacted} 는 항상 true 지만, 그것은 "제거를 마쳤다"가
     * 아니라 <b>"PII 가 유입되지 않는 경로다"</b>라는 선언이다. 실사용자가 없는 실험
     * 서비스이고 개인정보가 담긴 질문을 넣지 않는다는 전제로 운영하기 때문이다.
     * 계약상 이 boolean 은 <b>Spring 의 완료 선언</b>이고 FastAPI 는 값을 믿기만 한다.
     *
     * <p>이 값을 true 로 두면 제출이 <b>거부된다</b>. 전제가 깨지는 상황(외부 사용자 공개,
     * 실제 상담 로그 투입)에서 스위치 하나로 유출을 막기 위한 안전장치다. 실제 탐지·마스킹을
     * 구현하면 그때 이 검사를 redactor 호출로 바꾼다.
     *
     * <p>아무 동작도 하지 않는 설정값을 두지 않는 이유는, 그런 값은 누군가 켜 놓고
     * 보호받는다고 믿게 만들기 때문이다.
     */
    @Value("${llm.retraining.require-pii-redaction:false}")
    private boolean requirePiiRedaction;

    /** 제출 대기 중인 이벤트 수. 대시보드와 자동 트리거가 같이 본다. */
    public int pendingEventCount() {
        return repository.countEligible();
    }

    public List<FeedbackBatchRepository.BatchSummary> recentBatches(int limit) {
        return repository.recentBatches(limit);
    }

    /**
     * 대기 중인 이벤트를 배치로 묶어 제출한다.
     *
     * @param trigger {@code auto}(임계값 도달) 또는 {@code manual}(강제 실행 버튼)
     * @return 제출 시도 결과. 보낼 이벤트가 없으면 비어 있다.
     */
    public Optional<SubmissionResult> submitPending(String trigger) {
        guardPiiRedaction();
        List<EscalationRow> rows = repository.findEligible(batchSize);
        if (rows.isEmpty()) {
            log.info("재학습 제출 대상이 없습니다 (trigger={})", trigger);
            return Optional.empty();
        }
        String batchId = repository.claim(
                rows.stream().map(EscalationRow::id).toList(),
                trigger,
                "feedback-" + DAY.format(Instant.now(clock)) + "-");
        return Optional.of(submitClaimed(batchId, rows));
    }

    /**
     * 접수되지 않은 채 남은 배치를 <b>같은 batchId·같은 payload</b> 로 다시 보낸다.
     *
     * <p>연결이 끊겼을 뿐 FastAPI 가 이미 접수했을 수도 있는데, 같은 내용이면 기존
     * pipelineRunId 로 202 가 다시 온다(계약의 멱등성). 그래서 새로 고르지 않고 복원한다.
     */
    public Optional<SubmissionResult> retrySubmission(String batchId) {
        List<EscalationRow> rows = repository.findByBatch(batchId);
        if (rows.isEmpty()) {
            // 찍어 둔 행이 사라졌다면 payload 를 복원할 수 없다. 새 배치로 만들면
            // 버전을 한 번 더 태우므로 자동으로 처리하지 않고 사람이 보게 남긴다.
            log.error("재시도할 배치의 이벤트가 없습니다 batchId={} — 수동 확인이 필요합니다", batchId);
            return Optional.empty();
        }
        return Optional.of(submitClaimed(batchId, rows));
    }

    /**
     * PII 제거가 필요한 배포인데 구현이 없으면 제출을 막는다.
     *
     * <p>제출은 질문 원문을 외부로 내보내는 유일한 경로다. 여기서 막지 않으면
     * 설정만 바꿔 실사용자에게 연 순간 원문이 그대로 나간다.
     */
    private void guardPiiRedaction() {
        if (requirePiiRedaction) {
            throw new IllegalStateException(
                    "llm.retraining.require-pii-redaction=true 인데 PII 제거가 구현되어 있지 않습니다. "
                            + "질문 원문이 그대로 나가므로 제출을 중단합니다");
        }
    }

    private SubmissionResult submitClaimed(String batchId, List<EscalationRow> rows) {
        List<EscalationEvent> events = rows.stream().map(this::toEvent).toList();
        FeedbackBatchRequest request =
                new FeedbackBatchRequest(batchId, Instant.now(clock), events);

        try {
            EscalateAcceptance acceptance = retrainingClient.submit(request);
            repository.recordAcceptance(acceptance);
            log.info("재학습 배치 접수됨 batchId={} events={} candidate={}",
                    batchId, events.size(), acceptance.candidateModelVersion());
            return new SubmissionResult(batchId, events.size(), true, null);

        } catch (RetrainingClientException error) {
            if (error.isRetryable()) {
                // 배치 행과 exported_batch_id 를 그대로 둔다. 스케줄러가 같은 payload 로 다시 온다.
                log.warn("재학습 제출 실패(재시도 대상) batchId={} code={}", batchId, error.code());
                return new SubmissionResult(batchId, events.size(), false, error.code());
            }
            // 되돌리지 않으면 이 이벤트들이 영영 어떤 배치에도 못 들어간다.
            repository.release(batchId, error.code());
            log.error("재학습 제출 실패(복구 불가) batchId={} code={} — 이벤트를 대기 상태로 되돌렸습니다",
                    batchId, error.code());
            return new SubmissionResult(batchId, events.size(), false, error.code());
        }
    }

    private EscalationEvent toEvent(EscalationRow row) {
        String[] path = row.escalationPath();
        return new EscalationEvent(
                "esc-" + row.id(),
                row.requestId(),
                row.question(),
                RouterTier.fromLabel(row.initialTier()),
                // initialModel 과 finalModel 을 둘 다 escalation_path 에서 뽑는다.
                // final 만 final_model_id(claude-sonnet-4-6)로 쓰면 initial(haiku)과
                // 표기 단위가 달라져 ML 쪽에서 같은 모델이 두 이름으로 보인다.
                path[0],
                RouterTier.fromLabel(row.finalTier()),
                path[path.length - 1],
                true,                   // 선별 쿼리가 user_satisfied = true 만 가져온다
                row.escalationSeq(),
                row.createdAt(),
                row.routerModelVersion(),
                true);                  // requirePiiRedaction 주석 참고
    }

    /**
     * @param accepted 202 를 받았다는 뜻이지 <b>학습이 성공했다는 뜻이 아니다.</b>
     *                 결과는 폴링으로만 확정된다.
     */
    public record SubmissionResult(String batchId, int eventCount, boolean accepted,
                                   String failureCode) {
    }
}
