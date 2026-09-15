package com.llmrouter.client;

import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RetrainingStatus;
import com.llmrouter.dto.RouterModelStatus;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * FastAPI 를 부르지 않고 재학습 경로를 개발하기 위한 구현.
 *
 * <p><b>이 stub 이 필요한 이유는 다른 stub 들과 다르다.</b> embed/classify 는 여러 번
 * 불러도 남는 게 없지만, 재학습 제출은 <b>성공한 배치 하나가 팀원 쪽 모델 버전을
 * 한 칸 올린다</b>(계약: 승격까지 성공한 고유 HTTP batch 3회가 한 major cycle).
 * 개발 중에 batchId 를 바꿔가며 눌러보면 그때마다 실제 Vertex 파이프라인이 돌고
 * 버전이 소모된다. 되돌릴 방법이 없다.
 *
 * <p>그래서 로컬·개발에서는 이쪽이 붙고, 실연동은 {@code ROUTER_MODE=fastapi} 로
 * 명시적으로 전환할 때만 일어난다.
 *
 * <p>동작: 제출을 메모리에 기록하고 폴링할 때마다 단계를 한 칸씩 진행시켜
 * 결국 {@code succeeded} 로 끝낸다. 실패·거부 분기를 시험하려면 batchId 에
 * {@code -fail} 또는 {@code -reject} 를 붙인다.
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "llm.router.mode", havingValue = "stub", matchIfMissing = true)
public class StubRetrainingClient implements RetrainingClient {

    private static final String[] STAGES = {
            "initializing", "feedbackPreparation", "feedbackMerge",
            "training", "evaluation", "packaging", "gate", "publishing", "complete"
    };

    private final Map<String, AtomicInteger> polls = new ConcurrentHashMap<>();

    @Override
    public EscalateAcceptance submit(FeedbackBatchRequest request) {
        log.info("[stub] 피드백 배치 접수 batchId={} events={}",
                request.batchId(), request.events().size());
        polls.put(request.batchId(), new AtomicInteger());
        return new EscalateAcceptance(
                "escalation-acceptance-v1",
                "accepted",
                request.batchId(),
                "projects/stub/locations/us-west1/pipelineJobs/" + request.batchId(),
                "0.0",
                "router-modernbert-stub-" + request.batchId(),
                "minor_fine_tune",
                "successfulHttpRequest",
                "/retraining-runs/" + request.batchId());
    }

    @Override
    public RetrainingStatus status(String batchId) {
        AtomicInteger counter = polls.get(batchId);
        if (counter == null) {
            throw new RetrainingClientException(
                    "[stub] 모르는 batchId", 404, "RETRAINING_RUN_NOT_FOUND");
        }
        int step = counter.getAndIncrement();
        if (step < STAGES.length - 1) {
            return new RetrainingStatus("retraining-status-v1", batchId, "running",
                    STAGES[step], false, false, null, null);
        }
        if (batchId.endsWith("-fail")) {
            return new RetrainingStatus("retraining-status-v1", batchId, "failed", "training",
                    true, false,
                    new RetrainingStatus.Result("failure", "TRAINING_FAILED", true),
                    new RetrainingStatus.Activation("notApplicable", null));
        }
        if (batchId.endsWith("-reject")) {
            return new RetrainingStatus("retraining-status-v1", batchId, "rejected", "gate",
                    true, false,
                    new RetrainingStatus.Result("rejected", "DEPLOY_GATE_REJECTED", false),
                    new RetrainingStatus.Activation("notApplicable", null));
        }
        return new RetrainingStatus("retraining-status-v1", batchId, "succeeded", "complete",
                true, false,
                new RetrainingStatus.Result("succeeded", null, null),
                new RetrainingStatus.Activation("active", "router-modernbert-stub-" + batchId));
    }

    @Override
    public RouterModelStatus currentModel() {
        return new RouterModelStatus("static",
                new RouterModelStatus.ModelRef("router-modernbert-stub"),
                new RouterModelStatus.ModelRef("router-modernbert-stub"),
                null);
    }
}
