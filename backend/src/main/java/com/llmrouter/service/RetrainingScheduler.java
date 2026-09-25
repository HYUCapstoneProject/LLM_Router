package com.llmrouter.service;

import com.llmrouter.client.RetrainingClient;
import com.llmrouter.client.RetrainingClientException;
import com.llmrouter.dto.RetrainingStatus;
import com.llmrouter.repository.FeedbackBatchRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 재학습 배치의 주기 작업 — 자동 제출, 미접수 재시도, 상태 폴링.
 *
 * <p><b>자동 제출은 기본이 꺼져 있다.</b> 계약상 성공한 배치 하나가 모델 버전을 한 칸
 * 올리는데, 이 서비스는 실사용자가 없어 이벤트가 천천히 쌓인다. 잠들어 있는 사이
 * 임계값을 넘겨 제출이 나가면 팀원 쪽 파이프라인이 예고 없이 도는 셈이다. 켜는 것은
 * 데이터가 실제로 쌓이기 시작한 뒤의 선택이고, 그 전까지는 대시보드의 강제 실행 버튼으로
 * 사람이 시점을 정한다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RetrainingScheduler {

    private final FeedbackBatchService feedbackBatchService;
    private final FeedbackBatchRepository repository;
    private final RetrainingClient retrainingClient;

    @Value("${llm.retraining.auto-submit:false}")
    private boolean autoSubmit;

    @Value("${llm.retraining.legacy-enabled:true}")
    private boolean legacyEnabled = true;

    @Value("${llm.retraining.batch-size:1000}")
    private int batchSize;

    /**
     * 대기 이벤트가 임계값을 넘으면 자동 제출한다.
     *
     * <p>주기가 5분인 것은 이벤트가 그보다 빨리 쌓일 일이 없기 때문이다. 더 자주 돌아도
     * {@code countEligible()} 만 반복할 뿐이다.
     */
    @Scheduled(fixedDelayString = "${llm.retraining.submit-interval-ms:300000}")
    public void autoSubmit() {
        if (!legacyEnabled || !autoSubmit) {
            return;
        }
        int pending = feedbackBatchService.pendingEventCount();
        if (pending < batchSize) {
            return;
        }
        log.info("대기 이벤트 {}건이 임계값 {}건을 넘어 자동 제출합니다", pending, batchSize);
        feedbackBatchService.submitPending("auto");
    }

    /**
     * 202 를 받지 못한 배치를 같은 batchId·같은 payload 로 다시 보낸다.
     *
     * <p>연결이 끊긴 경우 FastAPI 는 이미 접수했을 수 있다. 같은 내용이면 기존
     * pipelineRunId 로 202 가 다시 오고(계약의 멱등성), 그 시점에 정상 경로로 합류한다.
     */
    @Scheduled(fixedDelayString = "${llm.retraining.retry-interval-ms:60000}")
    public void retryUnaccepted() {
        if (!legacyEnabled) return;
        for (String batchId : repository.findUnaccepted()) {
            log.info("미접수 배치 재제출 batchId={}", batchId);
            feedbackBatchService.retrySubmission(batchId);
        }
    }

    /**
     * 진행 중인 배치의 상태를 조회한다.
     *
     * <p>어느 배치를 지금 볼지는 {@link FeedbackBatchRepository#findDuePolls()} 가
     * 계약의 backoff 로 정한다. 여기서는 10초마다 "차례가 된 것이 있나"만 확인한다.
     */
    @Scheduled(fixedDelayString = "${llm.retraining.poll-interval-ms:10000}")
    public void pollRunning() {
        List<String> due = repository.findDuePolls();
        for (String batchId : due) {
            try {
                RetrainingStatus status = retrainingClient.status(batchId);
                repository.recordStatus(batchId, status);

                if (status.isPollingComplete()) {
                    logTerminal(batchId, status);
                }
            } catch (RetrainingClientException error) {
                // 폴링 실패로 배치 상태를 건드리지 않는다. 조회가 안 됐다는 것과
                // 학습이 실패했다는 것은 다르다. 다음 차례에 다시 본다.
                log.warn("재학습 상태 조회 실패 batchId={} code={}", batchId, error.code());
                repository.touchPolled(batchId);
            }
        }
    }

    private void logTerminal(String batchId, RetrainingStatus status) {
        switch (status.status()) {
            case "succeeded" -> {
                String activation = status.activationStatus();
                // 학습 성공과 serving 활성화 성공은 다른 사건이다.
                // activation 이 active 가 아니면 우리가 받는 분류는 아직 옛 모델의 것이다.
                if ("active".equals(activation)) {
                    log.info("재학습 성공·반영 완료 batchId={} model={}",
                            batchId, status.activation().servingModelVersion());
                } else {
                    log.warn("재학습은 성공했으나 serving 반영이 끝나지 않았습니다 "
                            + "batchId={} activation={}", batchId, activation);
                }
            }
            case "rejected" -> log.warn("재학습이 성능 gate 에서 거부되었습니다 batchId={} code={}",
                    batchId, status.failureCode());
            case "failed" -> log.error("재학습 기술 실패 batchId={} code={}",
                    batchId, status.failureCode());
            case "cancelled" -> log.warn("재학습이 취소되었습니다 batchId={}", batchId);
            default -> log.info("재학습 종료 batchId={} status={}", batchId, status.status());
        }
    }
}
