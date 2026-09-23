package com.llmrouter.controller;

import com.llmrouter.repository.FeedbackBatchRepository.BatchSummary;
import com.llmrouter.service.FeedbackBatchService;
import com.llmrouter.service.FeedbackBatchService.SubmissionResult;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Optional;

/**
 * 재학습 피드백 배치의 운영 화면용 API.
 *
 * <p>사용자 기능이 아니라 <b>우리가 시점을 정하기 위한</b> 창구다. 이 서비스는 실사용자가
 * 없어 이벤트가 천천히 쌓이므로, 임계값을 기다리는 대신 시연이나 실험 직전에 사람이
 * 눌러 내보내는 쪽이 현실적이다.
 */
@RestController
@RequestMapping("/api/retraining")
@RequiredArgsConstructor
public class RetrainingController {

    private final FeedbackBatchService feedbackBatchService;

    /** 대기 이벤트 수와 최근 배치 이력. 대시보드가 이 하나로 화면을 그린다. */
    @GetMapping("/status")
    public RetrainingOverview status(@RequestParam(defaultValue = "10") int limit) {
        return new RetrainingOverview(
                feedbackBatchService.pendingEventCount(),
                feedbackBatchService.recentBatches(Math.clamp(limit, 1, 50)));
    }

    /**
     * 대기 중인 이벤트를 지금 제출한다 (강제 실행 버튼).
     *
     * <p><b>누를 때마다 팀원 쪽 모델 버전이 한 칸 올라간다.</b> 계약상 승격까지 성공한
     * 고유 batch 3회가 한 major cycle이라, 시험 삼아 반복해서 누르면 버전이 소모된다.
     * 되돌릴 방법이 없으므로 프론트에서 확인 절차를 한 번 거치게 해 두었다.
     *
     * <p>응답의 {@code accepted=true} 는 <b>202 를 받았다</b>는 뜻이지 학습이 성공했다는
     * 뜻이 아니다. 결과는 이후 폴링이 채운다.
     *
     * @return 보낼 이벤트가 없으면 204
     */
    @PostMapping("/submit")
    public ResponseEntity<SubmissionResult> submit() {
        Optional<SubmissionResult> result = feedbackBatchService.submitPending("manual");
        return result.map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.noContent().build());
    }

    public record RetrainingOverview(int pendingEvents, List<BatchSummary> recentBatches) {
    }
}
