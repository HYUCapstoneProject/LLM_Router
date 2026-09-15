package com.llmrouter.service;

import com.llmrouter.client.RetrainingClient;
import com.llmrouter.client.RetrainingClientException;
import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RouterTier;
import com.llmrouter.repository.FeedbackBatchRepository;
import com.llmrouter.repository.FeedbackBatchRepository.EscalationRow;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class FeedbackBatchServiceTest {

    @Mock FeedbackBatchRepository repository;
    @Mock RetrainingClient retrainingClient;

    FeedbackBatchService service;

    /** batchId 에 UTC 날짜가 들어가므로 시각을 고정한다. 안 그러면 자정 근처에서만 깨진다. */
    private static final Clock FIXED =
            Clock.fixed(Instant.parse("2026-09-15T04:00:00Z"), ZoneOffset.UTC);

    @BeforeEach
    void setUp() {
        service = new FeedbackBatchService(repository, retrainingClient, FIXED);
        // @Value 필드는 직접 생성한 인스턴스에 주입되지 않는다.
        ReflectionTestUtils.setField(service, "batchSize", 1000);
        ReflectionTestUtils.setField(service, "requirePiiRedaction", false);
    }

    private EscalationRow row(long id) {
        return new EscalationRow(id, "req-" + id, "질문 " + id, "easy", "hard",
                new String[]{"haiku", "sonnet", "opus"}, 2,
                Instant.parse("2026-09-15T03:00:00Z"),
                "router-modernbert-base-routerbench-20260814-45dcc6d");
    }

    private EscalateAcceptance acceptance(String batchId) {
        return new EscalateAcceptance("escalation-acceptance-v1", "accepted", batchId,
                "projects/p/locations/us-west1/pipelineJobs/j", "2.1",
                "router-modernbert-feedback-v2.1", "minor_fine_tune",
                "successfulHttpRequest", "/retraining-runs/" + batchId);
    }

    @Test
    void 보낼_이벤트가_없으면_제출하지_않는다() {
        when(repository.findEligible(anyInt())).thenReturn(List.of());

        assertThat(service.submitPending("manual")).isEmpty();

        // 빈 배치를 보내면 batchId 만 소비하고 모델 버전이 올라간다.
        verifyNoInteractions(retrainingClient);
        verify(repository, never()).claim(any(), anyString(), anyString());
    }

    @Test
    void 계약이_요구하는_12필드를_그대로_보낸다() {
        when(repository.findEligible(anyInt())).thenReturn(List.of(row(7)));
        when(repository.claim(eq(List.of(7L)), eq("manual"), eq("feedback-20260915-")))
                .thenReturn("feedback-20260915-0001");
        when(retrainingClient.submit(any())).thenReturn(acceptance("feedback-20260915-0001"));

        service.submitPending("manual");

        ArgumentCaptor<FeedbackBatchRequest> sent =
                ArgumentCaptor.forClass(FeedbackBatchRequest.class);
        verify(retrainingClient).submit(sent.capture());
        FeedbackBatchRequest request = sent.getValue();

        assertThat(request.batchId()).isEqualTo("feedback-20260915-0001");
        assertThat(request.schemaVersion()).isEqualTo("feedback-inline-v1");
        assertThat(request.events()).hasSize(1);

        var event = request.events().get(0);
        assertThat(event.eventId()).isEqualTo("esc-7");
        assertThat(event.requestId()).isEqualTo("req-7");
        assertThat(event.initialLabel()).isEqualTo(RouterTier.easy);
        assertThat(event.finalLabel()).isEqualTo(RouterTier.hard);
        // 두 모델 이름을 같은 출처(escalation_path)에서 뽑아야 표기 단위가 어긋나지 않는다.
        assertThat(event.initialModel()).isEqualTo("haiku");
        assertThat(event.finalModel()).isEqualTo("opus");
        assertThat(event.userSatisfied()).isTrue();
        assertThat(event.piiRedacted()).isTrue();
        assertThat(event.escalationCount()).isEqualTo(2);
    }

    @Test
    void 재시도_가능한_오류는_찍어둔_행을_되돌리지_않는다() {
        when(repository.findEligible(anyInt())).thenReturn(List.of(row(1)));
        when(repository.claim(any(), anyString(), anyString())).thenReturn("feedback-20260915-0001");
        when(retrainingClient.submit(any())).thenThrow(new RetrainingClientException(
                "학습 중", 409, "RETRAINING_IN_PROGRESS"));

        Optional<FeedbackBatchService.SubmissionResult> result = service.submitPending("auto");

        assertThat(result).isPresent();
        assertThat(result.get().accepted()).isFalse();
        // 되돌리면 다음 시도가 새 batchId 로 나가고 모델 버전이 한 번 더 올라간다.
        verify(repository, never()).release(anyString(), anyString());
    }

    @Test
    void 복구_불가_오류는_행을_대기_상태로_되돌린다() {
        when(repository.findEligible(anyInt())).thenReturn(List.of(row(1)));
        when(repository.claim(any(), anyString(), anyString())).thenReturn("feedback-20260915-0001");
        when(retrainingClient.submit(any())).thenThrow(new RetrainingClientException(
                "계약 위반", 422, "REQUEST_VALIDATION_FAILED"));

        service.submitPending("auto");

        // 되돌리지 않으면 이 이벤트들이 영영 어떤 배치에도 못 들어간다.
        verify(repository).release("feedback-20260915-0001", "REQUEST_VALIDATION_FAILED");
    }

    @Test
    void 재시도는_같은_batchId_로_같은_이벤트를_다시_보낸다() {
        when(repository.findByBatch("feedback-20260915-0001")).thenReturn(List.of(row(1), row(2)));
        when(retrainingClient.submit(any())).thenReturn(acceptance("feedback-20260915-0001"));

        service.retrySubmission("feedback-20260915-0001");

        ArgumentCaptor<FeedbackBatchRequest> sent =
                ArgumentCaptor.forClass(FeedbackBatchRequest.class);
        verify(retrainingClient).submit(sent.capture());
        assertThat(sent.getValue().batchId()).isEqualTo("feedback-20260915-0001");
        assertThat(sent.getValue().events()).hasSize(2);
        // 새로 고르면 payload 가 달라져 409 BATCH_ID_CONFLICT 가 난다.
        verify(repository, never()).findEligible(anyInt());
        verify(repository, never()).claim(any(), anyString(), anyString());
    }

    @Test
    void 복원할_이벤트가_없으면_새_배치를_만들지_않는다() {
        when(repository.findByBatch("feedback-20260915-0009")).thenReturn(List.of());

        assertThat(service.retrySubmission("feedback-20260915-0009")).isEmpty();

        verifyNoInteractions(retrainingClient);
    }

    @Test
    void PII_제거가_필요한_배포에서는_제출을_거부한다() {
        ReflectionTestUtils.setField(service, "requirePiiRedaction", true);

        // 질문 원문이 나가는 유일한 경로다. 조회보다 먼저 막아야 한다.
        assertThatThrownBy(() -> service.submitPending("manual"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("PII");

        verifyNoInteractions(repository);
        verifyNoInteractions(retrainingClient);
    }
}
