package com.llmrouter.service;

import com.llmrouter.client.RetrainingClient;
import com.llmrouter.client.RetrainingClientException;
import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.EscalationEvent;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RouterTier;
import com.llmrouter.repository.FeedbackBatchRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
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
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class FeedbackBatchServiceTest {
    @Mock FeedbackBatchRepository repository;
    @Mock RetrainingClient retrainingClient;
    FeedbackBatchService service;
    private static final Instant NOW = Instant.parse("2026-09-26T04:00:00Z");
    private static final String BATCH = "feedback-20260926-0001";

    @BeforeEach
    void setUp() {
        service = new FeedbackBatchService(repository, retrainingClient, Clock.fixed(NOW, ZoneOffset.UTC));
        ReflectionTestUtils.setField(service, "batchSize", 1000);
        ReflectionTestUtils.setField(service, "requirePiiRedaction", false);
    }

    private FeedbackBatchRequest request() {
        return new FeedbackBatchRequest(BATCH, NOW.minusSeconds(120), List.of(
                new EscalationEvent("esc-7", "req-7", "question", RouterTier.easy, "haiku",
                        RouterTier.hard, "opus", true, 2, NOW.minusSeconds(3600), "router-v1", true)));
    }

    private EscalateAcceptance acceptance() {
        return new EscalateAcceptance("escalation-acceptance-v1", "accepted", BATCH,
                "projects/p/locations/r/pipelineJobs/j", "2.1", "router-v2.1",
                "minor_fine_tune", "successfulHttpRequest", "/retraining-runs/" + BATCH);
    }

    private void claim(FeedbackBatchRequest request) {
        when(repository.claimEligible(1000, "manual", "feedback-20260926-", NOW))
                .thenReturn(Optional.of(request));
    }

    @Test
    void emptyClaimDoesNotSubmit() {
        when(repository.claimEligible(anyInt(), anyString(), anyString(), any()))
                .thenReturn(Optional.empty());
        assertThat(service.submitPending("manual")).isEmpty();
        verifyNoInteractions(retrainingClient);
    }

    @Test
    void submitsTheRequestFrozenByTheAtomicClaim() {
        var frozen = request();
        claim(frozen);
        when(retrainingClient.submit(frozen)).thenReturn(acceptance());
        assertThat(service.submitPending("manual").orElseThrow().accepted()).isTrue();
        verify(retrainingClient).submit(same(frozen));
        verify(repository).recordAcceptance(acceptance());
    }

    @Test
    void timeoutThenRetryKeepsOriginalTimestampAndEveryEvent() {
        var frozen = request();
        claim(frozen);
        when(repository.findRequest(BATCH)).thenReturn(Optional.of(frozen));
        when(retrainingClient.submit(frozen))
                .thenThrow(new RetrainingClientException("timeout", null, null))
                .thenReturn(acceptance());
        assertThat(service.submitPending("manual").orElseThrow().accepted()).isFalse();
        assertThat(service.retrySubmission(BATCH).orElseThrow().accepted()).isTrue();
        verify(retrainingClient, times(2)).submit(same(frozen));
        verify(repository, never()).release(anyString(), anyString());
        verify(repository, times(1)).claimEligible(anyInt(), anyString(), anyString(), any());
    }

    @Test
    void definiteValidationRejectionReleasesEvents() {
        claim(request());
        when(retrainingClient.submit(any())).thenThrow(new RetrainingClientException(
                "invalid", 422, "REQUEST_VALIDATION_FAILED"));
        service.submitPending("manual");
        verify(repository).release(BATCH, "REQUEST_VALIDATION_FAILED");
    }

    @Test
    void conflictingRemoteBatchNeverRequeuesItsEventsUnderAnotherId() {
        claim(request());
        when(retrainingClient.submit(any())).thenThrow(new RetrainingClientException(
                "conflict", 409, "BATCH_ID_CONFLICT"));
        service.submitPending("manual");
        verify(repository).quarantine(BATCH, "BATCH_ID_CONFLICT");
        verify(repository, never()).release(anyString(), anyString());
    }

    @Test
    void malformedAcceptanceIsQuarantinedBecauseTheRemoteJobMayExist() {
        claim(request());
        when(retrainingClient.submit(any())).thenThrow(new RetrainingClientException(
                "invalid response", null, "RESPONSE_CONTRACT_MISMATCH"));
        service.submitPending("manual");
        verify(repository).quarantine(BATCH, "RESPONSE_CONTRACT_MISMATCH");
        verify(repository, never()).release(anyString(), anyString());
    }

    @Test
    void missingLegacySnapshotDoesNotReconstructOrSubmit() {
        when(repository.findRequest(BATCH)).thenReturn(Optional.empty());
        assertThat(service.retrySubmission(BATCH)).isEmpty();
        verifyNoInteractions(retrainingClient);
    }

    @Test
    void piiGuardBlocksBothFreshAndRetriedRequestsBeforeReadingTheDatabase() {
        ReflectionTestUtils.setField(service, "requirePiiRedaction", true);
        assertThatThrownBy(() -> service.submitPending("manual")).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> service.retrySubmission(BATCH)).isInstanceOf(IllegalStateException.class);
        verifyNoInteractions(repository, retrainingClient);
    }
}
