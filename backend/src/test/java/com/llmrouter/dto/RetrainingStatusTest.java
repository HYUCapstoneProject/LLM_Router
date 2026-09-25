package com.llmrouter.dto;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 폴링을 언제 멈춰도 되는지가 이 클래스의 전부다.
 *
 * <p>계약은 "{@code stale=true} 또는 {@code status=unknown} 이면 완료로 확정하지 않음"
 * 이라고 못 박는다. 이걸 놓치면 아직 돌고 있는 학습을 끝난 것으로 기록하고 폴링을 멈춘다.
 */
class RetrainingStatusTest {

    private RetrainingStatus of(String status, Boolean terminal, Boolean stale) {
        return new RetrainingStatus("v1", "b-1", status, "training", terminal, stale, null, null);
    }

    @Test
    void terminal_플래그를_그대로_따른다() {
        assertThat(of("succeeded", true, false).isTerminal()).isTrue();
        assertThat(of("running", false, false).isTerminal()).isFalse();
    }

    @Test
    void terminal_이_없으면_상태값으로_판단한다() {
        assertThat(of("succeeded", null, null).isTerminal()).isTrue();
        assertThat(of("rejected", null, null).isTerminal()).isTrue();
        assertThat(of("failed", null, null).isTerminal()).isTrue();
        assertThat(of("cancelled", null, null).isTerminal()).isTrue();
        assertThat(of("running", null, null).isTerminal()).isFalse();
        assertThat(of("queued", null, null).isTerminal()).isFalse();
    }

    @Test
    void stale_이면_terminal_이어도_확정하지_않는다() {
        // FastAPI 가 최신 상태를 확신하지 못한다는 뜻이다. 여기서 멈추면 결과를 잘못 적는다.
        assertThat(of("succeeded", true, true).isTerminal()).isFalse();
    }

    @Test
    void unknown_은_terminal_로_보지_않는다() {
        assertThat(of("unknown", true, false).isTerminal()).isFalse();
    }

    @Test
    void 중첩_필드가_없어도_널을_돌려준다() {
        // 진행 중인 배치는 result·activation 이 비어 있다. 여기서 NPE 가 나면 폴링이 죽는다.
        RetrainingStatus running = of("running", false, false);
        assertThat(running.failureCode()).isNull();
        assertThat(running.activationStatus()).isNull();
    }

    @Test
    void 학습_성공과_서빙_반영을_구분한다() {
        RetrainingStatus loadFailed = new RetrainingStatus("v1", "b-1", "succeeded", "complete",
                true, false,
                new RetrainingStatus.Result("succeeded", null, null),
                new RetrainingStatus.Activation("loadFailed", "router-v2", "router-v1"));

        assertThat(loadFailed.isTerminal()).isTrue();
        // gate 는 통과했지만 FastAPI 는 이전 known-good 모델로 응답하고 있다.
        assertThat(loadFailed.activationStatus()).isEqualTo("loadFailed");
    }

    @Test
    void pollingContinuesUntilActivationResolvesAndNeverConfirmsStaleResults() {
        for (String activation : new String[]{"pending", "active", "loadFailed", "superseded", "notApplicable"}) {
            RetrainingStatus fresh = new RetrainingStatus("retraining-status-v1", "b-1", "succeeded",
                    "complete", true, false, null,
                    new RetrainingStatus.Activation(activation, "router-v2", "router-v2"));
            assertThat(fresh.isPollingComplete()).isEqualTo(!"pending".equals(activation));
            RetrainingStatus stale = new RetrainingStatus(fresh.schemaVersion(), fresh.batchId(),
                    fresh.status(), fresh.stage(), true, true, null, fresh.activation());
            assertThat(stale.isPollingComplete()).isFalse();
        }
        assertThat(of("succeeded", true, false).isPollingComplete()).isFalse();
        assertThat(of("failed", true, false).isPollingComplete()).isTrue();
    }

    @Test
    void activeFlagWithoutMatchingVersionsCannotCompletePolling() {
        var status = new RetrainingStatus("retraining-status-v1", "b-1", "succeeded", "complete",
                true, false, null, new RetrainingStatus.Activation("active", "router-v2", "router-v1"));
        assertThat(status.isPollingComplete()).isFalse();
    }
}
