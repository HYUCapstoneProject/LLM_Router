package com.llmrouter.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.Set;

/**
 * FastAPI {@code GET /retraining-runs/{batchId}} 응답 (04 API Contract).
 *
 * <p>실패·거부·취소도 <b>조회 자체는 성공</b>이라 HTTP 200 으로 온다. 200 을 받았다는
 * 사실과 재학습이 잘 됐다는 사실은 다르다 — 분기는 {@link #status()} 로 한다.
 *
 * @param status     queued | running | cancelling | paused | succeeded | rejected
 *                   | failed | cancelled | unknown
 * @param stage      initializing | feedbackPreparation | feedbackMerge | training
 *                   | evaluation | packaging | gate | publishing | complete
 * @param terminal   true 면 더 폴링하지 않는다
 * @param stale      FastAPI 가 최신 상태를 확신하지 못한다는 표시.
 *                   true 면 terminal 이어도 완료로 확정하지 않는다
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RetrainingStatus(
        String schemaVersion,
        String batchId,
        String status,
        String stage,
        Boolean terminal,
        Boolean stale,
        Result result,
        Activation activation
) {

    /** 계약이 정의한 terminal 상태. {@code terminal} 필드가 없을 때의 방어선이다. */
    public static final Set<String> TERMINAL_STATUSES =
            Set.of("succeeded", "rejected", "failed", "cancelled");

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Result(String outcome, String failureCode, Boolean retryable) {
    }

    /**
     * <b>학습 성공과 serving 활성화 성공은 같은 사건이 아니다.</b>
     * {@code status=succeeded} 인데 {@code activation=loadFailed} 면 gate 는 통과했지만
     * FastAPI 는 이전 known-good 모델로 계속 응답하고 있다는 뜻이다.
     *
     * <p>필드 이름이 {@code desired}/{@code serving} 으로 나뉘어 있는 것이 핵심이다.
     * 승인된 모델과 실제로 메모리에 올라간 모델이 다를 수 있어서, 하나로 합친
     * {@code modelVersion} 같은 필드는 FastAPI 응답에 아예 없다.
     *
     * @param status pending | active | loadFailed | superseded | notApplicable
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Activation(String status, String desiredModelVersion,
                             String servingModelVersion) {
    }

    /**
     * 폴링을 멈춰도 되는가.
     *
     * <p>{@code stale=true} 나 {@code status=unknown} 은 terminal 로 보지 않는다.
     * 계약이 "완료로 확정하지 않음"이라고 못 박은 구간이다.
     */
    public boolean isTerminal() {
        if (Boolean.TRUE.equals(stale) || "unknown".equals(status)) {
            return false;
        }
        if (terminal != null) {
            return terminal;
        }
        return status != null && TERMINAL_STATUSES.contains(status);
    }

    public String failureCode() {
        return result == null ? null : result.failureCode();
    }

    public String activationStatus() {
        return activation == null ? null : activation.status();
    }
}
