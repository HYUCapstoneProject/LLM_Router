package com.llmrouter.client;

import java.util.Set;

/**
 * FastAPI 재학습 계약 호출 실패.
 *
 * <p>계약은 <b>{@code error} 문자열이 아니라 안정된 {@code code} 로 분기하라</b>고
 * 명시한다. 메시지는 사람이 읽으라고 있는 것이라 바뀔 수 있다. 그래서 {@link #code()}
 * 를 따로 들고 다닌다.
 */
public class RetrainingClientException extends RuntimeException {

    /**
     * 같은 payload 로 다시 보내면 되는 오류들. 그 외는 전부 사람이 봐야 한다.
     *
     * <p>{@code BATCH_ID_CONFLICT} 가 여기 없는 것이 중요하다 — 같은 batchId 로
     * <b>다른</b> 내용을 보냈다는 뜻이라, 자동 재시도하면 영원히 409 만 받는다.
     * {@code GCS_OBJECT_CONFLICT} 도 마찬가지로 덮어쓰기를 자동화하면 안 된다.
     */
    private static final Set<String> RETRYABLE_CODES = Set.of(
            "RETRAINING_IN_PROGRESS",
            "PIPELINE_SUBMISSION_FAILED",
            "GCS_BACKEND_UNAVAILABLE",
            "PIPELINE_BACKEND_UNAVAILABLE",
            "PIPELINE_STATUS_UNAVAILABLE"
    );

    private final Integer statusCode;
    private final String code;

    public RetrainingClientException(String message, Integer statusCode, String code) {
        this(message, statusCode, code, null);
    }

    public RetrainingClientException(String message, Integer statusCode, String code, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
        this.code = code;
    }

    public Integer statusCode() {
        return statusCode;
    }

    /** 계약이 정의한 안정된 오류 코드. 파싱에 실패했으면 null 이다. */
    public String code() {
        return code;
    }

    /**
     * 같은 batchId·같은 payload 로 backoff 재시도해도 되는가.
     *
     * <p>코드를 못 읽은 경우(null)에는 5xx 만 재시도한다. 4xx 를 코드 없이 재시도하면
     * 계약 위반 payload 를 무한히 다시 던지게 된다.
     */
    public boolean isRetryable() {
        if (code != null) {
            return RETRYABLE_CODES.contains(code);
        }
        return statusCode == null || statusCode >= 500;
    }
}
