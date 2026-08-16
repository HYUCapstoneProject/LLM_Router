package com.llmrouter.client;

/** FastAPI 호출 또는 응답 계약 검증 실패. ChatService가 이 예외만 안전 폴백한다. */
public class RouterClientException extends RuntimeException {

    private final Integer statusCode;

    public RouterClientException(String message, Integer statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public RouterClientException(String message, Integer statusCode, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    public Integer statusCode() {
        return statusCode;
    }
}
