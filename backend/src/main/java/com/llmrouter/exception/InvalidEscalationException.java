package com.llmrouter.exception;

/**
 * 모르는 모델 이름이거나, 같은 티어 또는 더 낮은 티어로 내려가려는 요청. → 400
 *
 * ML 의 {@code require_strict_tier_increase} 를 내보내기 단계가 아니라
 * 입구에서 강제한다. 여기서 막지 않으면 학습에 못 쓸 행이 DB 에 쌓이고,
 * 왜 내보내기 건수가 안 차는지 나중에 추적해야 한다.
 */
public class InvalidEscalationException extends RuntimeException {

    public InvalidEscalationException(String message) {
        super(message);
    }
}
