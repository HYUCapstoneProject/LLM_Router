package com.llmrouter.exception;

/**
 * 같은 요청이 아직 처리 중이다. → 409
 *
 * {@link EscalationConflictException}(이미 상위 모델로 답변함)과 나눠 둔 이유:
 * 사용자가 취할 행동이 정반대다. 이쪽은 "조금 기다리면 답이 온다"이고,
 * 저쪽은 "이미 끝났으니 화면이 낡았다"이다. 같은 문구를 쓰면 둘 다 틀린 안내가 된다.
 */
public class EscalationInProgressException extends RuntimeException {

    public EscalationInProgressException(String message) {
        super(message);
    }
}
