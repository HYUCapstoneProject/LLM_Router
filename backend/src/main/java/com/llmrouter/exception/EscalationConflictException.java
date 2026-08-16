package com.llmrouter.exception;

/**
 * 클라이언트가 낡은 화면을 보고 보낸 요청. → 409
 *
 * 체인의 말단 티어가 요청의 currentModel 과 다를 때만 여기로 온다.
 * 이미 더 위로 올라간 뒤인데 클라이언트는 그 사실을 모르는 상황이다.
 *
 * 중복·경쟁으로 인한 충돌은 여기가 아니라
 * {@link EscalationInProgressException} 이다. 예약을 LLM 호출보다 먼저 잡으므로
 * 그쪽은 요금이 나가기 전에 걸러진다.
 */
public class EscalationConflictException extends RuntimeException {

    public EscalationConflictException(String message) {
        super(message);
    }
}
