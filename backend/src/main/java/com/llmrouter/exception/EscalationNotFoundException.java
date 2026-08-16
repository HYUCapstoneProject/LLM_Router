package com.llmrouter.exception;

/**
 * /api/escalate 가 참조한 requestId 로 원본 요청을 찾지 못했다. → 404
 *
 * 본문의 message 로 폴백하지 않고 여기서 끊는 이유는
 * {@link com.llmrouter.dto.EscalationContext} 주석 참조.
 *
 * 알려진 부작용: RequestLogService 는 적재 실패를 의도적으로 삼키므로,
 * /api/chat 때 DB 가 잠깐 흔들렸다면 그 답변은 영구히 에스컬레이션할 수 없다.
 * 사용자가 다시 물으면 되는 문제라 감수한다.
 */
public class EscalationNotFoundException extends RuntimeException {

    public EscalationNotFoundException(String requestId) {
        super("원본 요청을 찾을 수 없습니다. requestId=" + requestId);
    }
}
