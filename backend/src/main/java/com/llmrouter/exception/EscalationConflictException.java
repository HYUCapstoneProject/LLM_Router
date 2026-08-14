package com.llmrouter.exception;

/**
 * 이미 처리된 에스컬레이션이거나, 클라이언트가 낡은 화면을 보고 보낸 요청. → 409
 *
 * 두 경로로 들어온다.
 *   1. 체인의 말단 티어가 요청의 currentModel 과 다름 (화면이 낡음)
 *   2. uq_escalation_log_tier 유니크 위반 (더블클릭 경쟁)
 *
 * 주의: LLM 호출이 INSERT 보다 먼저라 2번 경로에서는 <b>이미 요금이 나간 뒤</b>다.
 * 프론트 버튼 비활성화가 1차 방어이고 이건 2차 방어다.
 * 비용이 문제가 되면 insert-first 패턴으로 바꿔야 한다.
 */
public class EscalationConflictException extends RuntimeException {

    public EscalationConflictException(String message) {
        super(message);
    }
}
