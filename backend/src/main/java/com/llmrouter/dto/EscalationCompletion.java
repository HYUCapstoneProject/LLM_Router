package com.llmrouter.dto;

import lombok.Builder;

/**
 * 예약해 둔 행에 실제 결과를 채운다.
 *
 * <p>이 시점에 {@code user_satisfied} 를 true 로 올리고 앞 단계를 false 로 내린다.
 * 예약 시점이 아니라 여기서 뒤집는 이유: LLM 호출이 실패해 예약을 되돌리면
 * 앞 단계까지 잘못 내려간 상태가 남는다.
 */
@Builder
public record EscalationCompletion(
        String requestId,
        /** 예약 INSERT 가 돌려준 행 id. seq 로 찾으면 슬롯이 넘어간 뒤 남의 행을 건드린다 */
        long escalationLogId,
        String response,
        float[] embedding,
        String finalModelId,
        long inputTokens,
        long outputTokens,
        long latencyMs
) {}
