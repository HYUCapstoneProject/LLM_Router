package com.llmrouter.dto;

import java.util.List;

/**
 * 같은 (requestId, 목표 티어) 로 이미 존재하는 행. 예약 충돌 시 무엇을 할지 판단한다.
 *
 * <ul>
 *   <li>{@code response != null} → 이미 끝난 요청이다. 그 답을 그대로 돌려준다 (멱등, 200).
 *   <li>{@code response == null} 이고 최근 → 다른 요청이 처리 중이다 (409).
 *   <li>{@code response == null} 이고 오래됨 → 프로세스가 죽어 남은 유령 예약이다. 인수한다.
 * </ul>
 */
public record ExistingEscalation(
        int escalationSeq,
        String response,
        String initialTier,
        String finalTier,
        String finalModelId,
        List<String> escalationPath,
        long inputTokens,
        long outputTokens,
        long latencyMs
) {
    public boolean completed() {
        return response != null;
    }
}
