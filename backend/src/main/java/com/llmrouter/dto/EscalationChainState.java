package com.llmrouter.dto;

import java.util.List;

/**
 * 한 requestId 에 이미 쌓인 에스컬레이션 체인의 말단 상태.
 *
 * 체인이 없으면(=첫 에스컬레이션) Optional.empty 다.
 *
 * @param lastSeq        가장 큰 escalation_seq. 새 행은 여기에 +1 한다.
 * @param initialTier    체인 전체의 최초 라우터 판정. 모든 행이 같은 값을 갖는다.
 * @param lastFinalTier  말단 행의 final_tier = 지금 화면에 떠 있는 답변의 티어.
 *                       요청의 currentModel 과 다르면 클라이언트가 낡은 화면을 보고 있는 것이다.
 * @param escalationPath 말단 행의 누적 경로. 새 행은 여기에 목표 모델명을 덧붙인다.
 */
public record EscalationChainState(
        int lastSeq,
        String initialTier,
        String lastFinalTier,
        List<String> escalationPath
) {}
