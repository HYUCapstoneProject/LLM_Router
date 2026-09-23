package com.llmrouter.dto;

/**
 * LLM 호출 1회의 결과.
 *
 * 토큰 수를 같이 반환하는 이유: 최종 보고서의 필수 지표가
 * "always-Opus 대비 비용"이라, 호출 시점에 실제 사용량을 기록해두지 않으면
 * 나중에 복원할 수 없다. (09 Final Report / Cost Analysis)
 *
 * @param tier      라우터가 고른 티어 (easy / medium / hard)
 * @param modelId   실제 호출한 Claude 모델 ID
 */
public record LlmResult(
        String text,
        String tier,
        String modelId,
        long inputTokens,
        long outputTokens
) {}
