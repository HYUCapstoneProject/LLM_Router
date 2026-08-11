package com.llmrouter.dto;

/**
 * 시맨틱 캐시에서 꺼낸 응답.
 *
 * tier / modelId 는 "원래 이 답을 만들 때 쓴 모델"이다.
 * 캐시 히트는 LLM 호출이 0회라 비용도 0이지만,
 * 어느 모델의 답이 재사용되고 있는지는 캐시 품질 분석에 필요하다.
 *
 * @param similarity 이번 조회에서 매칭된 코사인 유사도.
 *                   request_log 에 남겨 EXP-003 임계값 튜닝의 근거로 쓴다.
 */
public record CachedResponse(
        long cacheEntryId,
        String text,
        String tier,
        String modelId,
        double similarity
) {}
