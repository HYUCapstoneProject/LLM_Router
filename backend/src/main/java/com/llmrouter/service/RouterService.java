package com.llmrouter.service;

import org.springframework.stereotype.Service;

/**
 * 질문 복잡도를 분류하여 Light / Heavy 모델을 선택한다.
 *
 * 구현 계획 (7-8월):
 *   1차 필터 - 규칙 기반 (토큰 수, 인사말 패턴)
 *   2차 필터 - ModernBERT 파인튜닝 분류기
 *
 * 현재: stub — 모든 요청을 sonnet으로 처리 (Baseline과 동일)
 */
@Service
public class RouterService {

    public enum ModelTier { LIGHT, MEDIUM, HEAVY }

    /**
     * @return LIGHT → haiku, HEAVY → sonnet
     */
    public ModelTier classify(String message) {
        // TODO: ModernBERT 분류기 연동
        return ModelTier.HEAVY;
    }

    public String resolveModelName(ModelTier tier) {
        return switch (tier) {
            case LIGHT -> "haiku";
            case MEDIUM -> "sonnet";
            case HEAVY -> "opus";
        };
    }
}
