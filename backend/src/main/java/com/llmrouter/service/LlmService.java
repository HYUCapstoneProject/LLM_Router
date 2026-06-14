package com.llmrouter.service;

import org.springframework.stereotype.Service;

/**
 * LLM API 호출 레이어 (Claude Haiku / Sonnet).
 *
 * 구현 계획 (5-6월 API 연동):
 *   - Anthropic Java SDK 또는 직접 HTTP 호출
 *   - 모델별 엔드포인트 분기
 *
 * 현재: stub — 더미 응답 반환
 */
@Service
public class LlmService {

    /**
     * 지정된 모델로 LLM API를 호출한다.
     *
     * @param model "haiku" 또는 "sonnet"
     * @param message 사용자 질문
     * @return LLM 응답 텍스트
     */
    public String call(String model, String message) {
        // TODO: Anthropic API 연동
        return "[%s stub] '%s' 에 대한 응답입니다.".formatted(model, message);
    }
}
