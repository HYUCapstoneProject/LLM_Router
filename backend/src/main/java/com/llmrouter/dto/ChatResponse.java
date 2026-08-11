package com.llmrouter.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ChatResponse {

        /** /api/escalate 가 참조할 식별자 (04 API Contract) */
    private String requestId;

    private String response;

    /** "haiku" | "sonnet" | "opus" — 프론트 배지 표시용 */
    private String model;

    /** "easy" | "medium" | "hard" — 라우터가 판정한 난이도 */
    private String tier;

    /** 실제 호출한 Claude 모델 ID (예: claude-haiku-4-5). 실험 로그용 */
    private String modelId;

    /** 시맨틱 캐시 히트 여부. true 면 LLM 호출 0회 = 비용 0 */
    private boolean cacheHit;

    /** 입력 토큰 수. 캐시 히트면 0 (LLM 호출이 없었으므로) */
    private long inputTokens;

    /** 출력 토큰 수. 캐시 히트면 0 */
    private long outputTokens;

    /** 전체 처리 시간 (ms) */
    private long latency;
}
