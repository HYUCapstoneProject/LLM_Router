package com.llmrouter.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ChatResponse {

    private String response;

    /** 실제 호출된 모델: "haiku" | "sonnet" | "opus" */
    private String model;

    /** 시맨틱 캐시 히트 여부 */
    private boolean cacheHit;

    /** 전체 처리 시간 (ms) */
    private long latency;
}
