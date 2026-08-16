package com.llmrouter.dto;

import lombok.Builder;

import java.util.List;

/**
 * 누적 집계 결과.
 *
 * 원문 질문과 민감정보는 담지 않는다. 이 API 는 프론트에 공개되므로
 * 집계값만 내보낸다 (PLAN 작업 규칙).
 */
@Builder
public record AnalyticsSummary(
        String period,
        String pricingVersion,
        Requests requests,
        Cache cache,
        List<TierUsage> byTier,
        Cost cost,
        Latency latency,
        List<ErrorCount> errors
) {

    @Builder
    public record Requests(
            long total,
            long chat,
            long escalation,
            long success,
            double successRate
    ) {}

    @Builder
    public record Cache(
            long hits,
            double hitRate,
            /** 캐시 히트로 아예 건너뛴 LLM 호출 수 */
            long llmCallsSkipped
    ) {}

    @Builder
    public record TierUsage(
            String tier,
            String modelId,
            long count,
            double share,
            long inputTokens,
            long outputTokens,
            double cost
    ) {}

    @Builder
    public record Cost(
            double actual,
            double alwaysOpus,
            double savedRatio,
            /** 이 숫자가 무엇을 비교한 것인지. 화면에 그대로 노출한다. */
            String note
    ) {}

    @Builder
    public record Latency(
            long avgMs,
            long cacheHitAvgMs,
            long cacheMissAvgMs
    ) {}

    @Builder
    public record ErrorCount(
            String status,
            long count
    ) {}
}
