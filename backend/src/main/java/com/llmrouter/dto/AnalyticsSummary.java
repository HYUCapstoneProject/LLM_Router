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
            long llmCallsSkipped,
            /** 질문이 글자까지 같아 임베딩 호출도 건너뛴 히트 */
            long exactHits,
            /** 임베딩 유사도로 잡은 히트 */
            long semanticHits,
            /**
             * exact 를 제외한 히트율.
             *
             * 전체 히트율은 "같은 질문을 다시 쳤다"까지 포함하므로 시맨틱 캐시의
             * 성능으로 제시하면 과대평가가 된다. 발표에는 이 값을 쓴다.
             */
            double semanticHitRate
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
