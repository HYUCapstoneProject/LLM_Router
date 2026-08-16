package com.llmrouter.service;

import com.llmrouter.config.PricingTable;
import com.llmrouter.dto.AnalyticsSummary;
import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

/**
 * request_log 누적 집계.
 *
 * 비용 계산에서 캐시 히트 행은 제외한다. 그 행들은 토큰이 0 이라
 * "같은 토큰을 Opus 로 보냈다면" 을 계산할 근거가 없기 때문이다.
 * 캐시의 효과는 비용이 아니라 "건너뛴 호출 수"로 따로 보고한다.
 */
@Service
@RequiredArgsConstructor
public class AnalyticsService {

    private final JdbcTemplate jdbc;
    private final PricingTable pricing;

    private static final String OVERALL_SQL = """
            SELECT
                count(*)                                                       AS total,
                count(*) FILTER (WHERE origin = 'chat')                        AS chat,
                count(*) FILTER (WHERE origin = 'escalate')                    AS escalation,
                count(*) FILTER (WHERE status = 'success')                     AS success,
                count(*) FILTER (WHERE cache_hit)                              AS cache_hits,
                count(*) FILTER (WHERE origin = 'chat' AND status = 'success') AS chat_success,
                -- avg 는 numeric 을 돌려준다. 소수점이 붙은 numeric 에 getLong 을 쓰면
                -- pgjdbc 가 "Bad value for type long" 을 던지므로 SQL 에서 정수로 만든다.
                coalesce(round(avg(latency_ms) FILTER (WHERE status = 'success')), 0)::bigint                   AS avg_ms,
                coalesce(round(avg(latency_ms) FILTER (WHERE cache_hit)), 0)::bigint                            AS hit_ms,
                coalesce(round(avg(latency_ms) FILTER (WHERE status = 'success' AND NOT cache_hit)), 0)::bigint AS miss_ms
            FROM request_log
            WHERE created_at >= ?
            """;

    /** LLM 을 실제로 호출한 행만. 캐시 히트는 토큰이 0 이라 단가 비교 대상이 아니다. */
    private static final String BY_TIER_SQL = """
            SELECT tier,
                   model_id,
                   count(*)                        AS cnt,
                   coalesce(sum(input_tokens), 0)  AS in_tok,
                   coalesce(sum(output_tokens), 0) AS out_tok
            FROM request_log
            WHERE created_at >= ?
              AND status = 'success'
              AND NOT cache_hit
              AND tier IS NOT NULL
            GROUP BY tier, model_id
            -- 알파벳 순이면 easy, hard, medium 이 되어 화면에서 읽기 나쁘다. 난이도 순으로 고정한다.
            ORDER BY CASE tier WHEN 'easy' THEN 1 WHEN 'medium' THEN 2 WHEN 'hard' THEN 3 ELSE 4 END
            """;

    private static final String ERRORS_SQL = """
            SELECT status, count(*) AS cnt
            FROM request_log
            WHERE created_at >= ? AND status <> 'success'
            GROUP BY status
            ORDER BY cnt DESC
            """;

    public AnalyticsSummary summary(String period) {
        OffsetDateTime since = cutoff(period);

        Row overall = jdbc.queryForObject(OVERALL_SQL, (rs, i) -> new Row(
                rs.getLong("total"),
                rs.getLong("chat"),
                rs.getLong("escalation"),
                rs.getLong("success"),
                rs.getLong("cache_hits"),
                rs.getLong("chat_success"),
                rs.getLong("avg_ms"),
                rs.getLong("hit_ms"),
                rs.getLong("miss_ms")), since);

        List<TierRow> tierRows = jdbc.query(BY_TIER_SQL, (rs, i) -> new TierRow(
                rs.getString("tier"),
                rs.getString("model_id"),
                rs.getLong("cnt"),
                rs.getLong("in_tok"),
                rs.getLong("out_tok")), since);

        long llmCalls = tierRows.stream().mapToLong(TierRow::count).sum();
        double actualCost = 0;
        double opusCost = 0;
        List<AnalyticsSummary.TierUsage> byTier = new ArrayList<>();

        for (TierRow row : tierRows) {
            double cost = pricing.cost(row.tier(), row.inputTokens(), row.outputTokens());
            actualCost += cost;
            opusCost += pricing.alwaysOpusCost(row.inputTokens(), row.outputTokens());
            byTier.add(AnalyticsSummary.TierUsage.builder()
                    .tier(row.tier())
                    .modelId(row.modelId())
                    .count(row.count())
                    .share(ratio(row.count(), llmCalls))
                    .inputTokens(row.inputTokens())
                    .outputTokens(row.outputTokens())
                    .cost(round(cost))
                    .build());
        }

        List<AnalyticsSummary.ErrorCount> errors = jdbc.query(ERRORS_SQL, (rs, i) ->
                AnalyticsSummary.ErrorCount.builder()
                        .status(rs.getString("status"))
                        .count(rs.getLong("cnt"))
                        .build(), since);

        return AnalyticsSummary.builder()
                .period(period)
                .pricingVersion(pricing.getVersion())
                .requests(AnalyticsSummary.Requests.builder()
                        .total(overall.total())
                        .chat(overall.chat())
                        .escalation(overall.escalation())
                        .success(overall.success())
                        .successRate(ratio(overall.success(), overall.total()))
                        .build())
                .cache(AnalyticsSummary.Cache.builder()
                        .hits(overall.cacheHits())
                        .hitRate(ratio(overall.cacheHits(), overall.chatSuccess()))
                        .llmCallsSkipped(overall.cacheHits())
                        .build())
                .byTier(byTier)
                .cost(AnalyticsSummary.Cost.builder()
                        .actual(round(actualCost))
                        .alwaysOpus(round(opusCost))
                        .savedRatio(opusCost > 0 ? round(1 - actualCost / opusCost) : 0)
                        .note("LLM을 실제 호출한 요청의 토큰을 Opus 단가로 환산한 값이다. "
                                + "Opus가 같은 길이로 답한다는 보장은 없으므로 단가 차이만 나타낸다. "
                                + "캐시로 건너뛴 호출은 이 계산에 포함되지 않는다.")
                        .build())
                .latency(AnalyticsSummary.Latency.builder()
                        .avgMs(overall.avgMs())
                        .cacheHitAvgMs(overall.hitMs())
                        .cacheMissAvgMs(overall.missMs())
                        .build())
                .errors(errors)
                .build();
    }

    /** today | 7d | all — 그 외 값은 all 로 본다 */
    private OffsetDateTime cutoff(String period) {
        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        return switch (period == null ? "all" : period) {
            case "today" -> now.truncatedTo(ChronoUnit.DAYS);
            case "7d" -> now.minusDays(7);
            default -> OffsetDateTime.of(1970, 1, 1, 0, 0, 0, 0, ZoneOffset.UTC);
        };
    }

    private double ratio(long numerator, long denominator) {
        return denominator == 0 ? 0 : round((double) numerator / denominator);
    }

    private double round(double value) {
        return Math.round(value * 10_000.0) / 10_000.0;
    }

    private record Row(long total, long chat, long escalation, long success,
                       long cacheHits, long chatSuccess,
                       long avgMs, long hitMs, long missMs) {}

    private record TierRow(String tier, String modelId, long count, long inputTokens, long outputTokens) {}
}
