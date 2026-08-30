package com.llmrouter.service;

import com.llmrouter.config.PricingTable;
import com.llmrouter.dto.AnalyticsSummary;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import java.sql.ResultSet;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * SQL 자체의 정확성은 여기서 검증되지 않는다(실 DB 필요).
 * 여기서 지키는 건 SQL 이 돌려준 값으로 만드는 계산 — 비용, 비율, 0 나눗셈 — 이다.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AnalyticsServiceTest {

    @Mock JdbcTemplate jdbc;
    AnalyticsService service;

    @BeforeEach
    void setUp() {
        service = new AnalyticsService(jdbc, new PricingTable("test", 1.0, 5.0, 3.0, 15.0, 5.0, 25.0));
    }

    @Test
    void costCompareUsesOnlyRowsThatActuallyCalledAnLlm() throws Exception {
        stubOverall(overall(10L, 8L, 2L, 10L, 4L, 1L, 3L, 8L, 900L, 12L, 1_400L));
        // easy 1M/1M → haiku 로 $6, 전부 opus 였다면 $30
        stubByTier(List.of(Map.of(
                "tier", "easy", "model_id", "claude-haiku-4-5",
                "cnt", 6L, "in_tok", 1_000_000L, "out_tok", 1_000_000L)));
        stubErrors(List.of());

        AnalyticsSummary summary = service.summary("all");

        assertThat(summary.cost().actual()).isEqualTo(6.0);
        assertThat(summary.cost().alwaysOpus()).isEqualTo(30.0);
        assertThat(summary.cost().savedRatio()).isCloseTo(0.8, within(1e-9));

        // 캐시 히트 4건은 토큰이 0 이라 위 비교에 들어가지 않는다.
        // 그 효과는 "건너뛴 호출 수"로만 보고한다.
        assertThat(summary.cache().llmCallsSkipped()).isEqualTo(4);
        assertThat(summary.cache().hitRate()).isEqualTo(0.5);

        // exact 1 / semantic 3. 시맨틱 히트율은 분모에서도 exact 를 뺀다: 3 / (8-1)
        assertThat(summary.cache().exactHits()).isEqualTo(1);
        assertThat(summary.cache().semanticHits()).isEqualTo(3);
        assertThat(summary.cache().semanticHitRate()).isCloseTo(3.0 / 7, within(1e-4));
    }

    /** 히트율 분모는 전체 요청이 아니라 성공한 chat 요청이다. 에스컬레이션에는 캐시 조회가 없다. */
    @Test
    void cacheHitRateExcludesEscalations() throws Exception {
        stubOverall(overall(30L, 10L, 20L, 30L, 5L, 0L, 5L, 10L, 100L, 10L, 200L));
        stubByTier(List.of());
        stubErrors(List.of());

        assertThat(service.summary("all").cache().hitRate()).isEqualTo(0.5);
    }

    @Test
    void emptyPeriodDoesNotDivideByZero() throws Exception {
        stubOverall(overall(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L));
        stubByTier(List.of());
        stubErrors(List.of());

        AnalyticsSummary summary = service.summary("today");

        assertThat(summary.requests().successRate()).isZero();
        assertThat(summary.cache().hitRate()).isZero();
        assertThat(summary.cost().savedRatio()).isZero();
        assertThat(summary.byTier()).isEmpty();
    }

    @Test
    void tierSharesSumToOneAcrossLlmCalls() throws Exception {
        stubOverall(overall(10L, 10L, 0L, 10L, 0L, 0L, 0L, 10L, 500L, 0L, 500L));
        stubByTier(List.of(
                row("easy", "claude-haiku-4-5", 6L),
                row("medium", "claude-sonnet-4-6", 3L),
                row("hard", "claude-opus-4-8", 1L)));
        stubErrors(List.of());

        assertThat(service.summary("all").byTier())
                .extracting(AnalyticsSummary.TierUsage::share)
                .containsExactly(0.6, 0.3, 0.1);
    }

    /** Map.of 는 인자 10쌍이 한계다. OVERALL_SQL 컬럼이 12개라 헬퍼로 만든다. */
    private static Map<String, Object> overall(long total, long chat, long escalation, long success,
                                               long cacheHits, long exactHits, long semanticHits,
                                               long chatSuccess, long avgMs, long hitMs, long missMs) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("total", total);           m.put("chat", chat);
        m.put("escalation", escalation); m.put("success", success);
        m.put("cache_hits", cacheHits);  m.put("exact_hits", exactHits);
        m.put("semantic_hits", semanticHits); m.put("chat_success", chatSuccess);
        m.put("avg_ms", avgMs);          m.put("hit_ms", hitMs);
        m.put("miss_ms", missMs);
        return m;
    }

    private static Map<String, Object> row(String tier, String modelId, long count) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("tier", tier);
        m.put("model_id", modelId);
        m.put("cnt", count);
        m.put("in_tok", 1_000L);
        m.put("out_tok", 1_000L);
        return m;
    }

    // --- JdbcTemplate 스텁 -------------------------------------------------
    // RowMapper 를 직접 호출해 준다. 그래야 컬럼명 오타가 테스트에서도 드러난다.

    @SuppressWarnings("unchecked")
    private <T> void stubOverall(Map<String, Object> columns) throws Exception {
        when(jdbc.queryForObject(anyString(), any(RowMapper.class), any(OffsetDateTime.class)))
                .thenAnswer(inv -> ((RowMapper<T>) inv.getArgument(1)).mapRow(resultSet(columns), 1));
    }

    @SuppressWarnings("unchecked")
    private <T> void stubByTier(List<Map<String, Object>> rows) throws Exception {
        when(jdbc.query(contains("GROUP BY tier"), any(RowMapper.class), any(OffsetDateTime.class)))
                .thenAnswer(inv -> mapAll((RowMapper<T>) inv.getArgument(1), rows));
    }

    @SuppressWarnings("unchecked")
    private <T> void stubErrors(List<Map<String, Object>> rows) throws Exception {
        when(jdbc.query(contains("status <> 'success'"), any(RowMapper.class), any(OffsetDateTime.class)))
                .thenAnswer(inv -> mapAll((RowMapper<T>) inv.getArgument(1), rows));
    }

    private static <T> List<T> mapAll(RowMapper<T> mapper, List<Map<String, Object>> rows) throws Exception {
        List<T> mapped = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            mapped.add(mapper.mapRow(resultSet(rows.get(i)), i + 1));
        }
        return mapped;
    }

    /** 없는 컬럼을 읽으면 예외가 나야 한다 — 오타를 통과시키면 테스트가 의미 없다. */
    private static ResultSet resultSet(Map<String, Object> columns) throws Exception {
        ResultSet rs = mock(ResultSet.class);
        when(rs.getLong(anyString())).thenAnswer(inv -> require(columns, inv.getArgument(0)));
        when(rs.getString(anyString())).thenAnswer(inv -> require(columns, inv.getArgument(0)));
        return rs;
    }

    private static Object require(Map<String, Object> columns, String name) {
        if (!columns.containsKey(name)) {
            throw new IllegalArgumentException("존재하지 않는 컬럼: " + name);
        }
        return columns.get(name);
    }
}
