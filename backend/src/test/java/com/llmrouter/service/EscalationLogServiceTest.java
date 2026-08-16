package com.llmrouter.service;

import com.llmrouter.dto.EscalationContext;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.sql.ResultSet;
import java.sql.SQLException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * RowMapper 가 드라이버 반환 타입에 대해 세우는 가정을 고정한다.
 *
 * <p>이 테스트가 있는 이유: EscalationServiceTest 는 {@code loadContext} 자체를
 * 목으로 대체하므로 매핑 코드가 한 줄도 실행되지 않는다. 그래서
 * {@code (Double) rs.getObject("similarity")} 가 캐시 히트 답변을 에스컬레이션할 때마다
 * ClassCastException 을 던지는데도 28개 테스트가 전부 통과했다.
 * (similarity 는 캐시 미스 유래 행에서 NULL 이라 e2e 도 못 잡았다)
 */
class EscalationLogServiceTest {

    /** pgJDBC 가 REAL / BIGINT 컬럼에 대해 실제로 돌려주는 타입 */
    private ResultSet resultSet(Object similarity, Object cacheEntryId) throws SQLException {
        ResultSet rs = mock(ResultSet.class);
        when(rs.getString("question")).thenReturn("질문");
        when(rs.getString("conversation_id")).thenReturn("conv-1");
        when(rs.getString("tier")).thenReturn("medium");
        when(rs.getBoolean("cache_hit")).thenReturn(similarity != null);
        when(rs.getObject("cache_entry_id")).thenReturn(cacheEntryId);
        when(rs.getObject("similarity")).thenReturn(similarity);
        when(rs.getString("router_model_version")).thenReturn("stub-medium-v0");
        when(rs.getString("embed_model_version")).thenReturn("stub-hash-v0");
        return rs;
    }

    @Test
    @DisplayName("REAL 컬럼이 Float 로 와도 Double 로 읽힌다 - 캐시 히트 에스컬레이션이 500 나던 버그")
    void mapsFloatSimilarityFromRealColumn() throws SQLException {
        // pgJDBC 는 REAL(float4) 컬럼에 Float 를 돌려준다. Double 로 캐스팅하면 터진다.
        EscalationContext ctx = EscalationLogService.CONTEXT_MAPPER
                .mapRow(resultSet(0.8312f, 42L), 0);

        assertThat(ctx.similarity()).isCloseTo(0.8312, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(ctx.cacheEntryId()).isEqualTo(42L);
        assertThat(ctx.cacheHit()).isTrue();
    }

    @Test
    @DisplayName("Double 로 와도 그대로 읽힌다 - 드라이버가 바뀌어도 안전")
    void mapsDoubleSimilarity() throws SQLException {
        EscalationContext ctx = EscalationLogService.CONTEXT_MAPPER
                .mapRow(resultSet(0.91d, 7L), 0);

        assertThat(ctx.similarity()).isCloseTo(0.91, org.assertj.core.data.Offset.offset(1e-9));
    }

    @Test
    @DisplayName("캐시 미스 유래 행은 similarity 와 cache_entry_id 가 NULL 이다")
    void mapsNullsForCacheMissOrigin() throws SQLException {
        EscalationContext ctx = EscalationLogService.CONTEXT_MAPPER
                .mapRow(resultSet(null, null), 0);

        assertThat(ctx.similarity()).isNull();
        assertThat(ctx.cacheEntryId()).isNull();
        assertThat(ctx.cacheHit()).isFalse();
    }

    @Test
    @DisplayName("BIGINT 가 Integer 로 좁혀 와도 읽힌다")
    void mapsNarrowedCacheEntryId() throws SQLException {
        EscalationContext ctx = EscalationLogService.CONTEXT_MAPPER
                .mapRow(resultSet(0.99f, 3), 0);

        assertThat(ctx.cacheEntryId()).isEqualTo(3L);
    }
}
