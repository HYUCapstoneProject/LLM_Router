package com.llmrouter.service;

import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.LlmResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CacheServiceTest {
    @Mock JdbcTemplate jdbc;
    CacheService service;

    @BeforeEach
    void setUp() {
        service = new CacheService(jdbc);
        ReflectionTestUtils.setField(service, "easyThreshold", 0.90);
        ReflectionTestUtils.setField(service, "mediumThreshold", 0.93);
        ReflectionTestUtils.setField(service, "hardThreshold", 0.96);
    }

    /** exact 조회는 벡터가 아니라 해시 하나만 바인딩한다 (인자 1개) */
    @Test
    void exactMissReturnsEmptyWithoutUpdatingHitCount() {
        when(jdbc.query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), any(Object.class)))
                .thenReturn(List.of());
        assertThat(service.getExact("처음 보는 질문")).isEmpty();
        verify(jdbc, never()).update(startsWith("UPDATE"), anyLong());
    }

    @Test
    void exactHitReturnsSimilarityOneAndIncrementsHitCount() {
        CachedResponse hit = new CachedResponse(5L, "answer", "easy", "m", 1.0);
        when(jdbc.query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), any(Object.class)))
                .thenReturn(List.of(hit));

        assertThat(service.getExact("같은 질문")).contains(hit);
        verify(jdbc).update(startsWith("UPDATE"), eq(5L));
    }

    /** 정규화가 Java 한 곳에만 있으므로, 바인딩되는 값은 해시여야 한다 */
    @Test
    void exactLookupBindsHashNotRawQuestion() {
        when(jdbc.query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), any(Object.class)))
                .thenReturn(List.of());

        service.getExact("  같은   질문  ");

        ArgumentCaptor<Object> bound = ArgumentCaptor.forClass(Object.class);
        verify(jdbc).query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), bound.capture());
        assertThat(bound.getValue())
                .isEqualTo(QuestionHash.of("같은 질문"))
                .asString().hasSize(64);
    }

    @Test
    void missReturnsEmptyWithoutUpdatingHitCount() {
        when(jdbc.query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), any(), any()))
                .thenReturn(List.of());
        assertThat(service.get(new float[]{0.1f, -0.2f})).isEmpty();
        verify(jdbc, never()).update(startsWith("UPDATE"), anyLong());
    }

    @Test
    void hitReturnsBestRowAndIncrementsHitCount() {
        CachedResponse hit = new CachedResponse(3L, "answer", "easy", "m", 0.98);
        when(jdbc.query(anyString(), any(org.springframework.jdbc.core.RowMapper.class), any(), any()))
                .thenReturn(List.of(hit));
        assertThat(service.get(new float[]{1f})).contains(hit);
        verify(jdbc).update(startsWith("UPDATE"), eq(3L));
    }

    @Test
    void putSerializesVectorAndUsesTierThresholds() {
        when(jdbc.queryForObject(anyString(), eq(Long.class), any(Object[].class))).thenReturn(11L);
        long id = service.put("q", new float[]{0.1f, -2f},
                new LlmResult("a", "hard", "opus", 1, 2), "embed-v1");

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).queryForObject(anyString(), eq(Long.class), args.capture());
        // question, question_hash, embedding, response, tier, model_id, embed_model_version, threshold
        assertThat(args.getValue()).containsExactly(
                "q", QuestionHash.of("q"), "[0.1,-2.0]", "a", "hard", "opus", "embed-v1", 0.96);
        assertThat(id).isEqualTo(11L);
    }

    @Test
    void unknownTierUsesMediumThreshold() {
        when(jdbc.queryForObject(anyString(), eq(Long.class), any(Object[].class))).thenReturn(12L);
        service.put("q", new float[]{1f}, new LlmResult("a", "future", "m", 0, 0), "embed-v1");
        verify(jdbc).queryForObject(anyString(), eq(Long.class), eq("q"), eq(QuestionHash.of("q")),
                eq("[1.0]"), eq("a"), eq("future"), eq("m"), any(), eq(0.93));
    }
}
