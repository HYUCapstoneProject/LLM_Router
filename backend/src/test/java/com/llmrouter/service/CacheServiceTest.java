package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
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
    @Mock RouterClient routerClient;
    CacheService service;

    @BeforeEach
    void setUp() {
        service = new CacheService(jdbc, routerClient);
        ReflectionTestUtils.setField(service, "easyThreshold", 0.90);
        ReflectionTestUtils.setField(service, "mediumThreshold", 0.93);
        ReflectionTestUtils.setField(service, "hardThreshold", 0.96);
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
        when(routerClient.embedModelVersion()).thenReturn("embed-v1");
        service.put("q", new float[]{0.1f, -2f}, new LlmResult("a", "hard", "opus", 1, 2));

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(anyString(), args.capture());
        assertThat(args.getValue()).containsExactly("q", "[0.1,-2.0]", "a", "hard", "opus", "embed-v1", 0.96);
    }

    @Test
    void unknownTierUsesMediumThreshold() {
        service.put("q", new float[]{1f}, new LlmResult("a", "future", "m", 0, 0));
        verify(jdbc).update(anyString(), eq("q"), eq("[1.0]"), eq("a"), eq("future"), eq("m"), any(), eq(0.93));
    }
}
