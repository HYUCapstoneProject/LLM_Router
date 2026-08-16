package com.llmrouter.service;

import com.llmrouter.dto.RequestLogEntry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.dao.DataAccessResourceFailureException;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class RequestLogServiceTest {

    @Mock JdbcTemplate jdbc;
    RequestLogService service;

    @BeforeEach
    void setUp() {
        service = new RequestLogService(jdbc);
    }

    @Test
    void successRecordIncludesOriginAndSuccessStatus() {
        RequestLogEntry entry = RequestLogEntry.builder()
                .requestId("00000000-0000-0000-0000-000000000001")
                .question("q")
                .conversationId("c1")
                .tier("medium")
                .modelId("sonnet")
                .cacheHit(false)
                .inputTokens(10)
                .outputTokens(20)
                .latencyMs(30)
                .origin(RequestLogEntry.ORIGIN_CHAT)
                .build();

        service.record(entry);

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(anyString(), args.capture());
        assertThat(args.getValue()[15]).isEqualTo(RequestLogEntry.ORIGIN_CHAT);
        assertThat(args.getValue()[16]).isEqualTo("success");
        assertThat(args.getValue()[17]).isNull();
    }

    @Test
    void databaseFailureIsClassifiedForAnalytics() {
        service.recordFailure("00000000-0000-0000-0000-000000000001", "q", null, 15,
                new DataAccessResourceFailureException("db down"));

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(anyString(), args.capture());
        assertThat(args.getValue()[4]).isEqualTo(RequestLogEntry.ORIGIN_CHAT);
        assertThat(args.getValue()[5]).isEqualTo("db_error");
        assertThat(args.getValue()[6]).isEqualTo("DataAccessResourceFailureException");
    }

    @Test
    void unexpectedFailureIsClassifiedAsInternalError() {
        service.recordFailure("00000000-0000-0000-0000-000000000001", "q",
                RouterService.Tier.HARD, 15, new IllegalStateException("boom"));

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(anyString(), args.capture());
        assertThat(args.getValue()[2]).isEqualTo("hard");
        assertThat(args.getValue()[5]).isEqualTo("internal_error");
        assertThat(args.getValue()[6]).isEqualTo("IllegalStateException");
    }
}
