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
        service = new RequestLogService(jdbc, tools.jackson.databind.json.JsonMapper.builder().build());
    }

    @Test
    void successRecordIncludesOriginAndSuccessStatus() {
        RequestLogEntry entry = RequestLogEntry.builder()
                .requestId("00000000-0000-0000-0000-000000000001")
                .question("q")
                .conversationId("c1")
                .tier("medium")
                .modelId("sonnet")
                .routerConfidence(0.87)
                .cacheHit(false)
                .inputTokens(10)
                .outputTokens(20)
                .latencyMs(30)
                .origin(RequestLogEntry.ORIGIN_CHAT)
                .cacheMatchType(RequestLogEntry.MATCH_SEMANTIC)
                .build();

        service.record(entry);

        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(anyString(), args.capture());
        assertThat(args.getValue()[6]).isEqualTo(0.87);
        assertThat(args.getValue()[16]).isEqualTo(RequestLogEntry.ORIGIN_CHAT);
        assertThat(args.getValue()[17]).isEqualTo("success");
        assertThat(args.getValue()[18]).isNull();
        assertThat(args.getValue()[19]).isEqualTo(RequestLogEntry.MATCH_SEMANTIC);
    }

    @Test
    void v1DoesNotRequireTheNewColumnAndBindsEveryPlaceholder() {
        service.record(RequestLogEntry.builder().requestId("00000000-0000-0000-0000-000000000001")
                .question("q").build());
        ArgumentCaptor<String> sql = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(sql.capture(), args.capture());
        assertThat(sql.getValue()).doesNotContain("routing_decision");
        assertThat(sql.getValue().chars().filter(c -> c == '?').count()).isEqualTo(args.getValue().length);
    }

    @Test
    void storesCompleteV2ProvenanceAsJsonWithoutInventingAnOutcome() {
        var decision = new com.llmrouter.client.RouterClassification.DecisionMetadata(
                "pointwise", "policy-v1", "a".repeat(64), "pool-v1",
                java.util.Map.of("low", "small", "medium", "mid", "high", "large"), "medium",
                "independent_success_probabilities", java.util.Map.of("low", 0.3, "medium", 0.8, "high", 0.9));
        service.record(RequestLogEntry.builder().requestId("00000000-0000-0000-0000-000000000001")
                .question("q").routingDecision(decision).build());
        ArgumentCaptor<String> sql = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<Object[]> args = ArgumentCaptor.forClass(Object[].class);
        verify(jdbc).update(sql.capture(), args.capture());
        assertThat(sql.getValue()).contains("routing_decision", "?::jsonb");
        assertThat(sql.getValue().chars().filter(c -> c == '?').count()).isEqualTo(args.getValue().length);
        var mapper = tools.jackson.databind.json.JsonMapper.builder().build();
        var json = mapper.readTree((String) args.getValue()[25]);
        assertThat(json).isEqualTo(mapper.valueToTree(decision));
        assertThat(json.has("success")).isFalse();
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
