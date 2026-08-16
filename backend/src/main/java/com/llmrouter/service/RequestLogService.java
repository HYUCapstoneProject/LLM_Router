package com.llmrouter.service;

import com.anthropic.errors.AnthropicIoException;
import com.anthropic.errors.AnthropicServiceException;
import com.llmrouter.dto.RequestLogEntry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/** 요청 단위 성공·실패 기록. 비용, 실패율, 지연시간 분석의 원천 데이터다. */
@Slf4j
@Service
@RequiredArgsConstructor
public class RequestLogService {

    private final JdbcTemplate jdbc;

    private static final String INSERT_SQL = """
            INSERT INTO request_log
                (request_id, question, conversation_id, tier, model_id,
                 router_model_version, embed_model_version,
                 cache_hit, similarity, cache_entry_id,
                 input_tokens, output_tokens, latency_ms, fallback_reason,
                 parent_request_id, origin, status, error_type)
            VALUES (?::uuid, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::uuid, ?, ?, ?)
            """;

    private static final String INSERT_FAILURE_SQL = """
            INSERT INTO request_log
                (request_id, question, tier, cache_hit,
                 input_tokens, output_tokens, latency_ms, origin, status, error_type)
            VALUES (?::uuid, ?, ?, false, 0, 0, ?, ?, ?, ?)
            """;

    public void record(RequestLogEntry entry) {
        try {
            jdbc.update(INSERT_SQL,
                    entry.requestId(),
                    entry.question(),
                    entry.conversationId(),
                    entry.tier(),
                    entry.modelId(),
                    entry.routerModelVersion(),
                    entry.embedModelVersion(),
                    entry.cacheHit(),
                    entry.similarity(),
                    entry.cacheEntryId(),
                    entry.inputTokens(),
                    entry.outputTokens(),
                    entry.latencyMs(),
                    entry.fallbackReason(),
                    entry.parentRequestId(),
                    entry.origin(),
                    "success",
                    null);
        } catch (Exception error) {
            // 관측 로그 적재 실패가 사용자 응답을 막으면 안 된다.
            log.error("request_log 적재 실패 requestId={} origin={}",
                    entry.requestId(), entry.origin(), error);
        }
    }

    /** tier는 임베딩·캐시 조회 단계에서 실패하면 null일 수 있다. */
    public void recordFailure(String requestId,
                              String question,
                              RouterService.Tier tier,
                              long latencyMs,
                              Throwable error) {
        try {
            jdbc.update(INSERT_FAILURE_SQL,
                    requestId,
                    question,
                    tier != null ? tier.name().toLowerCase() : null,
                    latencyMs,
                    RequestLogEntry.ORIGIN_CHAT,
                    classifyStatus(error),
                    error.getClass().getSimpleName());
        } catch (Exception persistenceError) {
            log.error("request_log 실패 기록 적재 실패 requestId={}", requestId, persistenceError);
        }
    }

    private String classifyStatus(Throwable error) {
        if (error instanceof AnthropicServiceException || error instanceof AnthropicIoException) {
            return "llm_error";
        }
        if (error instanceof DataAccessException) {
            return "db_error";
        }
        return "internal_error";
    }
}
