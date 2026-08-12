package com.llmrouter.service;

import com.anthropic.errors.AnthropicIoException;
import com.anthropic.errors.AnthropicServiceException;
import com.llmrouter.dto.ChatResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * 요청 단위 기록. 비용 그래프의 원천 데이터다.
 *
 * 콘솔 로그만으로는 집계가 안 된다. 최종 보고서 필수 그림
 * "Router vs always-Opus 비용/정확도"를 그리려면 요청별 토큰이
 * 질의 가능한 형태로 쌓여 있어야 하고, M7 freeze 때 숫자를 다시 만들 수 있어야 한다.
 *
 * 성공뿐 아니라 실패도 남긴다. 성공만 기록하면 실패율과 지연시간 분포를 낼 수 없고,
 * 사용자가 받은 requestId 로 DB 를 조회했을 때 아무것도 안 나온다.
 *
 * 기록 실패가 사용자 응답을 막으면 안 되므로 예외를 삼킨다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RequestLogService {

    private final JdbcTemplate jdbc;

    private static final String INSERT_SQL = """
            INSERT INTO request_log
                (request_id, question, tier, model_id,
                 router_model_version, embed_model_version,
                 cache_hit, similarity, input_tokens, output_tokens,
                 latency_ms, fallback_reason, status, error_type)
            VALUES (?::uuid, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """;

    private static final String INSERT_FAILURE_SQL = """
            INSERT INTO request_log
                (request_id, question, tier, cache_hit,
                 input_tokens, output_tokens, latency_ms, status, error_type)
            VALUES (?::uuid, ?, ?, false, 0, 0, ?, ?, ?)
            """;

    public void record(ChatResponse response,
                       String question,
                       String routerModelVersion,
                       String embedModelVersion,
                       Double similarity,
                       String fallbackReason) {
        try {
            jdbc.update(INSERT_SQL,
                    response.getRequestId(),
                    question,
                    response.getTier(),
                    response.getModelId(),
                    routerModelVersion,
                    embedModelVersion,
                    response.isCacheHit(),
                    similarity,
                    response.getInputTokens(),
                    response.getOutputTokens(),
                    response.getLatency(),
                    fallbackReason,
                    "success",
                    null);
        } catch (Exception e) {
            // 로그 적재 실패로 사용자 응답을 실패시키지 않는다.
            // 다만 실험 데이터가 비므로 반드시 눈에 띄게 남긴다.
            log.error("request_log 적재 실패 requestId={}", response.getRequestId(), e);
        }
    }

    /**
     * @param tier 실패 지점에 따라 null 일 수 있다.
     *             임베딩·캐시 조회 단계에서 터지면 분류를 하기 전이다.
     */
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
                    classifyStatus(error),
                    error.getClass().getSimpleName());
        } catch (Exception e) {
            log.error("request_log 실패 기록 적재 실패 requestId={}", requestId, e);
        }
    }

    /**
     * 실패 원인을 굵게 나눈다. 세부 원인은 error_type(예외 클래스명)에 남는다.
     * API 응답의 code 와 굳이 일치시키지 않는다 —
     * 그 매핑을 두 곳에서 관리하면 한쪽만 바뀌었을 때 어긋난다.
     */
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
