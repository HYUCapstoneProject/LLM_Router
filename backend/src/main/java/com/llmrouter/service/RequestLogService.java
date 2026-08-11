package com.llmrouter.service;

import com.llmrouter.dto.ChatResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * 요청 단위 기록. 비용 그래프의 원천 데이터다.
 *
 * 콘솔 로그만으로는 집계가 안 된다. 최종 보고서 필수 그림
 * "Router vs always-Opus 비용/정확도"를 그리려면 요청별 토큰이
 * 질의 가능한 형태로 쌓여 있어야 하고, M7 freeze 때 숫자를 다시 만들 수 있어야 한다.
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
                 latency_ms, fallback_reason)
            VALUES (?::uuid, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    fallbackReason);
        } catch (Exception e) {
            // 로그 적재 실패로 사용자 응답을 실패시키지 않는다.
            // 다만 실험 데이터가 비므로 반드시 눈에 띄게 남긴다.
            log.error("request_log 적재 실패 requestId={}", response.getRequestId(), e);
        }
    }
}
