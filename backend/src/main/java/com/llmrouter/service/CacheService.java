package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.LlmResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.StringJoiner;

/**
 * pgvector 기반 시맨틱 캐시.
 *
 * pgvector 에 접근하는 것은 Gateway 뿐이다 (Decision Log 2026-08-03).
 * FastAPI 는 벡터를 만들어 줄 뿐 DB 를 모른다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class CacheService {

    private final JdbcTemplate jdbc;
    private final RouterClient routerClient;

    @Value("${llm.cache.thresholds.easy}")
    private double easyThreshold;

    @Value("${llm.cache.thresholds.medium}")
    private double mediumThreshold;

    @Value("${llm.cache.thresholds.hard}")
    private double hardThreshold;

    /**
     * 안쪽 서브쿼리에서 HNSW 인덱스로 top-k 를 뽑고, 바깥에서 항목별 threshold 를 건다.
     * threshold 를 안쪽 WHERE 에 직접 넣으면 인덱스를 못 타고 전수 스캔이 된다.
     */
    private static final String SELECT_SQL = """
            SELECT * FROM (
                SELECT id,
                       response,
                       tier,
                       model_id,
                       threshold,
                       1 - (embedding <=> ?::vector) AS similarity
                FROM cache_entry
                ORDER BY embedding <=> ?::vector
                LIMIT 5
            ) t
            WHERE t.similarity >= t.threshold
            ORDER BY t.similarity DESC
            LIMIT 1
            """;

    private static final String INSERT_SQL = """
            INSERT INTO cache_entry
                (question, embedding, response, tier, model_id, embed_model_version, threshold)
            VALUES (?, ?::vector, ?, ?, ?, ?, ?)
            """;

    public Optional<CachedResponse> get(float[] embedding) {
        String vec = toVectorLiteral(embedding);

        List<CachedResponse> rows = jdbc.query(SELECT_SQL,
                (rs, rowNum) -> new CachedResponse(
                        rs.getLong("id"),
                        rs.getString("response"),
                        rs.getString("tier"),
                        rs.getString("model_id"),
                        rs.getDouble("similarity")),
                vec, vec);

        if (rows.isEmpty()) {
            return Optional.empty();
        }

        CachedResponse hit = rows.get(0);
        jdbc.update("UPDATE cache_entry SET hit_count = hit_count + 1, updated_at = now() WHERE id = ?",
                hit.cacheEntryId());

        log.info("캐시 히트 id={} similarity={}", hit.cacheEntryId(), hit.similarity());
        return Optional.of(hit);
    }

    public void put(String message, float[] embedding, LlmResult result) {
        jdbc.update(INSERT_SQL,
                message,
                toVectorLiteral(embedding),
                result.text(),
                result.tier(),
                result.modelId(),
                routerClient.embedModelVersion(),
                thresholdFor(result.tier()));

        log.info("캐시 저장 tier={} model={}", result.tier(), result.modelId());
    }

    /**
     * 캐시 "항목"의 티어로 임계값을 정한다.
     * 들어온 질문의 티어는 조회 시점에 알 수 없다 — classify 는 cache miss 일 때만 부르기 때문.
     * 의미상으로도 "이 항목은 hard 질문의 답이니 재사용에 더 높은 유사도를 요구한다"가 맞다.
     */
    private double thresholdFor(String tier) {
        return switch (tier) {
            case "easy" -> easyThreshold;
            case "hard" -> hardThreshold;
            default -> mediumThreshold;
        };
    }

    /** float[] → pgvector 리터럴 "[0.1,0.2,...]" */
    private String toVectorLiteral(float[] v) {
        StringJoiner sj = new StringJoiner(",", "[", "]");
        for (float f : v) {
            sj.add(Float.toString(f));
        }
        return sj.toString();
    }
}
