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
            RETURNING id
            """;

    /**
     * 에스컬레이션으로 더 나은 답을 얻었을 때 기존 항목을 갱신한다 (기획서 4-3).
     *
     * hit_count 를 0 으로 되돌리는 이유: 누적된 카운트는 방금 "불만족"으로
     * 판정된 답이 벌어들인 것이다. 그대로 두면 캐시 효과 지표가 부풀려진다.
     */
    private static final String REFRESH_SQL = """
            UPDATE cache_entry
               SET response   = ?,
                   tier       = ?,
                   model_id   = ?,
                   threshold  = ?,
                   hit_count  = 0,
                   updated_at = now()
             WHERE id = ?
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

    /**
     * @param question 개인정보가 마스킹된 질문. 임베딩도 같은 문자열로 만든 것이어야 한다 —
     *                 저장 시점과 조회 시점의 마스킹이 다르면 같은 질문인데 유사도가 떨어진다.
     * @return 생성된 행의 id. request_log.cache_entry_id 에 남겨
     *         나중에 /api/escalate 가 이 행을 정확히 지목해 갱신할 수 있게 한다.
     */
    public long put(String question, float[] embedding, LlmResult result) {
        Long id = jdbc.queryForObject(INSERT_SQL, Long.class,
                question,
                toVectorLiteral(embedding),
                result.text(),
                result.tier(),
                result.modelId(),
                routerClient.embedModelVersion(),
                thresholdFor(result.tier()));

        log.info("캐시 저장 id={} tier={} model={}", id, result.tier(), result.modelId());
        return id;
    }

    /**
     * 에스컬레이션으로 만족한 답을 얻으면 해당 캐시 항목을 그 답으로 갱신한다 (기획서 4-3).
     * 갱신하지 않으면 이후 유사 질문에 불만족스러웠던 답이 다시 나간다.
     *
     * 질문과 임베딩은 건드리지 않는다 — 갱신 대상은 "답"이지 "질문"이 아니다.
     */
    public void refresh(long cacheEntryId, LlmResult result) {
        int updated = jdbc.update(REFRESH_SQL,
                result.text(),
                result.tier(),
                result.modelId(),
                thresholdFor(result.tier()),
                cacheEntryId);

        if (updated == 0) {
            // 그 사이 캐시 행이 지워졌을 수 있다. 에스컬레이션 자체는 성공했으므로 실패시키지 않는다.
            log.warn("캐시 갱신 대상 없음 id={}", cacheEntryId);
            return;
        }
        log.info("캐시 갱신 id={} tier={} model={}", cacheEntryId, result.tier(), result.modelId());
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
