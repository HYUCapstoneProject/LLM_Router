package com.llmrouter.repository;

import com.llmrouter.dto.R1Observation;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.json.JsonMapper;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static com.llmrouter.dto.R1Observation.require;

@Repository
@RequiredArgsConstructor
@ConditionalOnProperty(name = "llm.retraining.observation.enabled", havingValue = "true")
public class R1ObservationRepository {
    private final JdbcTemplate jdbc;
    private final JsonMapper jsonMapper;

    public Optional<Invocation> findInvocation(UUID requestId) {
        return jdbc.query("""
                SELECT c.id, c.request_id, c.question, c.tier, c.model_id, c.cache_hit,
                       c.status, c.origin, c.created_at, c.router_model_version, c.fallback_reason,
                       r.request_id AS root_request_id, r.question AS root_question,
                       r.tier AS root_tier, r.model_id AS root_model_id,
                       r.cache_hit AS root_cache_hit, r.status AS root_status,
                       r.router_model_version AS root_router_model_version,
                       r.fallback_reason AS root_fallback_reason
                  FROM request_log c
                  JOIN request_log r ON r.request_id = CASE
                        WHEN c.origin = 'chat' THEN c.request_id ELSE c.parent_request_id END
                       AND r.origin = 'chat'
                 WHERE c.request_id = ?::uuid
                """, (rs, index) -> new Invocation(rs.getLong("id"),
                UUID.fromString(rs.getString("request_id")), UUID.fromString(rs.getString("root_request_id")),
                rs.getString("question"), rs.getString("tier"), rs.getString("model_id"),
                rs.getBoolean("cache_hit"), rs.getString("status"), rs.getString("origin"),
                rs.getTimestamp("created_at").toInstant(), rs.getString("router_model_version"),
                rs.getString("fallback_reason"), rs.getString("root_question"), rs.getString("root_tier"),
                rs.getString("root_model_id"), rs.getBoolean("root_cache_hit"), rs.getString("root_status"),
                rs.getString("root_router_model_version"), rs.getString("root_fallback_reason")),
                requestId.toString()).stream().findFirst();
    }

    /** Preserve the first explicit assessment. Retries are exact-content idempotent. */
    @Transactional
    public R1Observation save(R1Observation observation, Invocation invocation, String manifestSha256,
                              String targetContractVersion, String originalQuestionSha256) {
        String payload = jsonMapper.writeValueAsString(observation);
        int inserted = jdbc.update("""
                INSERT INTO r1_observation (event_id, assessment_id, request_id, request_log_id,
                    root_request_id, candidate_model_id, policy_manifest_sha256,
                    target_contract_version, original_question_sha256, observation_payload)
                VALUES (?, ?, ?::uuid, ?, ?::uuid, ?, ?, ?, ?, ?::jsonb)
                ON CONFLICT DO NOTHING
                """, observation.eventId(), observation.assessmentId(), observation.requestId().toString(),
                invocation.id(), invocation.rootRequestId().toString(), observation.candidateModelId(),
                manifestSha256, targetContractVersion, originalQuestionSha256, payload);
        if (inserted == 0) {
            List<Boolean> identical = jdbc.query("""
                    SELECT observation_payload = ?::jsonb AND policy_manifest_sha256 = ?
                           AND target_contract_version = ? AND original_question_sha256 = ? AS identical
                      FROM r1_observation
                     WHERE event_id = ? OR assessment_id = ?
                        OR (request_id = ?::uuid AND candidate_model_id = ?)
                     FOR UPDATE
                    """, (rs, index) -> rs.getBoolean("identical"), payload, manifestSha256,
                    targetContractVersion, originalQuestionSha256, observation.eventId(), observation.assessmentId(),
                    observation.requestId().toString(), observation.candidateModelId());
            require(identical.size() == 1 && identical.getFirst(), "assessment identity or outcome conflict");
        }
        return observation;
    }

    /** Explicit selection makes repeated exports stable as new observations arrive. */
    public List<R1Observation> findForExport(List<String> eventIds) {
        require(eventIds != null && !eventIds.isEmpty() && eventIds.size() <= 1000, "export event IDs");
        require(eventIds.stream().distinct().count() == eventIds.size(), "unique export event IDs");
        return eventIds.stream().sorted().map(id -> {
            List<R1Observation> rows = jdbc.query("SELECT observation_payload FROM r1_observation WHERE event_id = ?",
                    (rs, index) -> jsonMapper.readValue(rs.getString("observation_payload"), R1Observation.class), id);
            require(rows.size() == 1, "missing export event " + id);
            return rows.getFirst();
        }).toList();
    }

    public record Invocation(long id, UUID requestId, UUID rootRequestId, String question, String tier,
                             String modelId, boolean cacheHit, String status, String origin,
                             Instant recordedAt, String routerModelVersion, String fallbackReason,
                             String rootQuestion, String rootTier, String rootModelId, boolean rootCacheHit,
                             String rootStatus, String rootRouterModelVersion, String rootFallbackReason) { }
}
