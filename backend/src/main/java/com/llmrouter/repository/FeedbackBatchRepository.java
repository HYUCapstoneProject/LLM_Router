package com.llmrouter.repository;

import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.EscalationEvent;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RouterTier;
import com.llmrouter.dto.RetrainingStatus;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.json.JsonMapper;

import java.sql.Array;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * feedback_batch 와 escalation_log 의 내보내기 상태를 다룬다.
 *
 * <p><b>서비스에서 분리한 이유는 트랜잭션 때문이다.</b> {@code @Transactional} 은 프록시가
 * 가로채서 동작하는데, 같은 빈 안에서 메서드를 부르면 프록시를 거치지 않아 애노테이션이
 * <b>조용히 무시된다</b>. 컴파일도 통과하고 테스트도 대개 통과한 뒤, 중간에 예외가 난
 * 순간에만 절반만 커밋된 상태로 드러난다. 여기서는 "행에 batchId 를 찍는 것"과
 * "배치 행을 만드는 것"이 갈라지면 이벤트가 어디에도 안 잡히고 사라지므로, 별도 빈으로
 * 두어 프록시를 반드시 타게 한다.
 */
@Slf4j
@Repository
@RequiredArgsConstructor
public class FeedbackBatchRepository {

    /**
     * 재학습에 쓸 수 있는 이벤트를 고른다. 조건 하나하나가 없으면 라벨이 오염된다.
     *
     * <ul>
     *   <li>{@code user_satisfied} — 체인의 종결 행만. 중간 단계까지 보내면 같은 질문에
     *       final_label 이 다른 "만족" 이벤트가 여러 건 나가고, ML 의 충돌 탐지가
     *       그 질문군을 통째로 격리한다.
     *   <li>{@code response IS NOT NULL} — 예약만 되고 응답이 없는 행은 미완결이다.
     *   <li>{@code was_cache_hit = false} — <b>캐시 오탐과 라우터 과소 라우팅은 다른 신호다.</b>
     *       캐시 히트였다면 그 요청에는 classify 가 돌지도 않았다. 계약의 유효 조건
     *       "router 가 최초 tier/model 을 선택했다"를 만족하지 못한다.
     *   <li>{@code router_model_version IS NOT NULL} — ML 이 필수 컬럼으로 보고 거부한다.
     *   <li>{@code initial_tier}/{@code final_tier} 상향 — 계약이 허용하는 이동은
     *       easy→medium, easy→hard, medium→hard 뿐이다. 우리 체인은 단조 증가라 자연히
     *       성립하지만 스키마가 보장하는 성질이 아니므로 쿼리에서 막는다.
     *   <li>{@code escalation_path} 길이 2 이상 — initialModel/finalModel 을 못 뽑으면
     *       12필드를 채울 수 없다.
     * </ul>
     *
     * <p>오래된 것부터 내보낸다. 배치마다 시기가 섞이면 재학습 결과를 시점으로 해석할 수 없다.
     */
    private static final String ELIGIBLE_PREDICATE = """
            exported_batch_id IS NULL
              AND user_satisfied = true
              AND response IS NOT NULL
              AND was_cache_hit = false
              AND router_model_version IS NOT NULL
              AND array_length(escalation_path, 1) >= 2
              AND array_position(ARRAY['easy','medium','hard'], initial_tier)
                  < array_position(ARRAY['easy','medium','hard'], final_tier)
            """;

    private static final String COLUMNS = """
            id, request_id, question, initial_tier, final_tier,
            escalation_path, escalation_seq, created_at, router_model_version
            """;

    private final JdbcTemplate jdbc;
    private final JsonMapper jsonMapper;

    public int countEligible() {
        Integer count = jdbc.queryForObject(
                "SELECT count(*) FROM escalation_log WHERE " + ELIGIBLE_PREDICATE, Integer.class);
        return count == null ? 0 : count;
    }

    /** Select, claim and freeze the outbound request in one database transaction. */
    @Transactional
    public Optional<FeedbackBatchRequest> claimEligible(int limit, String trigger,
                                                       String dayPrefix, Instant requestedAt) {
        if (limit < 1) throw new IllegalArgumentException("batch size must be positive");
        List<EscalationRow> rows = jdbc.query(
                "SELECT " + COLUMNS + " FROM escalation_log WHERE " + ELIGIBLE_PREDICATE
                        + " ORDER BY created_at, id LIMIT ? FOR UPDATE SKIP LOCKED", ROW_MAPPER, limit);
        if (rows.isEmpty()) return Optional.empty();

        String batchId = dayPrefix + UUID.randomUUID();
        FeedbackBatchRequest request = new FeedbackBatchRequest(batchId, requestedAt,
                rows.stream().map(FeedbackBatchRepository::toEvent).toList());
        // The snapshot includes requestedAt and event values. Later escalation edits cannot
        // mutate an already submitted request, even after a process restart.
        jdbc.update("""
                INSERT INTO feedback_batch (batch_id, event_count, triggered_by, request_payload)
                VALUES (?, ?, ?, ?::jsonb)
                """, batchId, rows.size(), trigger, jsonMapper.writeValueAsString(request));
        String placeholders = IntStream.range(0, rows.size())
                .mapToObj(i -> "?").collect(Collectors.joining(","));
        Object[] params = new Object[rows.size() + 1];
        params[0] = batchId;
        for (int i = 0; i < rows.size(); i++) params[i + 1] = rows.get(i).id();
        int claimed = jdbc.update("UPDATE escalation_log SET exported_batch_id = ?"
                + " WHERE exported_batch_id IS NULL AND id IN (" + placeholders + ")", params);
        if (claimed != rows.size()) {
            throw new IllegalStateException("feedback event claim changed within transaction");
        }
        return Optional.of(request);
    }

    /** No fallback to mutable source rows when a legacy request has no snapshot. */
    public Optional<FeedbackBatchRequest> findRequest(String batchId) {
        return jdbc.query("""
                SELECT request_payload FROM feedback_batch
                 WHERE batch_id = ? AND request_payload IS NOT NULL
                   AND pipeline_run_id IS NULL AND pipeline_status = 'submitted'
                """, (rs, index) -> jsonMapper.readValue(
                        rs.getString("request_payload"), FeedbackBatchRequest.class), batchId)
                .stream().findFirst();
    }

    private static EscalationEvent toEvent(EscalationRow row) {
        String[] path = row.escalationPath();
        return new EscalationEvent("esc-" + row.id(), row.requestId(), row.question(),
                RouterTier.fromLabel(row.initialTier()), path[0],
                RouterTier.fromLabel(row.finalTier()), path[path.length - 1],
                true, row.escalationSeq(), row.createdAt(), row.routerModelVersion(), true);
    }

    // A confirmed acceptance may resolve an earlier uncertain quarantine, but a duplicate
    // 202 must not reset a job whose polling has already advanced beyond queued.
    @Transactional
    public void recordAcceptance(EscalateAcceptance acceptance) {
        jdbc.update("""
                UPDATE feedback_batch
                   SET pipeline_run_id = ?, logical_model_version = ?,
                       candidate_model_version = ?, training_mode = ?,
                       version_policy_mode = ?, status_url = ?,
                       pipeline_status = 'queued', failure_code = NULL,
                       retryable = NULL, polling_complete = false
                 WHERE batch_id = ? AND pipeline_run_id IS NULL
                """,
                acceptance.pipelineRunId(), acceptance.logicalModelVersion(),
                acceptance.candidateModelVersion(), acceptance.trainingMode(),
                acceptance.versionPolicyMode(), acceptance.statusUrl(), acceptance.batchId());
    }

    /**
     * 복구 불가 오류일 때 이벤트를 대기 상태로 되돌린다.
     *
     * <p>배치 행은 지우지 않고 실패로 남긴다. 같은 batchId 를 다시 쓰면 FastAPI 가
     * 409 {@code BATCH_ID_CONFLICT} 를 던지므로, 번호를 소비한 기록이 있어야 한다.
     */
    @Transactional
    public void release(String batchId, String failureCode) {
        // Lock/transition the batch before touching its events. A late failure from another
        // submission attempt must not release events after a 202 has already been recorded.
        int changed = jdbc.update("""
                UPDATE feedback_batch
                   SET pipeline_status = 'failed', failure_code = ?, retryable = false,
                       completed_at = now(), polling_complete = true
                 WHERE batch_id = ? AND pipeline_run_id IS NULL AND pipeline_status = 'submitted'
                """, failureCode, batchId);
        if (changed == 1) {
            jdbc.update("UPDATE escalation_log SET exported_batch_id = NULL WHERE exported_batch_id = ?",
                    batchId);
        }
    }

    /** A conflict may already represent an accepted remote job. Keep its event ownership. */
    public void quarantine(String batchId, String failureCode) {
        jdbc.update("""
                UPDATE feedback_batch SET pipeline_status = 'unknown', failure_code = ?,
                       retryable = false, polling_complete = true
                 WHERE batch_id = ? AND pipeline_run_id IS NULL AND pipeline_status = 'submitted'
                """, failureCode, batchId);
    }

    /** 202 를 받지 못한 배치. 같은 batchId·같은 payload 로 다시 보내야 한다. */
    public List<String> findUnaccepted() {
        return jdbc.queryForList("""
                SELECT batch_id FROM feedback_batch
                 WHERE pipeline_run_id IS NULL AND pipeline_status = 'submitted'
                   AND request_payload IS NOT NULL
                 ORDER BY submitted_at
                """, String.class);
    }

    /**
     * 지금 폴링할 차례가 된 배치.
     *
     * <p>계약의 폴링 정책을 SQL 로 옮긴 것이다 — <b>202 수신 10초 후 시작, 이후 15초 →
     * 30초 → 60초 상한으로 backoff.</b> 애플리케이션에서 배치별 타이머를 들고 있으면
     * 재시작할 때마다 간격이 초기화되고, 인스턴스가 둘이면 서로 다른 주기로 같은 배치를
     * 두드린다. 다음 호출 시점을 DB 의 {@code last_polled_at} 하나로 정하면 재시작 후에도 대기 간격을 유지한다. 여러 인스턴스의 중복 조회는
     * 가능하지만 조회와 동일 상태 반영은 멱등이다.
     *
     * <p>간격을 경과 시간으로 정하는 이유는 학습이 분 단위로 걸리기 때문이다. 제출 직후에는
     * 빨리 확인하고, 길어질수록 느슨하게 본다.
     */
    public List<String> findDuePolls() {
        return jdbc.queryForList("""
                SELECT batch_id FROM feedback_batch
                 WHERE pipeline_run_id IS NOT NULL
                   AND polling_complete = false
                   AND (
                        (last_polled_at IS NULL
                             AND submitted_at < now() - interval '10 seconds')
                     OR (last_polled_at < now() - CASE
                            WHEN submitted_at > now() - interval '1 minute'  THEN interval '15 seconds'
                            WHEN submitted_at > now() - interval '5 minutes' THEN interval '30 seconds'
                            ELSE interval '60 seconds'
                         END)
                   )
                 ORDER BY last_polled_at NULLS FIRST
                """, String.class);
    }

    /**
     * 폴링 결과를 반영한다.
     *
     * <p>{@code completed_at} 은 polling 완료일 때만 채운다. {@code stale=true} 처럼
     * "완료로 확정하지 않는" 구간을 완료로 적으면 실험 타임라인이 어긋난다.
     */
    @Transactional
    public void recordStatus(String batchId, RetrainingStatus status) {
        jdbc.update("""
                UPDATE feedback_batch
                   SET pipeline_status = ?, pipeline_stage = ?,
                       result_outcome = ?, failure_code = ?, retryable = ?,
                       activation_status = ?, polling_complete = ?,
                       last_polled_at = now(),
                       completed_at = CASE WHEN ?::boolean THEN COALESCE(completed_at, now()) ELSE NULL END
                 WHERE batch_id = ?
                """,
                status.status(), status.stage(),
                status.result() == null ? null : status.result().outcome(),
                status.failureCode() != null ? status.failureCode()
                        : "loadFailed".equals(status.activationStatus()) ? "MODEL_ACTIVATION_FAILED" : null,
                status.result() == null ? null : status.result().retryable(),
                status.activationStatus(), status.isPollingComplete(),
                status.isPollingComplete(),
                batchId);
    }

    public void touchPolled(String batchId) {
        jdbc.update("UPDATE feedback_batch SET last_polled_at = now() WHERE batch_id = ?", batchId);
    }

    public List<BatchSummary> recentBatches(int limit) {
        return jdbc.query("""
                SELECT batch_id, event_count, pipeline_status, pipeline_stage,
                       failure_code, activation_status, candidate_model_version,
                       triggered_by, submitted_at, completed_at
                  FROM feedback_batch
                 ORDER BY submitted_at DESC
                 LIMIT ?
                """, (rs, i) -> new BatchSummary(
                        rs.getString("batch_id"),
                        rs.getInt("event_count"),
                        rs.getString("pipeline_status"),
                        rs.getString("pipeline_stage"),
                        rs.getString("failure_code"),
                        rs.getString("activation_status"),
                        rs.getString("candidate_model_version"),
                        rs.getString("triggered_by"),
                        rs.getTimestamp("submitted_at").toInstant(),
                        rs.getTimestamp("completed_at") == null
                                ? null : rs.getTimestamp("completed_at").toInstant()),
                limit);
    }

    private static final RowMapper<EscalationRow> ROW_MAPPER = (rs, index) -> {
        Array path = rs.getArray("escalation_path");
        String[] values = path == null ? new String[0] : (String[]) path.getArray();
        return new EscalationRow(
                rs.getLong("id"),
                rs.getString("request_id"),
                rs.getString("question"),
                rs.getString("initial_tier"),
                rs.getString("final_tier"),
                values,
                rs.getInt("escalation_seq"),
                rs.getTimestamp("created_at").toInstant(),
                rs.getString("router_model_version"));
    };

    public record EscalationRow(
            long id, String requestId, String question,
            String initialTier, String finalTier, String[] escalationPath,
            int escalationSeq, Instant createdAt, String routerModelVersion) {
    }

    public record BatchSummary(
            String batchId, int eventCount, String status, String stage,
            String failureCode, String activationStatus, String candidateModelVersion,
            String triggeredBy, Instant submittedAt, Instant completedAt) {
    }
}
