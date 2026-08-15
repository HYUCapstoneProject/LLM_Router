package com.llmrouter.service;

import com.llmrouter.dto.EscalationChainState;
import com.llmrouter.dto.EscalationCompletion;
import com.llmrouter.dto.EscalationContext;
import com.llmrouter.dto.EscalationReservation;
import com.llmrouter.dto.ExistingEscalation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.sql.Array;
import java.sql.SQLException;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * escalation_log 입출력.
 *
 * 이 테이블은 "로그"라는 이름과 달리 재학습의 정정 라벨 그 자체다 (기획서 4-4).
 * 여기가 비면 최종 보고서 필수 그림인 "재학습 전후 에스컬레이션율"을 그릴 수 없다.
 *
 * 한 행의 생애: 예약(reserve) → 완료(complete). 또는 예약 → 해제(release).
 * response 가 NULL 인 행은 "진행 중"이며 완결된 기록이 아니다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EscalationLogService {

    private final JdbcTemplate jdbc;

    /**
     * origin='chat' 으로 제한한다. 에스컬레이션 호출이 남긴 request_log 행을
     * 다시 원본으로 삼으면 체인이 두 갈래로 갈라진다.
     * 프론트는 항상 최초 chat 의 requestId 를 보낸다.
     */
    private static final String SELECT_CONTEXT_SQL = """
            SELECT question, conversation_id, tier, cache_hit, cache_entry_id,
                   similarity, router_model_version, embed_model_version
            FROM request_log
            WHERE request_id = ?::uuid AND origin = 'chat'
            """;

    /**
     * 완결된 행만 본다 (response IS NOT NULL).
     *
     * 진행 중인 예약까지 체인으로 치면, 죽은 프로세스가 남긴 유령 예약 때문에
     * 정상 요청이 "이미 상위 티어로 올라갔다"는 이유로 409 를 맞고 영영 막힌다.
     * 예약과의 충돌은 뒤의 reserve() 가 유니크 인덱스로 잡는다.
     */
    private static final String SELECT_CHAIN_SQL = """
            SELECT escalation_seq, initial_tier, final_tier, escalation_path
            FROM escalation_log
            WHERE request_id = ?::uuid AND response IS NOT NULL
            ORDER BY escalation_seq DESC
            LIMIT 1
            """;

    private static final String SELECT_BY_TIER_SQL = """
            SELECT escalation_seq, response, initial_tier, final_tier, final_model_id,
                   escalation_path, input_tokens, output_tokens, coalesce(latency_ms, 0) AS latency_ms
            FROM escalation_log
            WHERE request_id = ?::uuid AND final_tier = ?
            """;

    /** user_satisfied=false 로 넣는다. 완료 전까지는 내보내기 대상이 아니다. */
    private static final String RESERVE_SQL = """
            INSERT INTO escalation_log
                (request_id, question, conversation_id,
                 initial_tier, escalation_path, final_tier, escalation_seq,
                 user_satisfied, router_model_version,
                 was_cache_hit, cache_entry_id, similarity)
            VALUES (?::uuid, ?, ?,
                    ?, ?::text[], ?, ?,
                    false, ?,
                    ?, ?, ?)
            """;

    private static final String COMPLETE_SQL = """
            UPDATE escalation_log
               SET response       = ?,
                   embedding      = ?::vector,
                   final_model_id = ?,
                   input_tokens   = ?,
                   output_tokens  = ?,
                   latency_ms     = ?,
                   user_satisfied = true
             WHERE request_id = ?::uuid AND escalation_seq = ?
            """;

    /** 완료된 행은 지우지 않는다. response IS NULL 가드가 그것을 보장한다. */
    private static final String RELEASE_SQL = """
            DELETE FROM escalation_log
             WHERE request_id = ?::uuid AND escalation_seq = ? AND response IS NULL
            """;

    /**
     * 티어로 좁히지 않고 이 requestId 의 오래된 미완료 예약을 전부 치운다.
     *
     * 예약은 (request_id, final_tier) 뿐 아니라 (request_id, escalation_seq) 로도 충돌한다.
     * haiku -> opus 처럼 중간 단계를 건너뛴 요청이 죽어 hard 예약을 남기면,
     * 그 뒤의 haiku -> sonnet 요청은 seq 충돌로 막히는데 티어가 달라 청소 대상에서 빠진다.
     * 그러면 자가 복구가 안 되고 수동 SQL 로만 풀린다.
     */
    private static final String DELETE_STALE_SQL = """
            DELETE FROM escalation_log
             WHERE request_id = ?::uuid
               AND response IS NULL
               AND created_at < now() - make_interval(secs => ?)
            """;

    /**
     * package-private 로 꺼내둔 이유: 이 매퍼가 드라이버가 돌려주는 타입에 의존하는데
     * 서비스 테스트는 loadContext 자체를 목으로 대체해 버려 그 가정을 검증하지 못한다.
     * ResultSet 을 목으로 만들어 직접 때려볼 수 있게 분리했다.
     */
    static final RowMapper<EscalationContext> CONTEXT_MAPPER = (rs, rowNum) -> new EscalationContext(
            rs.getString("question"),
            rs.getString("conversation_id"),
            rs.getString("tier"),
            rs.getBoolean("cache_hit"),
            nullableLong(rs.getObject("cache_entry_id")),
            nullableDouble(rs.getObject("similarity")),
            rs.getString("router_model_version"),
            rs.getString("embed_model_version"));

    public Optional<EscalationContext> loadContext(String requestId) {
        return jdbc.query(SELECT_CONTEXT_SQL, CONTEXT_MAPPER, requestId).stream().findFirst();
    }

    public Optional<EscalationChainState> loadChain(String requestId) {
        return jdbc.query(SELECT_CHAIN_SQL,
                (rs, rowNum) -> new EscalationChainState(
                        rs.getInt("escalation_seq"),
                        rs.getString("initial_tier"),
                        rs.getString("final_tier"),
                        readTextArray(rs.getArray("escalation_path"))),
                requestId).stream().findFirst();
    }

    public Optional<ExistingEscalation> findByTier(String requestId, String finalTier) {
        return jdbc.query(SELECT_BY_TIER_SQL,
                (rs, rowNum) -> new ExistingEscalation(
                        rs.getInt("escalation_seq"),
                        rs.getString("response"),
                        rs.getString("initial_tier"),
                        rs.getString("final_tier"),
                        rs.getString("final_model_id"),
                        readTextArray(rs.getArray("escalation_path")),
                        rs.getLong("input_tokens"),
                        rs.getLong("output_tokens"),
                        rs.getLong("latency_ms")),
                requestId, finalTier).stream().findFirst();
    }

    /**
     * 자리를 먼저 잡는다. 이미 같은 (requestId, seq) 나 (requestId, finalTier) 가 있으면
     * {@link org.springframework.dao.DuplicateKeyException} 이 난다 — 그게 이 메서드의 요점이다.
     */
    public void reserve(EscalationReservation r) {
        jdbc.update(RESERVE_SQL,
                r.requestId(),
                r.question(),
                r.conversationId(),
                r.initialTier(),
                toTextArrayLiteral(r.escalationPath()),
                r.finalTier(),
                r.escalationSeq(),
                r.routerModelVersion(),
                r.wasCacheHit(),
                r.cacheEntryId(),
                r.similarity());
    }

    /**
     * 예약한 행을 채우고 이 단계를 체인의 종결점으로 만든다. 반드시 한 트랜잭션이어야 한다.
     *
     * 둘이 쪼개지면 같은 질문에 대해 user_satisfied=true 인 행이 둘 이상 남고,
     * 각각 final_tier 가 달라 ML 의 충돌 탐지(같은 정규화 질문의 final_label 이 2종 이상 → 격리)가
     * 그 질문군을 통째로 버린다. 조용히 학습 데이터만 사라지는 종류의 사고다.
     */
    @Transactional
    public void complete(EscalationCompletion c) {
        jdbc.update("UPDATE escalation_log SET user_satisfied = false "
                        + "WHERE request_id = ?::uuid AND escalation_seq <> ?",
                c.requestId(), c.escalationSeq());

        int updated = jdbc.update(COMPLETE_SQL,
                c.response(),
                c.embedding() == null ? null : PgVector.literal(c.embedding()),
                c.finalModelId(),
                c.inputTokens(),
                c.outputTokens(),
                c.latencyMs(),
                c.requestId(),
                c.escalationSeq());

        if (updated == 0) {
            // 예약이 사라졌다. 여기까지 왔다는 건 LLM 요금은 이미 나갔다는 뜻이라 눈에 띄게 남긴다.
            throw new IllegalStateException(
                    "예약된 에스컬레이션 행이 없다 requestId=" + c.requestId() + " seq=" + c.escalationSeq());
        }

        log.info("에스컬레이션 완료 requestId={} seq={} model={} in={} out={}",
                c.requestId(), c.escalationSeq(), c.finalModelId(), c.inputTokens(), c.outputTokens());
    }

    /** LLM 호출이 실패했을 때 예약을 되돌린다. 안 풀면 그 티어로 다시 시도할 수 없다. */
    public void release(String requestId, int escalationSeq) {
        int deleted = jdbc.update(RELEASE_SQL, requestId, escalationSeq);
        if (deleted > 0) {
            log.info("에스컬레이션 예약 해제 requestId={} seq={}", requestId, escalationSeq);
        }
    }

    /**
     * 프로세스가 LLM 호출 도중 죽어 남은 유령 예약을 치운다.
     *
     * 이게 없으면 그 질문이 영구히 막히고 풀 방법이 수동 SQL 뿐이다.
     *
     * @return 치웠으면 true. 아직 진행 중이거나 이미 완료된 행뿐이면 false.
     */
    public boolean deleteStaleReservation(String requestId, Duration timeout) {
        int deleted = jdbc.update(DELETE_STALE_SQL, requestId, timeout.toSeconds());
        if (deleted > 0) {
            log.warn("유령 예약 {}건을 치우고 재시도한다 requestId={} ({} 보다 오래됨)",
                    deleted, requestId, timeout);
        }
        return deleted > 0;
    }

    /**
     * similarity 는 REAL 컬럼이라 pgJDBC 가 {@link Float} 를 돌려준다.
     * {@code (Double) rs.getObject(...)} 로 받으면 ClassCastException 이 나는데,
     * 캐시 미스 유래 행은 이 값이 NULL 이라 캐시 히트 답변을 에스컬레이션할 때만 터진다.
     * 컬럼 타입이 바뀌어도 안전하도록 Number 로 받아 변환한다.
     */
    private static Double nullableDouble(Object value) {
        return value == null ? null : ((Number) value).doubleValue();
    }

    private static Long nullableLong(Object value) {
        return value == null ? null : ((Number) value).longValue();
    }

    private static List<String> readTextArray(Array array) throws SQLException {
        if (array == null) {
            return List.of();
        }
        return Arrays.asList((String[]) array.getArray());
    }

    /**
     * "{haiku,sonnet}" — Postgres text[] 리터럴.
     *
     * 이스케이프를 하지 않는 이유: 원소는 항상 RouterService.toModelName 이 만든
     * haiku / sonnet / opus 셋 중 하나다. 사용자 입력이 여기 닿는 경로가 없다.
     * 그 전제가 깨지면 이스케이프가 필요해진다.
     */
    private static String toTextArrayLiteral(List<String> values) {
        return "{" + String.join(",", values) + "}";
    }
}
