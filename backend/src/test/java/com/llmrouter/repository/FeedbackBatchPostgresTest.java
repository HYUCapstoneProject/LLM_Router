package com.llmrouter.repository;

import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.RetrainingStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.transaction.support.TransactionTemplate;
import tools.jackson.databind.json.JsonMapper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Uses an isolated schema in a disposable PostgreSQL database; never an application schema. */
@EnabledIfEnvironmentVariable(named = "TEST_POSTGRES_URL", matches = ".+")
class FeedbackBatchPostgresTest {
    private JdbcTemplate admin;
    private JdbcTemplate jdbc;
    private FeedbackBatchRepository repository;
    private TransactionTemplate tx;
    private String schema;
    private static final Instant NOW = Instant.parse("2026-09-26T00:00:00Z");

    @BeforeEach
    void setUp() throws Exception {
        String url = System.getenv("TEST_POSTGRES_URL");
        String user = System.getenv().getOrDefault("TEST_POSTGRES_USER", "postgres");
        String password = System.getenv().getOrDefault("TEST_POSTGRES_PASSWORD", "");
        admin = new JdbcTemplate(new DriverManagerDataSource(url, user, password));
        schema = "feedback_test_" + UUID.randomUUID().toString().replace("-", "");
        admin.execute("CREATE SCHEMA " + schema);
        var source = new DriverManagerDataSource(url + (url.contains("?") ? "&" : "?")
                + "currentSchema=" + schema, user, password);
        jdbc = new JdbcTemplate(source);
        tx = new TransactionTemplate(new DataSourceTransactionManager(source));
        repository = new FeedbackBatchRepository(jdbc, JsonMapper.builder().findAndAddModules().build());
        jdbc.execute("""
                CREATE TABLE escalation_log (
                  id BIGSERIAL PRIMARY KEY, request_id UUID NOT NULL, question TEXT NOT NULL,
                  initial_tier TEXT, final_tier TEXT, escalation_path TEXT[], escalation_seq INTEGER,
                  created_at TIMESTAMPTZ DEFAULT now(), router_model_version TEXT,
                  exported_batch_id TEXT, user_satisfied BOOLEAN DEFAULT true,
                  response TEXT DEFAULT 'answer', was_cache_hit BOOLEAN DEFAULT false)
                """);
        String init = Files.readString(Path.of("../infra/postgres/init/01-schema.sql"));
        String table = init.substring(init.indexOf("CREATE TABLE feedback_batch ("));
        jdbc.execute(table);
    }

    @AfterEach
    void cleanUp() {
        if (admin != null && schema != null) admin.execute("DROP SCHEMA " + schema + " CASCADE");
    }

    private void seed(int count) {
        for (int i = 0; i < count; i++) {
            jdbc.update("""
                    INSERT INTO escalation_log (request_id, question, initial_tier, final_tier,
                        escalation_path, escalation_seq, router_model_version)
                    VALUES (?::uuid, ?, 'easy', 'hard', ARRAY['haiku','sonnet','opus'], 2, 'router-v1')
                    """, UUID.randomUUID().toString(), "question-" + i);
        }
    }

    private FeedbackBatchRequest claim(int limit) {
        return repository.claimEligible(limit, "manual", "feedback-20260926-", NOW).orElseThrow();
    }

    @Test
    void concurrentTransactionsClaimDisjointEventsAndKeepFrozenPayloadAfterSourceMutation() throws Exception {
        seed(3);
        var claimed = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> tx.execute(status -> {
                var request = claim(2);
                claimed.countDown();
                try {
                    if (!release.await(10, TimeUnit.SECONDS)) throw new IllegalStateException("test latch timeout");
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(error);
                }
                return request;
            }));
            assertThat(claimed.await(10, TimeUnit.SECONDS)).isTrue();
            FeedbackBatchRequest second;
            try {
                second = executor.submit(() -> tx.execute(status -> claim(2))).get(10, TimeUnit.SECONDS);
            } finally {
                release.countDown();
            }
            var original = first.get(10, TimeUnit.SECONDS);
            assertThat(original.events()).hasSize(2);
            assertThat(second.events()).hasSize(1);
            assertThat(original.events()).extracting("eventId")
                    .doesNotContainAnyElementsOf(second.events().stream().map(e -> e.eventId()).toList());
            jdbc.update("UPDATE escalation_log SET question = 'edited', user_satisfied = false");
            assertThat(repository.findRequest(original.batchId()).orElseThrow()).isEqualTo(original);
            assertThat(repository.countEligible()).isZero();
        }
    }

    @Test
    void rollbackReleasesClaimAndSnapshotTogether() {
        seed(1);
        assertThatThrownBy(() -> tx.execute(status -> {
            claim(1);
            throw new IllegalStateException("simulated crash before commit");
        })).isInstanceOf(IllegalStateException.class);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM feedback_batch", Integer.class)).isZero();
        assertThat(repository.countEligible()).isEqualTo(1);
    }

    @Test
    void staleAndPendingRemainPollableButLoadFailureIsRecordedAndStopsPolling() {
        seed(1);
        var request = tx.execute(status -> claim(1));
        jdbc.update("UPDATE feedback_batch SET pipeline_run_id='job', submitted_at=now()-interval '1 hour'");
        repository.recordStatus(request.batchId(), status(true, "active"));
        jdbc.update("UPDATE feedback_batch SET last_polled_at=now()-interval '1 hour'");
        assertThat(repository.findDuePolls()).containsExactly(request.batchId());
        repository.recordStatus(request.batchId(), status(false, "pending"));
        assertThat(jdbc.queryForObject("SELECT completed_at IS NULL FROM feedback_batch", Boolean.class)).isTrue();
        repository.recordStatus(request.batchId(), status(false, "loadFailed"));
        assertThat(repository.findDuePolls()).isEmpty();
        assertThat(jdbc.queryForObject("SELECT failure_code FROM feedback_batch", String.class))
                .isEqualTo("MODEL_ACTIVATION_FAILED");
    }

    @Test
    void lateFailuresCannotUndoAnAcceptedBatchAndAcceptanceResolvesEarlierQuarantine() throws Exception {
        seed(3);
        for (boolean releaseEvents : new boolean[]{false, true}) {
            var request = tx.execute(status -> claim(1));
            var accepted = new CountDownLatch(1);
            var failureStarted = new CountDownLatch(1);
            var commitAcceptance = new CountDownLatch(1);
            var acceptance = acceptance(request.batchId());
            try (var executor = Executors.newFixedThreadPool(2)) {
                var first = executor.submit(() -> tx.execute(status -> {
                    repository.recordAcceptance(acceptance);
                    accepted.countDown();
                    try {
                        if (!commitAcceptance.await(10, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("test latch timeout");
                        }
                    } catch (InterruptedException error) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(error);
                    }
                    return null;
                }));
                assertThat(accepted.await(10, TimeUnit.SECONDS)).isTrue();
                var lateFailure = executor.submit(() -> tx.execute(status -> {
                    failureStarted.countDown();
                    if (releaseEvents) repository.release(request.batchId(), "REQUEST_VALIDATION_FAILED");
                    else repository.quarantine(request.batchId(), "BATCH_ID_CONFLICT");
                    return null;
                }));
                try {
                    assertThat(failureStarted.await(10, TimeUnit.SECONDS)).isTrue();
                } finally {
                    commitAcceptance.countDown();
                }
                first.get(10, TimeUnit.SECONDS);
                lateFailure.get(10, TimeUnit.SECONDS);
            }
            assertThat(jdbc.queryForObject("""
                    SELECT pipeline_run_id = 'job' AND pipeline_status = 'queued'
                           AND polling_complete = false AND failure_code IS NULL
                      FROM feedback_batch WHERE batch_id = ?
                    """, Boolean.class, request.batchId())).isTrue();
            assertThat(jdbc.queryForObject("SELECT count(*) FROM escalation_log WHERE exported_batch_id = ?",
                    Integer.class, request.batchId())).isEqualTo(1);
            repository.recordStatus(request.batchId(), status(false, "active"));
            repository.recordAcceptance(acceptance);
            assertThat(jdbc.queryForObject("SELECT polling_complete FROM feedback_batch WHERE batch_id = ?",
                    Boolean.class, request.batchId())).isTrue();
        }
        var uncertain = tx.execute(status -> claim(1));
        repository.quarantine(uncertain.batchId(), "BATCH_ID_CONFLICT");
        repository.recordAcceptance(acceptance(uncertain.batchId()));
        assertThat(jdbc.queryForObject("SELECT pipeline_status FROM feedback_batch WHERE batch_id = ?",
                String.class, uncertain.batchId())).isEqualTo("queued");
        assertThat(repository.countEligible()).isZero();
    }

    private EscalateAcceptance acceptance(String batchId) {
        return new EscalateAcceptance("escalation-acceptance-v1", "accepted", batchId,
                "job", "2.1", "router-v2", "minor_fine_tune", "successfulHttpRequest",
                "/retraining-runs/" + batchId);
    }

    @Test
    void migrationQuarantinesLegacyUnacceptedBodiesWithoutReleasingEventsAndIsRepeatable() throws Exception {
        seed(1);
        jdbc.update("UPDATE escalation_log SET exported_batch_id='legacy'");
        jdbc.execute("DROP INDEX idx_feedback_batch_pending");
        jdbc.execute("ALTER TABLE feedback_batch DROP COLUMN request_payload, DROP COLUMN polling_complete");
        jdbc.update("INSERT INTO feedback_batch(batch_id,event_count) VALUES ('legacy', 1)");
        String migration = Files.readString(Path.of("../infra/postgres/migrations/10-feedback-batch-snapshot.sql"));
        jdbc.execute(migration);
        jdbc.execute(migration);
        assertThat(repository.findUnaccepted()).isEmpty();
        assertThat(repository.findRequest("legacy")).isEmpty();
        assertThat(jdbc.queryForObject("SELECT failure_code FROM feedback_batch", String.class))
                .isEqualTo("LEGACY_PAYLOAD_UNAVAILABLE");
        assertThat(jdbc.queryForObject("SELECT exported_batch_id FROM escalation_log", String.class))
                .isEqualTo("legacy");
    }

    private RetrainingStatus status(boolean stale, String activation) {
        return new RetrainingStatus("retraining-status-v1", "unused", "succeeded", "complete", true, stale,
                null, new RetrainingStatus.Activation(activation, "v2", "v2"));
    }
}
