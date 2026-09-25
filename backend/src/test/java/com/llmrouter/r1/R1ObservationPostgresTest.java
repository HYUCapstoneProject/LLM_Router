package com.llmrouter.r1;

import com.llmrouter.repository.R1ObservationRepository;
import com.llmrouter.service.R1ObservationService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.transaction.support.TransactionTemplate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

@EnabledIfEnvironmentVariable(named = "TEST_POSTGRES_URL", matches = ".+")
class R1ObservationPostgresTest {
    @TempDir Path files;
    private JdbcTemplate admin;
    private JdbcTemplate jdbc;
    private TransactionTemplate tx;
    private R1ObservationService service;
    private VerifiedR1PolicyBinding binding;
    private String schema;

    @BeforeEach
    void setUp() throws Exception {
        String url = System.getenv("TEST_POSTGRES_URL");
        String user = System.getenv().getOrDefault("TEST_POSTGRES_USER", "postgres");
        String password = System.getenv().getOrDefault("TEST_POSTGRES_PASSWORD", "");
        admin = new JdbcTemplate(new DriverManagerDataSource(url, user, password));
        schema = "r1_test_" + UUID.randomUUID().toString().replace("-", "");
        admin.execute("CREATE SCHEMA " + schema);
        var source = new DriverManagerDataSource(url + (url.contains("?") ? "&" : "?") + "currentSchema=" + schema,
                user, password);
        jdbc = new JdbcTemplate(source);
        tx = new TransactionTemplate(new DataSourceTransactionManager(source));
        jdbc.execute("""
                CREATE TABLE request_log (id BIGSERIAL PRIMARY KEY, request_id UUID UNIQUE,
                    parent_request_id UUID, question TEXT, tier TEXT, model_id TEXT, cache_hit BOOLEAN,
                    status TEXT, origin TEXT, created_at TIMESTAMPTZ, router_model_version TEXT, fallback_reason TEXT)
                """);
        String migration = Files.readString(Path.of("../infra/postgres/migrations/11-r1-observation.sql"));
        jdbc.execute(migration);
        jdbc.execute(migration);
        jdbc.update("""
                INSERT INTO request_log (request_id, question, tier, model_id, cache_hit, status,
                    origin, created_at, router_model_version)
                VALUES (?::uuid, 'private original question', 'easy', 'actual-low', false,
                    'success', 'chat', '2026-09-26T00:00:00Z'::timestamptz, 'router-r1-fixture')
                """, R1ObservationTest.REQUEST.toString());
        var repository = new R1ObservationRepository(jdbc, R1ObservationTest.JSON);
        service = new R1ObservationService(repository, R1ObservationTest.JSON, "target-v1");
        binding = R1ObservationTest.binding(files);
    }

    @AfterEach
    void cleanUp() {
        if (admin != null && schema != null) admin.execute("DROP SCHEMA " + schema + " CASCADE");
    }

    @Test
    void explicitOutcomeIsImmutableAndExportsOnlyRedactedContent() {
        var a = R1ObservationTest.assessment();
        var first = tx.execute(status -> service.record(a, binding));
        var retry = tx.execute(status -> service.record(a, binding));
        assertThat(retry).isEqualTo(first);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM r1_observation", Integer.class)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT request_log_id FROM r1_observation", Long.class)).isEqualTo(1L);
        assertThat(jdbc.queryForObject("SELECT policy_manifest_sha256 FROM r1_observation", String.class))
                .isEqualTo(binding.manifestSha256());
        String exported = service.exportJson(List.of(a.eventId()));
        assertThat(exported).isEqualTo(service.exportJson(List.of(a.eventId())));
        assertThat(exported).contains("sanitized question", "\"success\":false")
                .doesNotContain("private original", "actionScores", "observation_mask");
        var conflict = new R1ObservationService.Assessment("different-event", a.requestId(), a.candidateModelId(),
                true, a.assessmentSource(), "different-assessment", a.assessedAt(), a.targetContractVersion(),
                a.redactedQuestion(), true);
        assertThatThrownBy(() -> tx.execute(status -> service.record(conflict, binding)))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("outcome conflict");
        assertThat(service.exportJson(List.of(a.eventId()))).isEqualTo(exported);
    }

    @Test
    void simultaneousIdenticalImportsPersistOneObservation() throws Exception {
        var inserted = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> tx.execute(status -> {
                var row = service.record(R1ObservationTest.assessment(), binding);
                inserted.countDown();
                try {
                    if (!release.await(10, TimeUnit.SECONDS)) throw new IllegalStateException("test timeout");
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(error);
                }
                return row;
            }));
            assertThat(inserted.await(10, TimeUnit.SECONDS)).isTrue();
            var second = executor.submit(() -> tx.execute(status -> service.record(R1ObservationTest.assessment(), binding)));
            release.countDown();
            assertThat(first.get(10, TimeUnit.SECONDS)).isEqualTo(second.get(10, TimeUnit.SECONDS));
            assertThat(jdbc.queryForObject("SELECT count(*) FROM r1_observation", Integer.class)).isEqualTo(1);
        } finally {
            release.countDown();
        }
    }
}
