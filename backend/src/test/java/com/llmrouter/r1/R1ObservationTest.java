package com.llmrouter.r1;

import com.llmrouter.dto.R1Observation;
import com.llmrouter.repository.R1ObservationRepository;
import com.llmrouter.service.R1ObservationService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import tools.jackson.databind.json.JsonMapper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class R1ObservationTest {
    @TempDir Path root;
    static final JsonMapper JSON = JsonMapper.builder().build();
    static final UUID REQUEST = UUID.fromString("00000000-0000-4000-8000-000000000001");
    static final Instant CALL = Instant.parse("2026-09-26T00:00:00Z");

    static VerifiedR1PolicyBinding binding(Path root) throws Exception {
        Files.createDirectories(root);
        byte[] policy;
        try (var source = R1ObservationTest.class.getResourceAsStream("/contracts/r1-routing-policy-v2.json")) {
            policy = source.readAllBytes();
        }
        String sha = VerifiedR1PolicyBinding.sha256(policy);
        byte[] manifest = JSON.writeValueAsBytes(Map.of("schema_version", "model-package-v2",
                "gate_status", "approved", "objective", "pointwise", "model_version", "router-r1-fixture",
                "routing_policy_file", "routing_policy.json", "routing_policy_sha256", sha,
                "files", Map.of("routing_policy.json", sha)));
        Path manifestPath = root.resolve("model_manifest.json");
        Path policyPath = root.resolve("routing_policy.json");
        Files.write(manifestPath, manifest);
        Files.write(policyPath, policy);
        return VerifiedR1PolicyBinding.fromPinnedPackage(manifestPath, policyPath,
                VerifiedR1PolicyBinding.sha256(manifest));
    }

    static R1ObservationRepository.Invocation invocation(boolean cache, String model, String fallback) {
        return new R1ObservationRepository.Invocation(1, REQUEST, REQUEST, "original private question", "easy",
                model, cache, "success", "chat", CALL, "router-r1-fixture", fallback,
                "original private question", "easy", "actual-low", false, "success", "router-r1-fixture", null);
    }

    static R1ObservationService.Assessment assessment() {
        return new R1ObservationService.Assessment("observation-1", REQUEST, "actual-low", false,
                "reference_evaluator", "assessment-1", CALL.plusSeconds(60), "target-v1", "sanitized question", true);
    }

    @Test
    void verifiedV1CallAndExplicitNegativeProduceExactCrossRepoArtifactWithoutInventedScores() throws Exception {
        var repository = mock(R1ObservationRepository.class);
        when(repository.findInvocation(REQUEST)).thenReturn(Optional.of(invocation(false, "actual-low", null)));
        when(repository.save(any(), any(), anyString(), anyString(), anyString())).thenAnswer(call -> call.getArgument(0));
        var service = new R1ObservationService(repository, JSON, "target-v1");
        var observation = service.record(assessment(), binding(root));
        var artifact = JSON.valueToTree(new R1Observation.Batch(java.util.List.of(observation)));
        try (var resource = getClass().getResourceAsStream("/contracts/r1-observation-batch-v1.json")) {
            assertThat(artifact).isEqualTo(JSON.readTree(resource));
        }
        assertThat(artifact.toString()).doesNotContain("actionScores", "userSatisfied", "success_high");
        assertThat(observation.success()).isFalse();
    }

    @Test
    void missingCacheFallbackOrWrongActualModelCannotBecomeAnObservation() throws Exception {
        var policy = binding(root);
        for (var call : java.util.List.of(invocation(true, "actual-low", null),
                invocation(false, "other-model", null), invocation(false, "actual-low", "router_unavailable"))) {
            var repository = mock(R1ObservationRepository.class);
            when(repository.findInvocation(REQUEST)).thenReturn(Optional.of(call));
            var service = new R1ObservationService(repository, JSON, "target-v1");
            assertThatThrownBy(() -> service.record(assessment(), policy)).isInstanceOf(IllegalArgumentException.class);
            verify(repository, never()).save(any(), any(), anyString(), anyString(), anyString());
        }
    }

    @Test
    void changedPolicyBytesAndUnpinnedManifestAreRejected() throws Exception {
        var policy = binding(root);
        Files.writeString(root.resolve("routing_policy.json"), "{}");
        assertThatThrownBy(() -> VerifiedR1PolicyBinding.fromPinnedPackage(root.resolve("model_manifest.json"),
                root.resolve("routing_policy.json"), policy.manifestSha256()))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("policy file SHA");
        assertThatThrownBy(() -> VerifiedR1PolicyBinding.fromPinnedPackage(root.resolve("model_manifest.json"),
                root.resolve("routing_policy.json"), "0".repeat(64)))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("manifest SHA");
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "1", "\"false\"", "null"})
    void successMustBeAnExplicitBoolean(String token) throws Exception {
        String fixture;
        try (var stream = getClass().getResourceAsStream("/contracts/r1-observation-batch-v1.json")) {
            fixture = new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
        String corrupted = JSON.readTree(fixture).get("observations").get(0).toString()
                .replace("\"success\":false", "\"success\":" + token);
        assertThatThrownBy(() -> JSON.readValue(corrupted, R1Observation.class)).isInstanceOf(RuntimeException.class);
    }

    @Test
    void implicitCompletionOrMissingRedactionCannotCreateAssessment() {
        var a = assessment();
        assertThatThrownBy(() -> new R1ObservationService.Assessment(a.eventId(), a.requestId(), a.candidateModelId(),
                true, "chain_ended", a.assessmentId(), a.assessedAt(), a.targetContractVersion(), "q", true))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new R1ObservationService.Assessment(a.eventId(), a.requestId(), a.candidateModelId(),
                a.success(), a.assessmentSource(), a.assessmentId(), a.assessedAt(), a.targetContractVersion(), "q", false))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(strings = {"bad/id", "bad:id", "한글-id", "trailing\n"})
    void eventIdUsesTheSameRestrictedAlphabetAsPythonConsumers(String eventId) {
        var a = assessment();
        assertThatThrownBy(() -> new R1ObservationService.Assessment(eventId, a.requestId(), a.candidateModelId(),
                a.success(), a.assessmentSource(), a.assessmentId(), a.assessedAt(), a.targetContractVersion(),
                a.redactedQuestion(), true)).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("event_id");
    }

    @ParameterizedTest
    @ValueSource(strings = {"\u0085", "\u00a0", "\u2007", "\u202f"})
    void unicodeWhitespaceCannotBecomeAQuestionOrAnIdentifier(String whitespace) {
        var a = assessment();
        assertThatThrownBy(() -> new R1ObservationService.Assessment(a.eventId(), a.requestId(), a.candidateModelId(),
                a.success(), a.assessmentSource(), a.assessmentId(), a.assessedAt(), a.targetContractVersion(),
                whitespace, true)).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("question");
        assertThatThrownBy(() -> R1Observation.identifier("id" + whitespace + "value", "identifier"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void exportedUtcTimestampMustFitPythonDatetimeRange() {
        assertThatThrownBy(() -> new R1Observation(R1Observation.SCHEMA, "event", REQUEST, "question",
                "router", "a".repeat(64), "pool", "candidate", "low", true, "human_review", "assessment",
                Instant.parse("+10000-01-01T00:00:00Z"), true)).isInstanceOf(IllegalArgumentException.class);
    }
}
