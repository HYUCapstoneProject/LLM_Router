package com.llmrouter.client;

import com.llmrouter.dto.FeedbackBatchRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.*;
import static org.springframework.test.web.client.response.MockRestResponseCreators.*;

class FastApiRetrainingClientTest {
    private MockRestServiceServer server;
    private FastApiRetrainingClient client;
    private static final String STATUS = """
            {"schemaVersion":"retraining-status-v1","batchId":"batch-1","status":"succeeded",
             "stage":"complete","terminal":true,"stale":false,
             "activation":{"status":"pending","desiredModelVersion":"v2","servingModelVersion":"v1"}}
            """;
    private static final String ACCEPTANCE = """
            {"schemaVersion":"escalation-acceptance-v1","status":"accepted","batchId":"batch-1",
             "pipelineRunId":"projects/p/locations/r/pipelineJobs/j","candidateModelVersion":"v2",
             "statusUrl":"/retraining-runs/batch-1"}
            """;

    @BeforeEach
    void setUp() {
        var builder = RestClient.builder().baseUrl("http://router.internal");
        server = MockRestServiceServer.bindTo(builder).build();
        client = new FastApiRetrainingClient(builder.build(), "test-token");
    }

    @Test
    void authenticatedStatusKeepsPollingAfterTrainingUntilActivation() {
        server.expect(requestTo("http://router.internal/retraining-runs/batch-1"))
                .andExpect(method(HttpMethod.GET)).andExpect(header("Authorization", "Bearer test-token"))
                .andRespond(withSuccess(STATUS, MediaType.APPLICATION_JSON));
        var result = client.status("batch-1");
        assertThat(result.isTerminal()).isTrue();
        assertThat(result.isPollingComplete()).isFalse();
        server.verify();
    }

    @ParameterizedTest
    @ValueSource(strings = {"batchId", "schemaVersion", "status", "stage", "activation", "terminal"})
    void rejectsStatusContractCorruptionBeforeItCanUpdateAnotherBatch(String field) {
        String body = switch (field) {
            case "batchId" -> STATUS.replace("batch-1", "other-batch");
            case "schemaVersion" -> STATUS.replace("retraining-status-v1", "future-schema");
            case "status" -> STATUS.replace("succeeded", "invented");
            case "stage" -> STATUS.replace("complete", "invented");
            case "activation" -> STATUS.replace("pending", "invented");
            default -> STATUS.replace("\"terminal\":true", "\"terminal\":false");
        };
        server.expect(requestTo("http://router.internal/retraining-runs/batch-1"))
                .andRespond(withSuccess(body, MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.status("batch-1"))
                .isInstanceOfSatisfying(RetrainingClientException.class,
                        error -> assertThat(error.code()).isEqualTo("RESPONSE_CONTRACT_MISMATCH"));
    }

    @Test
    void staleTerminalResponseIsAcceptedButCannotEndPolling() {
        server.expect(requestTo("http://router.internal/retraining-runs/batch-1"))
                .andRespond(withSuccess(STATUS.replace("\"stale\":false", "\"stale\":true"),
                        MediaType.APPLICATION_JSON));
        assertThat(client.status("batch-1").isPollingComplete()).isFalse();
    }

    @ParameterizedTest
    @ValueSource(strings = {"http200", "batchId", "schemaVersion", "statusUrl", "pipelineRunId"})
    void malformedAcceptanceDoesNotCountAsAccepted(String defect) {
        String body = switch (defect) {
            case "batchId" -> ACCEPTANCE.replace("\"batchId\":\"batch-1\"", "\"batchId\":\"other\"");
            case "schemaVersion" -> ACCEPTANCE.replace("escalation-acceptance-v1", "wrong");
            case "statusUrl" -> ACCEPTANCE.replace("/retraining-runs/batch-1", "https://external.invalid/other");
            case "pipelineRunId" -> ACCEPTANCE.replace("projects/p/locations/r/pipelineJobs/j", "");
            default -> ACCEPTANCE;
        };
        server.expect(requestTo("http://router.internal/escalate"))
                .andRespond(withStatus(defect.equals("http200")
                        ? org.springframework.http.HttpStatus.OK : org.springframework.http.HttpStatus.ACCEPTED)
                        .contentType(MediaType.APPLICATION_JSON).body(body));
        assertThatThrownBy(() -> client.submit(new FeedbackBatchRequest("batch-1", Instant.now(), List.of())))
                .isInstanceOfSatisfying(RetrainingClientException.class,
                        error -> assertThat(error.code()).isEqualTo("RESPONSE_CONTRACT_MISMATCH"));
    }
}
