package com.llmrouter.client;

import com.llmrouter.service.RouterService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.ExpectedCount.once;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

/** Consumes real serving JSON produced from ML synthetic packages by check-routing-v2-cross-repo.py. */
class RoutingV2CrossRepoContractTest {
    private static JsonNode fixtures() throws IOException {
        try (InputStream stream = RoutingV2CrossRepoContractTest.class
                .getResourceAsStream("/contracts/routing-v2-cross-repo.json")) {
            assertThat(stream).isNotNull();
            return JsonMapper.builder().build().readTree(stream);
        }
    }

    private static JsonNode fixture(String objective) throws IOException {
        JsonNode root = fixtures();
        assertThat(root.get("syntheticOnly").booleanValue()).isTrue();
        return root.get("cases").valueStream()
                .filter(item -> objective.equals(item.get("objective").stringValue()))
                .findFirst().orElseThrow();
    }

    private static FastApiRouterClient client(RestClient client, JsonNode response, String hash, String highModel) {
        JsonNode models = response.get("candidateModelIds");
        return new FastApiRouterClient(client, "synthetic-secret", 1, "embed-default", "router-default",
                "v2", hash, response.get("candidatePoolVersion").stringValue(),
                models.get("low").stringValue(), models.get("medium").stringValue(), highModel,
                response.get("objective").stringValue());
    }

    @ParameterizedTest
    @ValueSource(strings = {"ordinal", "pointwise", "pairwise"})
    void acceptsProducerPackageThroughVerifiedServingResponse(String objective) throws IOException {
        JsonNode fixture = fixture(objective);
        JsonNode response = fixture.get("response");
        RestClient.Builder builder = RestClient.builder().baseUrl("http://synthetic-router");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        FastApiRouterClient client = client(builder.build(), response,
                fixture.get("canonicalPolicySha256").stringValue(),
                response.get("candidateModelIds").get("high").stringValue());
        server.expect(once(), requestTo("http://synthetic-router/router/classify/v2"))
                .andRespond(withSuccess(response.toString(), MediaType.APPLICATION_JSON));
        RouterClassification actual = client.classify("q");
        assertThat(actual.tier()).isEqualTo(RouterService.Tier.valueOf(
                response.get("tier").stringValue().toUpperCase(Locale.ROOT)));
        assertThat(actual.modelVersion()).isEqualTo(response.get("modelVersion").stringValue());
        assertThat(actual.decisionMetadata().objective()).isEqualTo(objective);
        assertThat(actual.decisionMetadata().policySha256())
                .isEqualTo(fixture.get("canonicalPolicySha256").stringValue())
                .isNotEqualTo(fixture.get("policyFileSha256").stringValue());
        if (objective.equals("pairwise")) {
            assertThat(actual.confidence()).isNull();
            assertThat(actual.decisionMetadata().scoreSemantics()).isEqualTo("relative_quality_scores");
        } else {
            assertThat(actual.confidence()).isEqualTo(response.get("confidence").doubleValue());
        }
        if (objective.equals("ordinal")) {
            assertThat(fixture.get("checkpointConfigSha256").stringValue())
                    .isNotEqualTo(fixture.get("packagedConfigSha256").stringValue());
            assertThat(fixture.get("actualOutputLabels")).isEqualTo(fixture.get("currentTierAliases"));
        } else {
            assertThat(fixture.get("checkpointConfigSha256").stringValue())
                    .isEqualTo(fixture.get("packagedConfigSha256").stringValue());
            assertThat(fixture.get("actualOutputLabels").get(0).stringValue()).isEqualTo("low");
            assertThat(fixture.get("currentTierAliases").get(0).stringValue()).isEqualTo("easy");
        }
        server.verify();
    }

    @ParameterizedTest
    @ValueSource(strings = {"ordinal", "pointwise", "pairwise"})
    void refusesFileDigestInPlaceOfCanonicalPolicyIdentity(String objective) throws IOException {
        JsonNode fixture = fixture(objective);
        JsonNode response = fixture.get("response");
        RestClient.Builder builder = RestClient.builder().baseUrl("http://synthetic-router");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        FastApiRouterClient client = client(builder.build(), response,
                fixture.get("policyFileSha256").stringValue(),
                response.get("candidateModelIds").get("high").stringValue());
        server.expect(once(), requestTo("http://synthetic-router/router/classify/v2"))
                .andRespond(withSuccess(response.toString(), MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class).hasMessageContaining("policySha256");
        server.verify();
    }

    @ParameterizedTest
    @ValueSource(strings = {"ordinal", "pointwise", "pairwise"})
    void refusesDriftInAnUnselectedActualProviderModel(String objective) throws IOException {
        JsonNode fixture = fixture(objective);
        JsonNode response = fixture.get("response");
        assertThat(response.get("action").stringValue()).isEqualTo("medium");
        RestClient.Builder builder = RestClient.builder().baseUrl("http://synthetic-router");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        FastApiRouterClient client = client(builder.build(), response,
                fixture.get("canonicalPolicySha256").stringValue(), "different-high-model");
        server.expect(once(), requestTo("http://synthetic-router/router/classify/v2"))
                .andRespond(withSuccess(response.toString(), MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class).hasMessageContaining("candidateModelIds");
        server.verify();
    }

    @Test
    void legacyRollbackResponseStillWorksWithV1Client() throws IOException {
        JsonNode root = fixtures();
        assertThat(root.get("rollbackPointerManifestCorrect").booleanValue()).isTrue();
        JsonNode response = root.get("legacyResponse");
        RestClient.Builder builder = RestClient.builder().baseUrl("http://synthetic-router");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        FastApiRouterClient client = new FastApiRouterClient(builder.build(), "synthetic-secret", 1,
                "embed-default", "router-default");
        server.expect(once(), requestTo("http://synthetic-router/router/classify"))
                .andRespond(withSuccess(response.toString(), MediaType.APPLICATION_JSON));
        RouterClassification actual = client.classify("q");
        assertThat(actual.tier()).isEqualTo(RouterService.Tier.MEDIUM);
        assertThat(actual.confidence()).isEqualTo(response.get("confidence").doubleValue());
        assertThat(actual.modelVersion()).isEqualTo("router-synthetic-legacy");
        assertThat(actual.decisionMetadata()).isNull();
        server.verify();
    }
}
