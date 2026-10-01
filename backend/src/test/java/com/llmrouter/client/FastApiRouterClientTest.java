package com.llmrouter.client;

import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.service.RouterService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.MDC;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.convert.ApplicationConversionService;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.ExpectedCount.once;
import static org.springframework.test.web.client.ExpectedCount.twice;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.header;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class FastApiRouterClientTest {

    private static final String REQUEST_ID = "00000000-0000-4000-8000-000000000001";
    private static final String POLICY_SHA = "a1".repeat(32);
    private MockRestServiceServer server;
    private FastApiRouterClient client;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://router.internal:8000");
        server = MockRestServiceServer.bindTo(builder).build();
        client = new FastApiRouterClient(builder.build(), "secret-token", 2,
                "embed-default", "router-default");
        MDC.put(RequestIdFilter.REQUEST_ID, REQUEST_ID);
    }

    @AfterEach
    void tearDown() {
        MDC.clear();
    }

    @Test
    void springCreatesFastApiClientWithConfiguredConstructor() {
        new ApplicationContextRunner()
                .withInitializer(context -> context.getBeanFactory()
                        .setConversionService(ApplicationConversionService.getSharedInstance()))
                .withUserConfiguration(FastApiRouterClient.class)
                .withPropertyValues(
                        "llm.router.mode=fastapi",
                        "llm.router.token=secret-token")
                .run(context -> assertThat(context)
                        .hasSingleBean(FastApiRouterClient.class));
    }

    private ApplicationContextRunner v2Context() {
        return new ApplicationContextRunner()
                .withInitializer(context -> context.getBeanFactory()
                        .setConversionService(ApplicationConversionService.getSharedInstance()))
                .withUserConfiguration(FastApiRouterClient.class)
                .withPropertyValues(
                        "llm.router.mode=fastapi", "llm.router.token=secret-token",
                        "llm.router.classify-contract=v2",
                        "llm.router.expected-policy-sha256=" + POLICY_SHA,
                        "llm.router.candidate-pool-version=service-pool-v1",
                        "llm.models.easy=provider-small", "llm.models.medium=provider-medium",
                        "llm.models.hard=provider-large");
    }

    @Test
    void v2StartsOnlyWithCompletePolicyAndActualModelConfiguration() {
        v2Context().run(context -> assertThat(context).hasSingleBean(FastApiRouterClient.class));
    }

    @ParameterizedTest
    @ValueSource(strings = {"llm.router.expected-policy-sha256", "llm.router.candidate-pool-version",
            "llm.models.easy", "llm.models.medium", "llm.models.hard"})
    void v2RefusesMissingConfiguration(String property) {
        v2Context().withPropertyValues(property + "=")
                .run(context -> assertThat(context).hasFailed());
    }

    @ParameterizedTest
    @ValueSource(strings = {"ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789",
            "a1", "sha256:a1", "g"})
    void v2RequiresCanonicalPolicyHash(String hash) {
        v2Context().withPropertyValues("llm.router.expected-policy-sha256=" + hash)
                .run(context -> assertThat(context).hasFailed());
    }

    @Test
    void v2DoesNotTrimAnUncanonicalHash() {
        assertThatThrownBy(() -> new FastApiRouterClient(RestClient.create(), "secret-token", 1,
                "embed-default", "router-default", "v2", " " + POLICY_SHA, "pool",
                "small", "medium", "large"))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("SHA256");
    }

    @Test
    void unknownContractDoesNotSilentlySelectV1() {
        v2Context().withPropertyValues("llm.router.classify-contract=v3")
                .run(context -> assertThat(context).hasFailed());
    }

    @Test
    void embedSendsContractHeadersAndReturnsRequestScopedModelVersion() {
        server.expect(once(), requestTo("http://router.internal:8000/router/embed"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(header("Authorization", "Bearer secret-token"))
                .andExpect(header("X-Request-ID", REQUEST_ID))
                .andExpect(content().json("{\"message\":\"hello\"}"))
                .andRespond(withSuccess(embedResponse(1024, 1024), MediaType.APPLICATION_JSON));

        RouterEmbedding result = client.embed("hello");

        assertThat(result.vector()).hasSize(1024);
        assertThat(result.modelVersion()).isEqualTo("bge-large-en-v1.5");
        assertThat(client.embedModelVersion()).isEqualTo("bge-large-en-v1.5");
        server.verify();
    }

    @Test
    void classifyMapsTierAndIgnoresReferenceModelField() {
        server.expect(once(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withSuccess("""
                        {"tier":"hard","model":"some-reference-model","confidence":0.87,
                         "modelVersion":"router-v2"}
                        """, MediaType.APPLICATION_JSON));

        RouterClassification result = client.classify("hard question");

        assertThat(result.tier()).isEqualTo(RouterService.Tier.HARD);
        assertThat(result.confidence()).isEqualTo(0.87);
        assertThat(result.modelVersion()).isEqualTo("router-v2");
        server.verify();
    }

    @ParameterizedTest
    @ValueSource(strings = {"", ",\"confidence\":null", ",\"confidence\":-0.1", ",\"confidence\":1.1"})
    void v1RejectsMissingNullOrOutOfRangeConfidence(String confidence) {
        server.expect(once(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withSuccess("{\"tier\":\"hard\",\"modelVersion\":\"router-v2\""
                        + confidence + "}", MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class).hasMessageContaining("계약");
        assertThat(client.routerModelVersion()).isEqualTo("router-default");
        server.verify();
    }

    private void useV2Client() {
        useV2Client("pointwise");
    }

    private void useV2Client(String expectedObjective) {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://router.internal:8000");
        server = MockRestServiceServer.bindTo(builder).build();
        client = new FastApiRouterClient(builder.build(), "secret-token", 2,
                "embed-default", "router-default", "v2", POLICY_SHA, "service-pool-v1",
                "provider-small", "provider-medium", "provider-large", expectedObjective);
    }

    private static String decisionResponse(String objective) {
        String semantics = switch (objective) {
            case "ordinal" -> "difficulty_class_probabilities";
            case "pointwise" -> "independent_success_probabilities";
            default -> "relative_quality_scores";
        };
        String scores = switch (objective) {
            case "ordinal" -> "{\"low\":0.2,\"medium\":0.3,\"high\":0.5}";
            case "pointwise" -> "{\"low\":0.2,\"medium\":0.8,\"high\":0.9}";
            default -> "{\"low\":-2.0,\"medium\":0.0,\"high\":4.0}";
        };
        return """
                {"schemaVersion":"router-decision-v2","tier":"easy","model":"haiku",
                 "modelVersion":"router-v2","objective":"%s","policyVersion":"policy-v1",
                 "policySha256":"%s","candidatePoolVersion":"service-pool-v1",
                 "candidateModelIds":{"low":"provider-small","medium":"provider-medium","high":"provider-large"},
                 "action":"low","scoreSemantics":"%s","actionScores":%s,"confidence":%s}
                """.formatted(objective, POLICY_SHA, semantics, scores,
                "pairwise".equals(objective) ? "null" : "0.2");
    }

    @ParameterizedTest
    @ValueSource(strings = {"ordinal", "pointwise", "pairwise"})
    void v2AcceptsFrozenActionAndPreservesObjectiveMetadata(String objective) {
        useV2Client(objective);
        server.expect(once(), requestTo("http://router.internal:8000/router/classify/v2"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(header("Authorization", "Bearer secret-token"))
                .andExpect(header("X-Request-ID", REQUEST_ID))
                .andExpect(content().json("{\"message\":\"q\"}"))
                .andRespond(withSuccess(decisionResponse(objective), MediaType.APPLICATION_JSON));
        RouterClassification result = client.classify("q");
        // The policy may choose low after applying cost, even when high has the highest score.
        assertThat(result.tier()).isEqualTo(RouterService.Tier.EASY);
        assertThat(result.modelVersion()).isEqualTo("router-v2");
        assertThat(client.routerModelVersion()).isEqualTo("router-v2");
        assertThat(result.confidence()).isEqualTo("pairwise".equals(objective) ? null : 0.2);
        assertThat(result.decisionMetadata().objective()).isEqualTo(objective);
        assertThat(result.decisionMetadata().policySha256()).isEqualTo(POLICY_SHA);
        assertThat(result.decisionMetadata().candidateModelIds()).isEqualTo(Map.of(
                "low", "provider-small", "medium", "provider-medium", "high", "provider-large"));
        assertThatThrownBy(() -> result.decisionMetadata().actionScores().put("low", 0.0))
                .isInstanceOf(UnsupportedOperationException.class);
        server.verify();
    }

    @ParameterizedTest
    @CsvSource({"medium,medium,sonnet,0.3", "high,hard,opus,0.5"})
    void v2MapsRemainingActionsToExistingTiers(String action, String tier, String model, double confidence) {
        useV2Client("ordinal");
        String response = decisionResponse("ordinal")
                .replace("\"action\":\"low\"", "\"action\":\"" + action + "\"")
                .replace("\"tier\":\"easy\"", "\"tier\":\"" + tier + "\"")
                .replace("\"model\":\"haiku\"", "\"model\":\"" + model + "\"")
                .replace("\"confidence\":0.2", "\"confidence\":" + confidence);
        server.expect(once(), requestTo("http://router.internal:8000/router/classify/v2"))
                .andRespond(withSuccess(response, MediaType.APPLICATION_JSON));
        RouterClassification result = client.classify("q");
        assertThat(result.tier()).isEqualTo(RouterService.Tier.valueOf(tier.toUpperCase(java.util.Locale.ROOT)));
        assertThat(result.confidence()).isEqualTo(confidence);
        server.verify();
    }

    static Stream<Arguments> invalidDecisions() {
        String ordinal = decisionResponse("ordinal");
        String pairwise = decisionResponse("pairwise");
        return Stream.of(
                Arguments.of(ordinal.replace("router-decision-v2", "router-decision-v1"), "schemaVersion"),
                Arguments.of(ordinal.replace("\"modelVersion\":\"router-v2\"", "\"modelVersion\":null"), "modelVersion"),
                Arguments.of(ordinal.replace("\"policyVersion\":\"policy-v1\"", "\"policyVersion\":\"\""), "policyVersion"),
                Arguments.of(ordinal.replace(POLICY_SHA, "b2".repeat(32)), "policySha256"),
                Arguments.of(ordinal.replace("service-pool-v1", "another-pool"), "candidatePoolVersion"),
                Arguments.of(ordinal.replace("provider-large", "unselected-model-changed"), "candidateModelIds"),
                Arguments.of(ordinal.replace("\"high\":\"provider-large\"", "\"extra\":\"provider-large\""), "candidateModelIds"),
                Arguments.of(ordinal.replace("\"action\":\"low\"", "\"action\":\"LOW\""), "action/tier/model"),
                Arguments.of(ordinal.replace("\"tier\":\"easy\"", "\"tier\":\"hard\""), "action/tier/model"),
                Arguments.of(ordinal.replace("\"model\":\"haiku\"", "\"model\":\"opus\""), "action/tier/model"),
                Arguments.of(ordinal.replace("\"objective\":\"ordinal\"", "\"objective\":\"unknown\""), "objective/scoreSemantics"),
                Arguments.of(ordinal.replace("difficulty_class_probabilities", "relative_quality_scores"), "objective/scoreSemantics"),
                Arguments.of(ordinal.replace("\"low\":0.2", "\"low\":null"), "actionScores"),
                Arguments.of(ordinal.replace("\"low\":0.2", "\"extra\":0.2"), "actionScores"),
                Arguments.of(ordinal.replace("\"low\":0.2", "\"low\":1e400"), "actionScores"),
                Arguments.of(ordinal.replace("\"low\":0.2", "\"low\":-0.2"), "probability actionScores"),
                Arguments.of(ordinal.replace("\"low\":0.2", "\"low\":0.3"), "ordinal probability sum"),
                Arguments.of(ordinal.replace("\"confidence\":0.2", "\"confidence\":0.5"), "confidence"),
                Arguments.of(ordinal.replace("\"confidence\":0.2", "\"confidence\":null"), "confidence"),
                Arguments.of(decisionResponse("pointwise").replace("\"low\":0.2", "\"low\":0.0")
                        .replace("\"confidence\":0.2", "\"confidence\":-0.0000000001"), "confidence"),
                Arguments.of(pairwise.replace("\"confidence\":null", "\"confidence\":0.0"), "pairwise confidence")
        );
    }

    @ParameterizedTest
    @MethodSource("invalidDecisions")
    void v2RejectsContractMismatchWithoutDowngradeOrVersionUpdate(String response, String field) {
        useV2Client();
        server.expect(once(), requestTo("http://router.internal:8000/router/classify/v2"))
                .andRespond(withSuccess(response, MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class).hasMessageContaining(field);
        assertThat(client.routerModelVersion()).isEqualTo("router-default");
        server.verify();
    }

    @ParameterizedTest
    @ValueSource(strings = {"ordinal", "pairwise"})
    void r1PinRejectsAnotherObjectiveEvenWhenPolicyAndPoolMatch(String objective) {
        useV2Client();
        server.expect(once(), requestTo("http://router.internal:8000/router/classify/v2"))
                .andRespond(withSuccess(decisionResponse(objective), MediaType.APPLICATION_JSON));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class).hasMessageContaining("expected objective");
        assertThat(client.routerModelVersion()).isEqualTo("router-default");
        server.verify();
    }

    @Test
    void v2DoesNotDowngradeWhenEndpointIsUnavailable() {
        useV2Client();
        server.expect(once(), requestTo("http://router.internal:8000/router/classify/v2"))
                .andRespond(withStatus(HttpStatus.NOT_FOUND));
        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class)
                .extracting(error -> ((RouterClientException) error).statusCode()).isEqualTo(404);
        server.verify();
    }

    @Test
    void retriesServiceUnavailableOnce() {
        server.expect(twice(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withStatus(HttpStatus.SERVICE_UNAVAILABLE));

        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class)
                .extracting(error -> ((RouterClientException) error).statusCode())
                .isEqualTo(503);
        server.verify();
    }

    @Test
    void tokenLimitRejectionIsNotARouterOutage() {
        server.expect(once(), requestTo("http://router.internal:8000/router/embed"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"error\":\"message has 513 tokens; the limit is 512\","
                                + "\"code\":\"MESSAGE_TOO_LONG\",\"requestId\":\"r\"}"));

        assertThatThrownBy(() -> client.embed("q"))
                .isInstanceOf(MessageTooLongException.class)
                .isNotInstanceOf(RouterClientException.class);
        server.verify();
    }

    @Test
    void otherBadRequestStaysARouterClientException() {
        server.expect(once(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"error\":\"x\",\"code\":\"INVALID_MESSAGE\",\"requestId\":\"r\"}"));

        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class)
                .extracting(error -> ((RouterClientException) error).statusCode())
                .isEqualTo(400);
        server.verify();
    }

    @Test
    void doesNotRetryAuthenticationFailure() {
        server.expect(once(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withStatus(HttpStatus.UNAUTHORIZED));

        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class)
                .extracting(error -> ((RouterClientException) error).statusCode())
                .isEqualTo(401);
        server.verify();
    }

    @Test
    void rejectsEmbeddingWithWrongDimension() {
        server.expect(once(), requestTo("http://router.internal:8000/router/embed"))
                .andRespond(withSuccess(embedResponse(2, 2), MediaType.APPLICATION_JSON));

        assertThatThrownBy(() -> client.embed("q"))
                .isInstanceOf(RouterClientException.class)
                .hasMessageContaining("계약");
        server.verify();
    }

    @Test
    void wrapsMalformedJsonSoChatCanFallback() {
        server.expect(once(), requestTo("http://router.internal:8000/router/classify"))
                .andRespond(withSuccess("not-json", MediaType.APPLICATION_JSON));

        assertThatThrownBy(() -> client.classify("q"))
                .isInstanceOf(RouterClientException.class)
                .hasMessageContaining("응답 처리 실패");
        server.verify();
    }

    private String embedResponse(int vectorSize, int dim) {
        String values = String.join(",", Collections.nCopies(vectorSize, "0.03125"));
        return "{\"embedding\":[" + values + "],\"dim\":" + dim
                + ",\"modelVersion\":\"bge-large-en-v1.5\"}";
    }
}
