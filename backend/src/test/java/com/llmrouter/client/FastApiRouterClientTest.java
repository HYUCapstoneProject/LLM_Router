package com.llmrouter.client;

import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.service.RouterService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.MDC;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.convert.ApplicationConversionService;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.util.Collections;

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
