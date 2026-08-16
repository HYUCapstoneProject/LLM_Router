package com.llmrouter.client;

import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.service.RouterService;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/** FastAPI의 /router/embed, /router/classify 계약을 구현한 HTTP 클라이언트. */
@Slf4j
@Component
@ConditionalOnProperty(name = "llm.router.mode", havingValue = "fastapi")
public class FastApiRouterClient implements RouterClient {

    static final int EMBEDDING_DIMENSION = 1024;

    private final RestClient restClient;
    private final String token;
    private final int maxAttempts;
    private final AtomicReference<String> currentEmbedModelVersion;
    private final AtomicReference<String> currentRouterModelVersion;

    @Autowired
    public FastApiRouterClient(
            @Value("${llm.router.base-url:http://10.138.0.2:8000}") String baseUrl,
            @Value("${llm.router.token:}") String token,
            @Value("${llm.router.connect-timeout:PT2S}") Duration connectTimeout,
            @Value("${llm.router.read-timeout:PT15S}") Duration readTimeout,
            @Value("${llm.router.max-attempts:2}") int maxAttempts,
            @Value("${llm.router.embed-model-version:bge-large-en-v1.5}") String embedModelVersion,
            @Value("${llm.router.model-version:router-modernbert-base-routerbench-20260814-45dcc6d}")
            String routerModelVersion
    ) {
        this(createRestClient(RestClient.builder(), baseUrl, connectTimeout, readTimeout), token, maxAttempts,
                embedModelVersion, routerModelVersion);
    }

    FastApiRouterClient(RestClient restClient, String token, int maxAttempts,
                        String embedModelVersion, String routerModelVersion) {
        if (token == null || token.isBlank()) {
            throw new IllegalArgumentException("fastapi 모드에서는 llm.router.token이 필요합니다");
        }
        if (maxAttempts < 1) {
            throw new IllegalArgumentException("llm.router.max-attempts는 1 이상이어야 합니다");
        }
        this.restClient = restClient;
        this.token = token;
        this.maxAttempts = maxAttempts;
        this.currentEmbedModelVersion = new AtomicReference<>(embedModelVersion);
        this.currentRouterModelVersion = new AtomicReference<>(routerModelVersion);
    }

    private static RestClient createRestClient(RestClient.Builder builder, String baseUrl,
                                               Duration connectTimeout, Duration readTimeout) {
        // 하나의 HttpClient를 애플리케이션 생명주기 동안 재사용해 keep-alive와 커넥션 풀을 공유한다.
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(connectTimeout)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(readTimeout);
        return builder.baseUrl(baseUrl).requestFactory(requestFactory).build();
    }

    @Override
    public RouterEmbedding embed(String message) {
        EmbedResponse response = invoke("embed", () -> restClient.post()
                .uri("/router/embed")
                .headers(headers -> addHeaders(headers, requestId()))
                .body(new MessageRequest(message))
                .retrieve()
                .body(EmbedResponse.class));

        if (response == null || response.embedding() == null
                || response.dim() != EMBEDDING_DIMENSION
                || response.embedding().length != EMBEDDING_DIMENSION
                || response.modelVersion() == null || response.modelVersion().isBlank()
                || containsNonFinite(response.embedding())) {
            throw new RouterClientException("FastAPI embed 응답이 계약과 다릅니다", null);
        }

        currentEmbedModelVersion.set(response.modelVersion());
        return new RouterEmbedding(response.embedding(), response.modelVersion());
    }

    @Override
    public RouterClassification classify(String message) {
        ClassifyResponse response = invoke("classify", () -> restClient.post()
                .uri("/router/classify")
                .headers(headers -> addHeaders(headers, requestId()))
                .body(new MessageRequest(message))
                .retrieve()
                .body(ClassifyResponse.class));

        if (response == null || response.tier() == null
                || response.modelVersion() == null || response.modelVersion().isBlank()
                || !Double.isFinite(response.confidence())
                || response.confidence() < 0.0 || response.confidence() > 1.0) {
            throw new RouterClientException("FastAPI classify 응답이 계약과 다릅니다", null);
        }

        RouterService.Tier tier;
        try {
            tier = RouterService.Tier.valueOf(response.tier().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException error) {
            throw new RouterClientException("FastAPI가 알 수 없는 tier를 반환했습니다: " + response.tier(),
                    null, error);
        }

        currentRouterModelVersion.set(response.modelVersion());
        return new RouterClassification(tier, response.confidence(), response.modelVersion());
    }

    private void addHeaders(org.springframework.http.HttpHeaders headers, String requestId) {
        headers.setBearerAuth(token);
        headers.set("X-Request-ID", requestId);
    }

    private String requestId() {
        String requestId = MDC.get(RequestIdFilter.REQUEST_ID);
        return requestId == null || requestId.isBlank() ? UUID.randomUUID().toString() : requestId;
    }

    private boolean containsNonFinite(float[] values) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                return true;
            }
        }
        return false;
    }

    private <T> T invoke(String operation, Supplier<T> call) {
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.get();
            } catch (RestClientResponseException error) {
                int status = error.getStatusCode().value();
                if (attempt < maxAttempts && (status == 500 || status == 503)) {
                    log.warn("FastAPI {} 실패(status={}), 재시도 {}/{}", operation, status,
                            attempt + 1, maxAttempts);
                    continue;
                }
                throw new RouterClientException(
                        "FastAPI " + operation + " 실패(status=" + status + ")", status, error);
            } catch (ResourceAccessException error) {
                if (attempt < maxAttempts) {
                    log.warn("FastAPI {} 연결 실패, 재시도 {}/{}", operation, attempt + 1, maxAttempts);
                    continue;
                }
                throw new RouterClientException("FastAPI " + operation + " 연결 실패", null, error);
            } catch (RestClientException error) {
                // JSON 파싱 실패 등도 외부 계약 장애다. ChatService가 폴백할 수 있게 타입을 통일한다.
                throw new RouterClientException("FastAPI " + operation + " 응답 처리 실패", null, error);
            }
        }
        throw new IllegalStateException("도달할 수 없는 FastAPI 재시도 상태");
    }

    @Override
    public String embedModelVersion() {
        return currentEmbedModelVersion.get();
    }

    @Override
    public String routerModelVersion() {
        return currentRouterModelVersion.get();
    }

    private record MessageRequest(String message) {
    }

    private record EmbedResponse(float[] embedding, int dim, String modelVersion) {
    }

    private record ClassifyResponse(String tier, String model, double confidence, String modelVersion) {
    }
}
