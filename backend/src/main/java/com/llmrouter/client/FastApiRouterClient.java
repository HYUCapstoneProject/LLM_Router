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
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/** FastAPI의 /router/embed, /router/classify 계약을 구현한 HTTP 클라이언트. */
@Slf4j
@Component
@ConditionalOnProperty(name = "llm.router.mode", havingValue = "fastapi")
public class FastApiRouterClient implements RouterClient {

    static final int EMBEDDING_DIMENSION = 1024;
    private static final Set<String> ACTIONS = Set.of("low", "medium", "high");
    private static final Map<String, String> ACTION_TIERS = Map.of(
            "low", "easy", "medium", "medium", "high", "hard");
    private static final Map<String, String> ACTION_MODELS = Map.of(
            "low", "haiku", "medium", "sonnet", "high", "opus");
    private static final Map<String, String> SCORE_SEMANTICS = Map.of(
            "ordinal", "difficulty_class_probabilities",
            "pointwise", "independent_success_probabilities",
            "pairwise", "relative_quality_scores");

    private final RestClient restClient;
    private final String token;
    private final int maxAttempts;
    private final String classifyContract;
    private final String expectedObjective;
    private final String expectedPolicySha256;
    private final String candidatePoolVersion;
    private final Map<String, String> candidateModelIds;
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
            String routerModelVersion,
            @Value("${llm.router.classify-contract:v1}") String classifyContract,
            @Value("${llm.router.expected-policy-sha256:}") String expectedPolicySha256,
            @Value("${llm.router.candidate-pool-version:}") String candidatePoolVersion,
            @Value("${llm.models.easy:}") String easyModel,
            @Value("${llm.models.medium:}") String mediumModel,
            @Value("${llm.models.hard:}") String hardModel,
            @Value("${llm.router.expected-objective:pointwise}") String expectedObjective
    ) {
        this(createRestClient(RestClient.builder(), baseUrl, connectTimeout, readTimeout), token, maxAttempts,
                embedModelVersion, routerModelVersion, classifyContract, expectedPolicySha256,
                candidatePoolVersion, easyModel, mediumModel, hardModel, expectedObjective);
    }

    FastApiRouterClient(RestClient restClient, String token, int maxAttempts,
                        String embedModelVersion, String routerModelVersion) {
        this(restClient, token, maxAttempts, embedModelVersion, routerModelVersion,
                "v1", "", "", "", "", "");
    }

    FastApiRouterClient(RestClient restClient, String token, int maxAttempts,
                        String embedModelVersion, String routerModelVersion, String classifyContract,
                        String expectedPolicySha256, String candidatePoolVersion,
                        String easyModel, String mediumModel, String hardModel) {
        this(restClient, token, maxAttempts, embedModelVersion, routerModelVersion, classifyContract,
                expectedPolicySha256, candidatePoolVersion, easyModel, mediumModel, hardModel, "pointwise");
    }

    FastApiRouterClient(RestClient restClient, String token, int maxAttempts,
                        String embedModelVersion, String routerModelVersion, String classifyContract,
                        String expectedPolicySha256, String candidatePoolVersion,
                        String easyModel, String mediumModel, String hardModel, String expectedObjective) {
        if (token == null || token.isBlank()) {
            throw new IllegalArgumentException("fastapi 모드에서는 llm.router.token이 필요합니다");
        }
        if (maxAttempts < 1) {
            throw new IllegalArgumentException("llm.router.max-attempts는 1 이상이어야 합니다");
        }
        if (!"v1".equals(classifyContract) && !"v2".equals(classifyContract)) {
            throw new IllegalArgumentException("llm.router.classify-contract는 v1 또는 v2여야 합니다");
        }
        if ("v2".equals(classifyContract)) {
            if (expectedObjective == null || !SCORE_SEMANTICS.containsKey(expectedObjective)) {
                throw new IllegalArgumentException("v2 expected-objective는 ordinal, pointwise, pairwise 중 하나여야 합니다");
            }
            if (expectedPolicySha256 == null || !expectedPolicySha256.matches("[0-9a-f]{64}")) {
                throw new IllegalArgumentException(
                        "v2 모드에서는 llm.router.expected-policy-sha256에 소문자 64자리 SHA256이 필요합니다");
            }
            if (isBlank(candidatePoolVersion)) {
                throw new IllegalArgumentException("v2 모드에서는 llm.router.candidate-pool-version이 필요합니다");
            }
            if (isBlank(easyModel) || isBlank(mediumModel) || isBlank(hardModel)) {
                throw new IllegalArgumentException("v2 모드에서는 llm.models.easy/medium/hard가 모두 필요합니다");
            }
        }
        this.restClient = restClient;
        this.token = token;
        this.maxAttempts = maxAttempts;
        this.classifyContract = classifyContract;
        this.expectedObjective = expectedObjective;
        this.expectedPolicySha256 = expectedPolicySha256;
        this.candidatePoolVersion = candidatePoolVersion;
        this.candidateModelIds = "v2".equals(classifyContract)
                ? Map.of("low", easyModel, "medium", mediumModel, "high", hardModel) : Map.of();
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
        if ("v2".equals(classifyContract)) {
            return classifyV2(message);
        }
        ClassifyResponse response = invoke("classify", () -> restClient.post()
                .uri("/router/classify")
                .headers(headers -> addHeaders(headers, requestId()))
                .body(new MessageRequest(message))
                .retrieve()
                .body(ClassifyResponse.class));

        if (response == null || response.tier() == null
                || response.modelVersion() == null || response.modelVersion().isBlank()
                || response.confidence() == null || !Double.isFinite(response.confidence())
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

    private RouterClassification classifyV2(String message) {
        DecisionResponse response = invoke("classify/v2", () -> restClient.post()
                .uri("/router/classify/v2")
                .headers(headers -> addHeaders(headers, requestId()))
                .body(new MessageRequest(message))
                .retrieve()
                .body(DecisionResponse.class));
        if (response == null || !"router-decision-v2".equals(response.schemaVersion())) {
            throw decisionMismatch("schemaVersion");
        }
        if (isBlank(response.modelVersion()) || isBlank(response.policyVersion())) {
            throw decisionMismatch("modelVersion/policyVersion");
        }
        if (!expectedPolicySha256.equals(response.policySha256())) {
            throw decisionMismatch("policySha256");
        }
        if (!candidatePoolVersion.equals(response.candidatePoolVersion())) {
            throw decisionMismatch("candidatePoolVersion");
        }
        if (!candidateModelIds.equals(response.candidateModelIds())) {
            throw decisionMismatch("candidateModelIds (llm.models)");
        }
        String action = response.action();
        if (action == null || !ACTIONS.contains(action)
                || !ACTION_TIERS.get(action).equals(response.tier())
                || !ACTION_MODELS.get(action).equals(response.model())) {
            throw decisionMismatch("action/tier/model");
        }
        String objective = response.objective();
        if (objective == null || !SCORE_SEMANTICS.containsKey(objective)
                || !SCORE_SEMANTICS.get(objective).equals(response.scoreSemantics())) {
            throw decisionMismatch("objective/scoreSemantics");
        }
        Map<String, Double> scores = response.actionScores();
        if (scores == null || !ACTIONS.equals(scores.keySet())
                || scores.values().stream().anyMatch(score -> score == null || !Double.isFinite(score))) {
            throw decisionMismatch("actionScores");
        }
        if ("pairwise".equals(objective)) {
            if (response.confidence() != null) {
                throw decisionMismatch("pairwise confidence must be null");
            }
        } else {
            if (scores.values().stream().anyMatch(score -> score < 0.0 || score > 1.0)) {
                throw decisionMismatch("probability actionScores");
            }
            if ("ordinal".equals(objective)
                    && Math.abs(scores.values().stream().mapToDouble(Double::doubleValue).sum() - 1.0) > 1e-6) {
                throw decisionMismatch("ordinal probability sum");
            }
            if (response.confidence() == null || !Double.isFinite(response.confidence())
                    || response.confidence() < 0.0 || response.confidence() > 1.0
                    || Math.abs(response.confidence() - scores.get(action)) > 1e-9) {
                throw decisionMismatch("confidence must equal selected action score");
            }
        }
        if (!expectedObjective.equals(objective)) {
            throw decisionMismatch("expected objective");
        }
        RouterService.Tier tier = RouterService.Tier.valueOf(response.tier().toUpperCase(Locale.ROOT));
        currentRouterModelVersion.set(response.modelVersion());
        return new RouterClassification(tier, response.confidence(), response.modelVersion(),
                new RouterClassification.DecisionMetadata(objective, response.policyVersion(),
                        response.policySha256(), response.candidatePoolVersion(), response.candidateModelIds(),
                        action, response.scoreSemantics(), scores));
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static RouterClientException decisionMismatch(String field) {
        return new RouterClientException("FastAPI classify/v2 응답 계약 불일치: " + field, null);
    }

    @Override
    public RouterRerank rerank(String query, java.util.List<String> candidates) {
        if (candidates == null || candidates.isEmpty()) {
            throw new RouterClientException("rerank candidates 가 비어 있습니다", null);
        }

        RerankResponse response = invoke("rerank", () -> restClient.post()
                .uri("/router/rerank")
                .headers(headers -> addHeaders(headers, requestId()))
                .body(new RerankRequest(query, candidates))
                .retrieve()
                .body(RerankResponse.class));

        // 길이가 어긋나면 점수를 엉뚱한 후보에 붙이게 된다. 조용히 틀리느니 실패시킨다.
        if (response == null || response.scores() == null
                || response.scores().length != candidates.size()
                || response.modelVersion() == null || response.modelVersion().isBlank()) {
            throw new RouterClientException("FastAPI rerank 응답이 계약과 다릅니다", null);
        }
        for (double score : response.scores()) {
            // NaN 은 어떤 임계값과 비교해도 false 라 조용한 캐시 미스가 된다.
            if (!Double.isFinite(score)) {
                throw new RouterClientException("FastAPI rerank 가 유한하지 않은 점수를 반환했습니다", null);
            }
        }
        return new RouterRerank(response.scores(), response.modelVersion());
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

    private record ClassifyResponse(String tier, String model, Double confidence, String modelVersion) {
    }

    private record DecisionResponse(String schemaVersion, String tier, String model, Double confidence,
                                    String modelVersion, String objective, String policyVersion,
                                    String policySha256, String candidatePoolVersion,
                                    Map<String, String> candidateModelIds, String action,
                                    String scoreSemantics, Map<String, Double> actionScores) {
    }

    private record RerankRequest(String query, java.util.List<String> candidates) {
    }

    private record RerankResponse(double[] scores, String modelVersion) {
    }
}
