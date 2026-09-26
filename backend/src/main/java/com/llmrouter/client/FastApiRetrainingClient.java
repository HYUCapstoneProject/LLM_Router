package com.llmrouter.client;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RetrainingStatus;
import com.llmrouter.dto.RouterModelStatus;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.UUID;
import java.util.Set;
import java.util.function.Supplier;

/**
 * 재학습 계약의 HTTP 구현 (04 API Contract).
 *
 * <p>추론용 {@link FastApiRouterClient} 와 별도의 {@link RestClient} 를 쓴다.
 * 제출은 GCS artifact 생성과 Vertex 제출까지 기다리므로 추론(15초)보다 긴 타임아웃이
 * 필요하고, 그 타임아웃을 추론 경로에 같이 걸면 FastAPI 가 느려질 때 사용자 요청이
 * 그만큼 붙잡힌다.
 *
 * <p><b>재시도는 이 클래스에서 하지 않는다.</b> 계약의 재시도 정책은 5초·15초·30초
 * 이후 60초 상한인데, 그 시간 동안 스레드를 붙잡고 있으면 배치 제출이 요청 스레드를
 * 점유한다. 재시도는 호출자(스케줄러)가 다음 주기에 같은 batchId 로 다시 부르는 방식으로
 * 처리하고, 여기서는 오류를 코드와 함께 올려보내기만 한다.
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "llm.router.mode", havingValue = "fastapi")
public class FastApiRetrainingClient implements RetrainingClient {

    private static final Set<String> STATUSES = Set.of("queued", "running", "cancelling", "paused",
            "succeeded", "rejected", "failed", "cancelled", "unknown");
    private static final Set<String> STAGES = Set.of("initializing", "feedbackPreparation",
            "feedbackMerge", "training", "evaluation", "packaging", "gate", "publishing", "complete");
    private static final Set<String> ACTIVATIONS = Set.of("pending", "active", "loadFailed",
            "superseded", "notApplicable");

    private final RestClient restClient;
    private final String token;

    @Autowired
    public FastApiRetrainingClient(
            @Value("${llm.router.base-url:http://10.138.0.2:8000}") String baseUrl,
            @Value("${llm.router.token:}") String token,
            @Value("${llm.router.connect-timeout:PT2S}") Duration connectTimeout,
            @Value("${llm.retraining.read-timeout:PT60S}") Duration readTimeout
    ) {
        this(createRestClient(baseUrl, connectTimeout, readTimeout), token);
    }

    FastApiRetrainingClient(RestClient restClient, String token) {
        if (token == null || token.isBlank()) {
            throw new IllegalArgumentException("fastapi 모드에서는 llm.router.token이 필요합니다");
        }
        this.restClient = restClient;
        this.token = token;
    }

    private static RestClient createRestClient(String baseUrl, Duration connectTimeout,
                                               Duration readTimeout) {
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(connectTimeout)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(readTimeout);
        return RestClient.builder().baseUrl(baseUrl).requestFactory(requestFactory).build();
    }

    @Override
    public EscalateAcceptance submit(FeedbackBatchRequest request) {
        var response = invoke("escalate", () -> restClient.post()
                .uri("/escalate")
                .headers(this::addHeaders)
                .body(request)
                .retrieve()
                .toEntity(EscalateAcceptance.class));
        EscalateAcceptance acceptance = response.getBody();

        if (response.getStatusCode().value() != 202 || acceptance == null
                || !"escalation-acceptance-v1".equals(acceptance.schemaVersion())
                || !"accepted".equals(acceptance.status())
                || !request.batchId().equals(acceptance.batchId())
                || blank(acceptance.pipelineRunId()) || blank(acceptance.candidateModelVersion())
                || !("/retraining-runs/" + request.batchId()).equals(acceptance.statusUrl())) {
            throw contractMismatch("escalate");
        }
        return acceptance;
    }

    @Override
    public RetrainingStatus status(String batchId) {
        // batchId 를 경로에 직접 끼우지 않고 템플릿 변수로 넘긴다.
        // 영문·숫자·점·밑줄·하이픈만 허용되는 값이지만, 검증이 한 곳에서 빠지는 순간
        // 경로 조작이 되는 자리라 인코딩을 클라이언트에 맡긴다.
        RetrainingStatus status = invoke("retraining-runs", () -> restClient.get()
                .uri("/retraining-runs/{batchId}", batchId)
                .headers(this::addHeaders)
                .retrieve()
                .body(RetrainingStatus.class));

        if (status == null || !"retraining-status-v1".equals(status.schemaVersion())
                || !batchId.equals(status.batchId()) || status.status() == null
                || !STATUSES.contains(status.status()) || status.stage() == null
                || !STAGES.contains(status.stage()) || status.terminal() == null || status.stale() == null
                || status.activation() == null || status.activationStatus() == null
                || !ACTIVATIONS.contains(status.activationStatus())
                || (!Boolean.TRUE.equals(status.stale()) && !"unknown".equals(status.status())
                    && status.terminal() != RetrainingStatus.TERMINAL_STATUSES.contains(status.status()))
                || ("active".equals(status.activationStatus())
                    && (blank(status.activation().desiredModelVersion())
                        || !status.activation().desiredModelVersion()
                            .equals(status.activation().servingModelVersion())))) {
            throw contractMismatch("retraining-runs");
        }
        return status;
    }

    @Override
    public RouterModelStatus currentModel() {
        RouterModelStatus status = invoke("router/model/current", () -> restClient.get()
                .uri("/router/model/current")
                .headers(this::addHeaders)
                .retrieve()
                .body(RouterModelStatus.class));

        if (status == null || status.status() == null
                || !Set.of("inSync", "loading", "loadFailed", "static", "unavailable").contains(status.status())
                || ("inSync".equals(status.status()) && !status.isInSync())) {
            throw contractMismatch("router/model/current");
        }
        return status;
    }

    /**
     * 상대 {@code statusUrl} 을 base URL 에 안전하게 resolve 한다.
     *
     * <p>계약이 "문자열 단순 결합 대신 URI builder 로 resolve" 하라고 못 박은 자리다.
     * 현재 구현은 경로가 {@code /retraining-runs/{batchId}} 로 고정이라 batchId 만
     * 다시 뽑아 쓰지만, FastAPI 가 경로를 바꾸면 이 메서드만 고치면 된다.
     */
    static String resolveStatusPath(String statusUrl) {
        return UriComponentsBuilder.fromUriString(statusUrl).build().getPath();
    }

    private static boolean blank(String value) {
        return value == null || value.isBlank();
    }

    private static RetrainingClientException contractMismatch(String operation) {
        // A malformed response does not prove that the remote submission was rejected.
        return new RetrainingClientException("FastAPI " + operation + " 응답이 계약과 다릅니다",
                null, "RESPONSE_CONTRACT_MISMATCH");
    }

    private void addHeaders(HttpHeaders headers) {
        headers.setBearerAuth(token);
        headers.set("X-Request-ID", requestId());
    }

    private String requestId() {
        String requestId = MDC.get(RequestIdFilter.REQUEST_ID);
        return requestId == null || requestId.isBlank() ? UUID.randomUUID().toString() : requestId;
    }

    private <T> T invoke(String operation, Supplier<T> call) {
        try {
            return call.get();
        } catch (RestClientResponseException error) {
            int status = error.getStatusCode().value();
            String code = extractCode(error);
            // 토큰과 질문 원문은 로그에 남기지 않는다. 코드와 상태만 남긴다.
            log.warn("FastAPI {} 실패 status={} code={}", operation, status, code);
            throw new RetrainingClientException(
                    "FastAPI " + operation + " 실패(status=" + status + ", code=" + code + ")",
                    status, code, error);
        } catch (ResourceAccessException error) {
            // 연결 실패·타임아웃. 계약상 client timeout 은 같은 batchId 로 재시도 대상이다.
            // 제출이 이미 접수됐을 수도 있으므로 새 batchId 를 만들면 안 된다.
            throw new RetrainingClientException(
                    "FastAPI " + operation + " 연결 실패", null, null, error);
        } catch (RestClientException error) {
            throw new RetrainingClientException(
                    "FastAPI " + operation + " 응답 처리 실패", null, null, error);
        }
    }

    /** 계약의 오류 본문에서 안정된 {@code code} 를 꺼낸다. 못 읽으면 null. */
    private String extractCode(RestClientResponseException error) {
        try {
            ErrorBody body = error.getResponseBodyAs(ErrorBody.class);
            return body == null ? null : body.code();
        } catch (RuntimeException ignored) {
            // 오류 본문이 계약 형태가 아닐 수도 있다. 그때는 상태 코드로만 분기한다.
            return null;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private record ErrorBody(String error, String code, String requestId) {
    }
}
