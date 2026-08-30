package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClassification;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterEmbedding;
import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.ChatRequest;
import com.llmrouter.dto.ChatResponse;
import com.llmrouter.dto.LlmResult;
import com.llmrouter.dto.RequestLogEntry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private final RouterClient routerClient;
    private final RouterService routerService;
    private final CacheService cacheService;
    private final LlmService llmService;
    private final RequestLogService requestLogService;

    public ChatResponse chat(ChatRequest request) {
        long start = System.currentTimeMillis();
        String requestId = MDC.get(RequestIdFilter.REQUEST_ID);
        String message = request.getMessage();

        // 임베딩·캐시 조회 단계에서 실패하면 아직 분류 전이므로 null로 기록한다.
        RouterService.Tier tier = null;
        RouterEmbedding embeddingResult = null;
        String fallbackReason = null;

        try {
            // 1. exact match 를 먼저 본다. 글자까지 같은 질문이면 임베딩 호출 자체가 필요 없다.
            //    FastAPI 가 us-west1, 게이트웨이가 asia-northeast3 라 embed 왕복만 약 120ms 다.
            //    오탐이 원리적으로 0 이라 임계값도 보지 않는다.
            Optional<CachedResponse> cached = cacheService.getExact(message);
            String matchType = cached.isPresent() ? RequestLogEntry.MATCH_EXACT : null;

            // 2. exact 미스일 때만 임베딩해서 시맨틱 캐시를 본다.
            if (cached.isEmpty()) {
                try {
                    embeddingResult = routerClient.embed(message);
                    cached = cacheService.get(embeddingResult.vector());
                    if (cached.isPresent()) {
                        matchType = RequestLogEntry.MATCH_SEMANTIC;
                    }
                } catch (RouterClientException error) {
                    // 임베딩 서버가 죽어도 Claude 응답까지 막지는 않는다. 캐시만 우회한다.
                    fallbackReason = "embed_unavailable";
                    log.warn("임베딩 실패로 캐시를 우회한다 requestId={} status={}",
                            requestId, error.statusCode(), error);
                }
            }

            // 3. 캐시 히트면 분류와 Claude 호출을 생략한다.
            if (cached.isPresent()) {
                CachedResponse hit = cached.get();
                ChatResponse response = ChatResponse.builder()
                        .requestId(requestId)
                        .response(hit.text())
                        .model(RouterService.toModelName(hit.tier()))
                        .tier(hit.tier())
                        .modelId(hit.modelId())
                        .cacheHit(true)
                        .inputTokens(0)
                        .outputTokens(0)
                        .latency(System.currentTimeMillis() - start)
                        .build();

                requestLogService.record(RequestLogEntry.builder()
                        .requestId(requestId)
                        .question(message)
                        .conversationId(request.getConversationId())
                        .tier(hit.tier())
                        .modelId(hit.modelId())
                        .routerModelVersion(null)
                        .routerConfidence(null)
                        // exact 히트는 임베딩을 부르지 않았으므로 버전이 없다.
                        .embedModelVersion(embeddingResult == null ? null : embeddingResult.modelVersion())
                        .cacheHit(true)
                        .cacheMatchType(matchType)
                        .similarity(hit.similarity())
                        .cacheEntryId(hit.cacheEntryId())
                        .inputTokens(0)
                        .outputTokens(0)
                        .latencyMs(response.getLatency())
                        .origin(RequestLogEntry.ORIGIN_CHAT)
                        .build());
                return response;
            }

            // 4. 캐시 미스일 때만 분류하고 Claude를 호출한다.
            RouterClassification classification = null;
            try {
                classification = routerService.classify(message);
                tier = classification.tier();
            } catch (RouterClientException error) {
                // 라우터 장애 시 가장 비싸지도, 성능이 낮지도 않은 MEDIUM을 안전 기본값으로 쓴다.
                tier = RouterService.Tier.MEDIUM;
                fallbackReason = fallbackReason == null
                        ? "classify_unavailable"
                        : "embed_and_classify_unavailable";
                log.warn("라우팅 실패로 MEDIUM 폴백한다 requestId={} status={}",
                        requestId, error.statusCode(), error);
            }
            LlmResult result = llmService.call(tier, message);

            // 5. 생성한 캐시 행을 요청 로그와 연결해 이후 에스컬레이션에서 갱신한다.
            Long cacheEntryId = embeddingResult == null
                    ? null
                    : cacheService.put(message, embeddingResult.vector(), result,
                            embeddingResult.modelVersion());

            ChatResponse response = ChatResponse.builder()
                    .requestId(requestId)
                    .response(result.text())
                    .model(RouterService.toModelName(result.tier()))
                    .tier(result.tier())
                    .modelId(result.modelId())
                    .cacheHit(false)
                    .inputTokens(result.inputTokens())
                    .outputTokens(result.outputTokens())
                    .latency(System.currentTimeMillis() - start)
                    .build();

            requestLogService.record(RequestLogEntry.builder()
                    .requestId(requestId)
                    .question(message)
                    .conversationId(request.getConversationId())
                    .tier(result.tier())
                    .modelId(result.modelId())
                    .routerModelVersion(classification == null ? null : classification.modelVersion())
                    .routerConfidence(classification == null ? null : classification.confidence())
                    .embedModelVersion(embeddingResult == null ? null : embeddingResult.modelVersion())
                    .cacheHit(false)
                    .cacheEntryId(cacheEntryId)
                    .inputTokens(result.inputTokens())
                    .outputTokens(result.outputTokens())
                    .latencyMs(response.getLatency())
                    .fallbackReason(fallbackReason)
                    .origin(RequestLogEntry.ORIGIN_CHAT)
                    .build());
            return response;
        } catch (Exception error) {
            requestLogService.recordFailure(requestId, message, tier,
                    System.currentTimeMillis() - start, error);
            throw error;
        }
    }
}
