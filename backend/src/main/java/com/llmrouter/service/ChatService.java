package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
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

        try {
            // 1. 캐시 조회를 위해 질문을 임베딩한다.
            float[] embedding = routerClient.embed(message);

            // 2. 캐시 히트면 분류와 Claude 호출을 생략한다.
            Optional<CachedResponse> cached = cacheService.get(embedding);
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
                        .embedModelVersion(routerClient.embedModelVersion())
                        .cacheHit(true)
                        .similarity(hit.similarity())
                        .cacheEntryId(hit.cacheEntryId())
                        .inputTokens(0)
                        .outputTokens(0)
                        .latencyMs(response.getLatency())
                        .origin(RequestLogEntry.ORIGIN_CHAT)
                        .build());
                return response;
            }

            // 3. 캐시 미스일 때만 분류하고 Claude를 호출한다.
            tier = routerService.classify(message);
            LlmResult result = llmService.call(tier, message);

            // 4. 생성한 캐시 행을 요청 로그와 연결해 이후 에스컬레이션에서 갱신한다.
            long cacheEntryId = cacheService.put(message, embedding, result);

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
                    .routerModelVersion(routerClient.routerModelVersion())
                    .embedModelVersion(routerClient.embedModelVersion())
                    .cacheHit(false)
                    .cacheEntryId(cacheEntryId)
                    .inputTokens(result.inputTokens())
                    .outputTokens(result.outputTokens())
                    .latencyMs(response.getLatency())
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
