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
        // 성공/실패 응답이 같은 id 를 쓰도록 필터가 만든 값을 재사용한다
        String requestId = MDC.get(RequestIdFilter.REQUEST_ID);
        String message = request.getMessage();

        // 1. 임베딩 — cache hit 여부와 무관하게 항상 필요하다.
        //    cache 조회 자체가 벡터를 선행 요구하기 때문이다.
        float[] embedding = routerClient.embed(message);

        // 2. 시맨틱 캐시 조회
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
                    .inputTokens(0)   // LLM 호출 없음 = 비용 0
                    .outputTokens(0)
                    .latency(System.currentTimeMillis() - start)
                    .build();

            requestLogService.record(RequestLogEntry.builder()
                    .requestId(requestId)
                    .question(message)
                    .conversationId(request.getConversationId())
                    .tier(hit.tier())
                    .modelId(hit.modelId())
                    .routerModelVersion(null)   // 캐시 히트면 classify 를 안 부른다
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

        // 3. 난이도 분류 (miss 일 때만)
        RouterService.Tier tier = routerService.classify(message);

        // 4. 티어에 맞는 Claude 호출
        LlmResult result = llmService.call(tier, message);

        // 5. 캐시 적재.
        //    반환된 id 를 request_log 에 남겨야 /api/escalate 가 나중에
        //    "불만족한 답이 어느 캐시 행에서 나왔는지" 를 알고 그 행을 갱신할 수 있다.
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
    }
}
