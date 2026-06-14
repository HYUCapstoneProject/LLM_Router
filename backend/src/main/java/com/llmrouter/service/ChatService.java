package com.llmrouter.service;

import com.llmrouter.dto.ChatRequest;
import com.llmrouter.dto.ChatResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ChatService {

    private final RouterService routerService;
    private final CacheService cacheService;
    private final LlmService llmService;

    public ChatResponse chat(ChatRequest request) {
        long start = System.currentTimeMillis();
        String message = request.getMessage();

        // 1. 시맨틱 캐시 조회
        var cached = cacheService.get(message);
        if (cached.isPresent()) {
            return ChatResponse.builder()
                    .response(cached.get())
                    .model("haiku")   // 캐시 히트는 항상 최저비용으로 표시
                    .cacheHit(true)
                    .latency(System.currentTimeMillis() - start)
                    .build();
        }

        // 2. 라우팅 분류
        RouterService.ModelTier tier = routerService.classify(message);
        String model = routerService.resolveModelName(tier);

        // 3. LLM 호출
        String response = llmService.call(model, message);

        // 4. 캐시 저장
        cacheService.put(message, response);

        return ChatResponse.builder()
                .response(response)
                .model(model)
                .cacheHit(false)
                .latency(System.currentTimeMillis() - start)
                .build();
    }
}
