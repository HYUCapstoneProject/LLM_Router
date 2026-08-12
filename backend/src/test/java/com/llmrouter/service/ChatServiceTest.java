package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.ChatRequest;
import com.llmrouter.dto.ChatResponse;
import com.llmrouter.dto.LlmResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.MDC;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ChatServiceTest {
    @Mock RouterClient routerClient;
    @Mock CacheService cacheService;
    @Mock RouterService routerService;
    @Mock LlmService llmService;
    @Mock RequestLogService requestLogService;
    ChatService service;

    @BeforeEach
    void setUp() {
        service = new ChatService(routerClient, routerService, cacheService, llmService, requestLogService);
        MDC.put("requestId", "00000000-0000-0000-0000-000000000001");
    }

    @Test
    void cacheHitSkipsClassifierAndLlmAndRecordsSimilarity() {
        float[] embedding = {0.1f, 0.2f};
        when(routerClient.embed("hello")).thenReturn(embedding);
        when(cacheService.get(embedding)).thenReturn(Optional.of(
                new CachedResponse(7L, "cached", "easy", "model-e", 0.97)));
        when(routerClient.embedModelVersion()).thenReturn("embed-v1");

        ChatResponse response = service.chat(request("hello"));

        assertThat(response.getResponse()).isEqualTo("cached");
        assertThat(response.isCacheHit()).isTrue();
        assertThat(response.getInputTokens()).isZero();
        verifyNoInteractions(routerService, llmService);
        verify(cacheService, never()).put(anyString(), any(), any());
        verify(requestLogService).record(response, "hello", null, "embed-v1", 0.97, null);
    }

    @Test
    void cacheMissClassifiesCallsLlmStoresCacheAndRecordsVersions() {
        float[] embedding = {0.3f};
        LlmResult result = new LlmResult("answer", "hard", "model-h", 12, 34);
        when(routerClient.embed("question")).thenReturn(embedding);
        when(cacheService.get(embedding)).thenReturn(Optional.empty());
        when(routerService.classify("question")).thenReturn(RouterService.Tier.HARD);
        when(llmService.call(RouterService.Tier.HARD, "question")).thenReturn(result);
        when(routerClient.routerModelVersion()).thenReturn("router-v1");
        when(routerClient.embedModelVersion()).thenReturn("embed-v1");

        ChatResponse response = service.chat(request("question"));

        assertThat(response.getResponse()).isEqualTo("answer");
        assertThat(response.isCacheHit()).isFalse();
        assertThat(response.getTier()).isEqualTo("hard");
        verify(cacheService).put("question", embedding, result);
        verify(requestLogService).record(response, "question", "router-v1", "embed-v1", null, null);
    }

    @Test
    void embeddingFailureIsRecordedWithoutTierAndRethrown() {
        RuntimeException failure = new RuntimeException("embed down");
        when(routerClient.embed("q")).thenThrow(failure);

        assertThatThrownBy(() -> service.chat(request("q"))).isSameAs(failure);
        verify(requestLogService).recordFailure(eq("00000000-0000-0000-0000-000000000001"),
                eq("q"), isNull(), anyLong(), same(failure));
    }

    @Test
    void llmFailureIsRecordedWithClassifiedTierAndRethrown() {
        RuntimeException failure = new RuntimeException("llm down");
        float[] embedding = {1f};
        when(routerClient.embed("q")).thenReturn(embedding);
        when(cacheService.get(embedding)).thenReturn(Optional.empty());
        when(routerService.classify("q")).thenReturn(RouterService.Tier.MEDIUM);
        when(llmService.call(RouterService.Tier.MEDIUM, "q")).thenThrow(failure);

        assertThatThrownBy(() -> service.chat(request("q"))).isSameAs(failure);
        verify(requestLogService).recordFailure(eq("00000000-0000-0000-0000-000000000001"),
                eq("q"), eq(RouterService.Tier.MEDIUM), anyLong(), same(failure));
        verify(cacheService, never()).put(anyString(), any(), any());
    }

    private ChatRequest request(String message) {
        ChatRequest request = new ChatRequest();
        ReflectionTestUtils.setField(request, "message", message);
        ReflectionTestUtils.setField(request, "conversationId", "conversation-1");
        return request;
    }
}
