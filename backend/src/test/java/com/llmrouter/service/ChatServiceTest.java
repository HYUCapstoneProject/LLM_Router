package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClassification;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterEmbedding;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.ChatRequest;
import com.llmrouter.dto.ChatResponse;
import com.llmrouter.dto.LlmResult;
import com.llmrouter.dto.RequestLogEntry;
import com.llmrouter.dto.RerankShadow;
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
    @Mock RerankShadowService rerankShadowService;
    ChatService service;

    @BeforeEach
    void setUp() {
        service = new ChatService(routerClient, routerService, cacheService, llmService,
                requestLogService, rerankShadowService);
        MDC.put("requestId", "00000000-0000-0000-0000-000000000001");
        // exact match 는 이제 모든 요청의 첫 단계다. 기본값은 미스로 두고
        // exact 를 검증하는 테스트에서만 따로 덮어쓴다.
        when(cacheService.getExact(anyString())).thenReturn(Optional.empty());
    }

    @Test
    void cacheHitSkipsClassifierAndLlmAndRecordsSimilarity() {
        float[] embedding = {0.1f, 0.2f};
        when(routerClient.embed("hello")).thenReturn(new RouterEmbedding(embedding, "embed-v1"));
        CachedResponse hit = new CachedResponse(7L, "hello", "cached", "easy", "model-e", 0.97);
        when(cacheService.get(embedding)).thenReturn(Optional.of(hit));
        when(rerankShadowService.evaluate("hello", hit))
                .thenReturn(new RerankShadow(0.91, "rerank-v1", false));

        ChatResponse response = service.chat(request("hello"));

        assertThat(response.getResponse()).isEqualTo("cached");
        assertThat(response.isCacheHit()).isTrue();
        assertThat(response.getInputTokens()).isZero();
        verifyNoInteractions(routerService, llmService);
        verify(cacheService, never()).put(anyString(), any(), any(), anyString());
        verify(requestLogService).record(argThat(entry ->
                entry.requestId().equals(response.getRequestId())
                        && entry.question().equals("hello")
                        && entry.conversationId().equals("conversation-1")
                        && entry.cacheHit()
                        && entry.cacheEntryId().equals(7L)
                        && entry.similarity().equals(0.97)
                        && entry.cacheMatchType().equals(RequestLogEntry.MATCH_SEMANTIC)
                        && entry.rerankScore().equals(0.91)
                        && entry.rerankModelVersion().equals("rerank-v1")
                        && !entry.rerankWouldReject()
                        && entry.embedModelVersion().equals("embed-v1")
                        && entry.origin().equals(RequestLogEntry.ORIGIN_CHAT)));
    }

    /**
     * exact 히트는 임베딩 호출 자체를 건너뛴다. 이게 이 계층의 존재 이유다 —
     * FastAPI 가 다른 리전이라 embed 왕복만 약 120ms 다.
     */
    @Test
    void exactMatchSkipsEmbeddingEntirely() {
        when(cacheService.getExact("hello")).thenReturn(Optional.of(
                new CachedResponse(3L, "hello", "cached", "easy", "model-e", 1.0)));

        ChatResponse response = service.chat(request("hello"));

        assertThat(response.getResponse()).isEqualTo("cached");
        assertThat(response.isCacheHit()).isTrue();
        assertThat(response.getInputTokens()).isZero();
        verifyNoInteractions(routerClient, routerService, llmService);
        verify(cacheService, never()).get(any());
        verify(requestLogService).record(argThat(entry ->
                entry.cacheHit()
                        && entry.cacheMatchType().equals(RequestLogEntry.MATCH_EXACT)
                        // 임베딩을 부르지 않았으므로 버전이 없다
                        && entry.embedModelVersion() == null
                        && entry.similarity().equals(1.0)
                        && entry.cacheEntryId().equals(3L)));
    }

    @Test
    void cacheMissClassifiesCallsLlmStoresCacheAndRecordsVersions() {
        float[] embedding = {0.3f};
        LlmResult result = new LlmResult("answer", "hard", "model-h", 12, 34);
        when(routerClient.embed("question")).thenReturn(new RouterEmbedding(embedding, "embed-v1"));
        when(cacheService.get(embedding)).thenReturn(Optional.empty());
        when(routerService.classify("question"))
                .thenReturn(new RouterClassification(RouterService.Tier.HARD, 0.9, "router-v1"));
        when(llmService.call(RouterService.Tier.HARD, "question")).thenReturn(result);
        when(cacheService.put("question", embedding, result, "embed-v1")).thenReturn(9L);

        ChatResponse response = service.chat(request("question"));

        assertThat(response.getResponse()).isEqualTo("answer");
        assertThat(response.isCacheHit()).isFalse();
        assertThat(response.getTier()).isEqualTo("hard");
        verify(cacheService).put("question", embedding, result, "embed-v1");
        verify(requestLogService).record(argThat(entry ->
                entry.requestId().equals(response.getRequestId())
                        && entry.question().equals("question")
                        && entry.conversationId().equals("conversation-1")
                        && !entry.cacheHit()
                        && entry.cacheEntryId().equals(9L)
                        && entry.routerModelVersion().equals("router-v1")
                        && entry.routerConfidence().equals(0.9)
                        && entry.embedModelVersion().equals("embed-v1")
                        && entry.origin().equals(RequestLogEntry.ORIGIN_CHAT)));
    }

    @Test
    void unexpectedEmbeddingFailureIsRecordedWithoutTierAndRethrown() {
        RuntimeException failure = new RuntimeException("embed down");
        when(routerClient.embed("q")).thenThrow(failure);

        assertThatThrownBy(() -> service.chat(request("q"))).isSameAs(failure);
        verify(requestLogService).recordFailure(eq("00000000-0000-0000-0000-000000000001"),
                eq("q"), isNull(), anyLong(), same(failure));
    }

    @Test
    void embeddingApiFailureBypassesCacheButStillUsesClassifier() {
        when(routerClient.embed("q"))
                .thenThrow(new RouterClientException("embed down", 503));
        when(routerService.classify("q"))
                .thenReturn(new RouterClassification(RouterService.Tier.HARD, 0.8, "router-v1"));
        LlmResult result = new LlmResult("answer", "hard", "model-h", 1, 2);
        when(llmService.call(RouterService.Tier.HARD, "q")).thenReturn(result);

        ChatResponse response = service.chat(request("q"));

        assertThat(response.getResponse()).isEqualTo("answer");
        // exact 조회는 DB만 보므로 임베딩 장애와 무관하게 시도된다.
        // 임베딩이 없으면 벡터 조회와 저장만 건너뛴다.
        verify(cacheService).getExact("q");
        verify(cacheService, never()).get(any());
        verify(cacheService, never()).put(anyString(), any(), any(), anyString());
        verify(requestLogService).record(argThat(entry ->
                entry.fallbackReason().equals("embed_unavailable")
                        && entry.embedModelVersion() == null
                        && entry.routerModelVersion().equals("router-v1")
                        && entry.cacheEntryId() == null));
    }

    @Test
    void classifierApiFailureUsesMediumAndRecordsFallback() {
        float[] embedding = {1f};
        when(routerClient.embed("q")).thenReturn(new RouterEmbedding(embedding, "embed-v1"));
        when(cacheService.get(embedding)).thenReturn(Optional.empty());
        when(routerService.classify("q"))
                .thenThrow(new RouterClientException("router down", 503));
        LlmResult result = new LlmResult("answer", "medium", "model-m", 1, 2);
        when(llmService.call(RouterService.Tier.MEDIUM, "q")).thenReturn(result);
        when(cacheService.put("q", embedding, result, "embed-v1")).thenReturn(10L);

        ChatResponse response = service.chat(request("q"));

        assertThat(response.getTier()).isEqualTo("medium");
        verify(requestLogService).record(argThat(entry ->
                entry.fallbackReason().equals("classify_unavailable")
                        && entry.routerModelVersion() == null
                        && entry.cacheEntryId().equals(10L)));
    }

    @Test
    void llmFailureIsRecordedWithClassifiedTierAndRethrown() {
        RuntimeException failure = new RuntimeException("llm down");
        float[] embedding = {1f};
        when(routerClient.embed("q")).thenReturn(new RouterEmbedding(embedding, "embed-v1"));
        when(cacheService.get(embedding)).thenReturn(Optional.empty());
        when(routerService.classify("q"))
                .thenReturn(new RouterClassification(RouterService.Tier.MEDIUM, 0.7, "router-v1"));
        when(llmService.call(RouterService.Tier.MEDIUM, "q")).thenThrow(failure);

        assertThatThrownBy(() -> service.chat(request("q"))).isSameAs(failure);
        verify(requestLogService).recordFailure(eq("00000000-0000-0000-0000-000000000001"),
                eq("q"), eq(RouterService.Tier.MEDIUM), anyLong(), same(failure));
        verify(cacheService, never()).put(anyString(), any(), any(), anyString());
    }

    private ChatRequest request(String message) {
        ChatRequest request = new ChatRequest();
        ReflectionTestUtils.setField(request, "message", message);
        ReflectionTestUtils.setField(request, "conversationId", "conversation-1");
        return request;
    }
}
