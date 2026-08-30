package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterRerank;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.RerankShadow;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RerankShadowServiceTest {

    @Mock RouterClient routerClient;

    /** 스크리닝이 고른 AND 조건 값 */
    private static final double SIMILARITY = 0.8774;
    private static final double SCORE = 0.03581769;

    private RerankShadowService service(boolean enabled) {
        return new RerankShadowService(routerClient, enabled, SIMILARITY, SCORE);
    }

    private static CachedResponse hit(double similarity) {
        return new CachedResponse(1L, "저장된 질문", "답", "easy", "model-e", similarity);
    }

    @Test
    void doesNothingWhenDisabled() {
        RerankShadow shadow = service(false).evaluate("질문", hit(0.99));

        assertThat(shadow.score()).isNull();
        assertThat(shadow.wouldReject()).isNull();
        verifyNoInteractions(routerClient);
    }

    @Test
    void recordsScoreAndKeepsHitWhenBothConditionsPass() {
        when(routerClient.rerank("질문", List.of("저장된 질문")))
                .thenReturn(new RouterRerank(new double[]{0.9}, "rerank-v1"));

        RerankShadow shadow = service(true).evaluate("질문", hit(0.95));

        assertThat(shadow.score()).isEqualTo(0.9);
        assertThat(shadow.modelVersion()).isEqualTo("rerank-v1");
        assertThat(shadow.wouldReject()).isFalse();
    }

    /**
     * 스크리닝이 고른 건 AND 조건이다. 코사인만으로는 paraphrase 와 hard negative 가
     * 겹쳐서 갈리지 않았으므로 한쪽만 봐서는 결과가 재현되지 않는다.
     */
    @Test
    void wouldRejectWhenRerankScoreIsBelowThresholdEvenIfSimilarityIsHigh() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.001}, "rerank-v1"));

        assertThat(service(true).evaluate("질문", hit(0.99)).wouldReject()).isTrue();
    }

    @Test
    void wouldRejectWhenSimilarityIsBelowThresholdEvenIfScoreIsHigh() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.99}, "rerank-v1"));

        assertThat(service(true).evaluate("질문", hit(0.80)).wouldReject()).isTrue();
    }

    /**
     * shadow 의 규칙은 "관측이 동작을 바꾸지 않는다"이다.
     * 리랭커가 죽어도 사용자 응답에는 영향이 없어야 한다.
     */
    @Test
    void swallowsRerankFailureSoTheCachedAnswerStillGoesOut() {
        when(routerClient.rerank(anyString(), any()))
                .thenThrow(new RouterClientException("리랭커 없음", 503));

        RerankShadow shadow = service(true).evaluate("질문", hit(0.95));

        assertThat(shadow.score()).isNull();
        assertThat(shadow.wouldReject()).isNull();
    }

    /** 05 마이그레이션 이전 캐시 행은 질문 원문이 없어 후보로 넘길 것이 없다. */
    @Test
    void skipsWhenCandidateQuestionIsMissing() {
        CachedResponse legacy = new CachedResponse(1L, null, "답", "easy", "model-e", 0.95);

        assertThat(service(true).evaluate("질문", legacy).score()).isNull();
        verifyNoInteractions(routerClient);
    }
}
