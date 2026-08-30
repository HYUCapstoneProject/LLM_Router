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
        return candidate(1L, "저장된 질문", similarity);
    }

    private static CachedResponse candidate(long id, String question, double similarity) {
        return new CachedResponse(id, question, "답", "easy", "model-e", similarity);
    }

    @Test
    void doesNothingWhenDisabled() {
        RerankShadow shadow = service(false).evaluate("질문", List.of(hit(0.99)));

        assertThat(shadow.score()).isNull();
        assertThat(shadow.wouldReject()).isNull();
        verifyNoInteractions(routerClient);
    }

    @Test
    void recordsScoreAndKeepsHitWhenBothConditionsPass() {
        when(routerClient.rerank("질문", List.of("저장된 질문")))
                .thenReturn(new RouterRerank(new double[]{0.9}, "rerank-v1"));

        RerankShadow shadow = service(true).evaluate("질문", List.of(hit(0.95)));

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

        assertThat(service(true).evaluate("질문", List.of(hit(0.99))).wouldReject()).isTrue();
    }

    @Test
    void wouldRejectWhenSimilarityIsBelowThresholdEvenIfScoreIsHigh() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.99}, "rerank-v1"));

        assertThat(service(true).evaluate("질문", List.of(hit(0.80))).wouldReject()).isTrue();
    }

    /**
     * shadow 의 규칙은 "관측이 동작을 바꾸지 않는다"이다.
     * 리랭커가 죽어도 사용자 응답에는 영향이 없어야 한다.
     */
    @Test
    void swallowsRerankFailureSoTheCachedAnswerStillGoesOut() {
        when(routerClient.rerank(anyString(), any()))
                .thenThrow(new RouterClientException("리랭커 없음", 503));

        RerankShadow shadow = service(true).evaluate("질문", List.of(hit(0.95)));

        assertThat(shadow.score()).isNull();
        assertThat(shadow.wouldReject()).isNull();
    }

    /**
     * 리랭커 1등이 코사인 1등과 다르면 재정렬로 기록한다.
     *
     * 후보를 top-k 로 넘기기 전에는 잴 수 없던 값이다. 최선 1건만 넘기면
     * 리랭커는 거부권만 행사할 수 있다.
     */
    @Test
    void detectsThatRerankerWouldHavePickedADifferentCandidate() {
        List<CachedResponse> candidates = List.of(
                candidate(1L, "코사인 1등", 0.95),
                candidate(2L, "리랭커 1등", 0.90));
        when(routerClient.rerank("질문", List.of("코사인 1등", "리랭커 1등")))
                .thenReturn(new RouterRerank(new double[]{0.20, 0.95}, "rerank-v1"));

        RerankShadow shadow = service(true).evaluate("질문", candidates);

        // 서빙된 건 여전히 코사인 1등이므로 점수도 그쪽 것을 남긴다.
        assertThat(shadow.score()).isEqualTo(0.20);
        assertThat(shadow.candidateCount()).isEqualTo(2);
        assertThat(shadow.wouldReorder()).isTrue();
        assertThat(shadow.wouldReject()).isFalse();
    }

    @Test
    void singleCandidateCanNeverReorder() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.9}, "rerank-v1"));

        RerankShadow shadow = service(true).evaluate("질문", List.of(hit(0.95)));

        assertThat(shadow.candidateCount()).isEqualTo(1);
        assertThat(shadow.wouldReorder()).isFalse();
    }

    /** 원문 없는 레거시 행은 후보에서 빠지되 나머지는 정상 채점되어야 한다. */
    @Test
    void skipsLegacyCandidatesWithoutQuestionText() {
        List<CachedResponse> candidates = List.of(
                candidate(1L, "정상 후보", 0.95),
                new CachedResponse(2L, null, "답", "easy", "model-e", 0.90));
        when(routerClient.rerank("질문", List.of("정상 후보")))
                .thenReturn(new RouterRerank(new double[]{0.8}, "rerank-v1"));

        RerankShadow shadow = service(true).evaluate("질문", candidates);

        assertThat(shadow.candidateCount()).isEqualTo(1);
        assertThat(shadow.score()).isEqualTo(0.8);
    }

    @Test
    void skipsWhenThereAreNoCandidates() {
        assertThat(service(true).evaluate("질문", List.of()).score()).isNull();
        verifyNoInteractions(routerClient);
    }

    /** 서빙된 후보 자체에 원문이 없으면 남길 점수가 없다. */
    @Test
    void skipsWhenServedCandidateHasNoQuestionText() {
        CachedResponse legacy = new CachedResponse(1L, null, "답", "easy", "model-e", 0.95);

        assertThat(service(true).evaluate("질문", List.of(legacy)).score()).isNull();
        verifyNoInteractions(routerClient);
    }
}
