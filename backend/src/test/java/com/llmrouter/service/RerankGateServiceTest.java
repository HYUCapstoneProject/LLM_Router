package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterRerank;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.RerankDecision;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RerankGateServiceTest {

    @Mock RouterClient routerClient;

    /** 스크리닝이 고른 AND 조건 값 */
    private static final double SIMILARITY = 0.8774;
    private static final double SCORE = 0.03581769;

    private RerankGateService gate(String mode) {
        return new RerankGateService(routerClient, mode, SIMILARITY, SCORE);
    }

    private static CachedResponse candidate(long id, String question, double similarity) {
        return new CachedResponse(id, question, "답", "easy", "model-e", similarity);
    }

    private static List<CachedResponse> one(double similarity) {
        return List.of(candidate(1L, "저장된 질문", similarity));
    }

    @Test
    void offModeNeverCallsTheReranker() {
        RerankDecision d = gate("off").evaluate("질문", one(0.95));

        assertThat(d.accepted()).isNotNull();
        assertThat(d.score()).isNull();
        verifyNoInteractions(routerClient);
    }

    @Test
    void unknownModeFailsFastInsteadOfSilentlyDisabling() {
        // 오타가 조용히 off 로 떨어지면 리랭커가 안 도는 걸 아무도 모른다.
        assertThatThrownBy(() -> gate("enfroce"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("off|shadow|enforce");
    }

    // --- enforce ---------------------------------------------------------

    @Test
    void enforceServesTheCandidateWhenBothConditionsPass() {
        when(routerClient.rerank("질문", List.of("저장된 질문")))
                .thenReturn(new RouterRerank(new double[]{0.9}, "rerank-v1"));

        RerankDecision d = gate("enforce").evaluate("질문", one(0.95));

        assertThat(d.accepted()).isNotNull();
        assertThat(d.wouldReject()).isFalse();
        assertThat(d.score()).isEqualTo(0.9);
        assertThat(d.modelVersion()).isEqualTo("rerank-v1");
    }

    /**
     * 실측에서 막힌 유형이다 — Python↔Java 같은 entity swap 은 코사인 0.9371 인데
     * 리랭커 점수가 0.0005 였다.
     */
    @Test
    void enforceRejectsWhenRerankScoreIsLowEvenIfSimilarityIsHigh() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.0005}, "rerank-v1"));

        RerankDecision d = gate("enforce").evaluate("질문", one(0.9371));

        assertThat(d.accepted()).isNull();          // 캐시 미스로 떨어진다
        assertThat(d.wouldReject()).isTrue();
        assertThat(d.reason()).isEqualTo("rejected");
    }

    @Test
    void enforceRejectsWhenSimilarityIsBelowTheJointThreshold() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.99}, "rerank-v1"));

        assertThat(gate("enforce").evaluate("질문", one(0.80)).accepted()).isNull();
    }

    /** 코사인 1등이 아니라 리랭커 1등을 서빙해야 재정렬이 의미가 있다. */
    @Test
    void enforcePicksTheRerankWinnerNotTheCosineWinner() {
        List<CachedResponse> candidates = List.of(
                candidate(1L, "코사인 1등", 0.95),
                candidate(2L, "리랭커 1등", 0.90));
        when(routerClient.rerank("질문", List.of("코사인 1등", "리랭커 1등")))
                .thenReturn(new RouterRerank(new double[]{0.20, 0.95}, "rerank-v1"));

        RerankDecision d = gate("enforce").evaluate("질문", candidates);

        assertThat(d.accepted().cacheEntryId()).isEqualTo(2L);
        assertThat(d.wouldReorder()).isTrue();
        assertThat(d.reason()).isEqualTo("accepted_reordered");
    }

    /**
     * fail-closed. 검증할 수 없는 캐시 답을 내보내는 것보다
     * Claude 를 한 번 더 부르는 편이 싸다.
     */
    @Test
    void enforceFallsBackToCacheMissWhenTheRerankerIsUnavailable() {
        when(routerClient.rerank(anyString(), any()))
                .thenThrow(new RouterClientException("리랭커 없음", 503));

        RerankDecision d = gate("enforce").evaluate("질문", one(0.95));

        assertThat(d.accepted()).isNull();
        assertThat(d.reason()).isEqualTo("unavailable_fail_closed");
    }

    // --- shadow ----------------------------------------------------------

    @Test
    void shadowRecordsTheVerdictButStillServesTheCosineWinner() {
        when(routerClient.rerank(anyString(), any()))
                .thenReturn(new RouterRerank(new double[]{0.0005}, "rerank-v1"));

        RerankDecision d = gate("shadow").evaluate("질문", one(0.9371));

        assertThat(d.accepted()).isNotNull();       // 동작은 바뀌지 않는다
        assertThat(d.wouldReject()).isTrue();       // 다만 기록은 남는다
        assertThat(d.reason()).isEqualTo("shadow");
    }

    @Test
    void shadowSwallowsRerankFailureSoTheCachedAnswerStillGoesOut() {
        when(routerClient.rerank(anyString(), any()))
                .thenThrow(new RouterClientException("리랭커 없음", 503));

        RerankDecision d = gate("shadow").evaluate("질문", one(0.95));

        assertThat(d.accepted()).isNotNull();
        assertThat(d.score()).isNull();
    }

    // --- 경계 ------------------------------------------------------------

    @Test
    void noCandidatesMeansCacheMissWithoutCallingTheReranker() {
        assertThat(gate("enforce").evaluate("질문", List.of()).accepted()).isNull();
        verifyNoInteractions(routerClient);
    }

    /**
     * 05 마이그레이션 이전 행은 질문 원문이 없다. 이것 때문에 캐시를 통째로
     * 무력화하면 손해가 더 크므로 enforce 에서도 막지 않는다.
     */
    @Test
    void legacyRowsWithoutQuestionTextArePassedThroughEvenInEnforce() {
        CachedResponse legacy = new CachedResponse(1L, null, "답", "easy", "model-e", 0.95);

        RerankDecision d = gate("enforce").evaluate("질문", List.of(legacy));

        assertThat(d.accepted()).isNotNull();
        assertThat(d.reason()).isEqualTo("no_scorable_candidate");
        verifyNoInteractions(routerClient);
    }
}
