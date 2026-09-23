package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterRerank;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.RerankDecision;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 캐시 후보를 cross-encoder 로 재검증한다.
 *
 * <p>bi-encoder 는 두 문장을 각각 따로 압축하므로 {@code Python} 이 {@code Java} 로
 * 바뀌어도 벡터가 거의 안 움직인다. 배포된 서버에 92문항을 던져 측정한 결과다.
 *
 * <pre>
 *                        recall    오탐    오탐률
 *   코사인 단독(0.80)     100.0%     19     36.5%
 *   코사인 AND 리랭커      90.5%      1      1.9%
 * </pre>
 *
 * <p>정탐 2건을 잃고 오탐 18건을 막는다. 잃은 2건은 Claude 를 한 번 더 부를 뿐
 * 사용자는 정상 답변을 받지만, 막지 못한 오탐은 <b>틀린 답이 그대로 나간다.</b>
 * 그래서 이 거래를 받는다.
 */
@Slf4j
@Service
public class RerankGateService {

    private final RouterClient routerClient;
    private final RerankMode mode;
    private final double similarityThreshold;
    private final double scoreThreshold;

    public RerankGateService(
            RouterClient routerClient,
            @Value("${llm.cache.rerank.mode:off}") String mode,
            @Value("${llm.cache.rerank.similarity-threshold:0.8774}") double similarityThreshold,
            @Value("${llm.cache.rerank.score-threshold:0.03581769}") double scoreThreshold) {
        this.routerClient = routerClient;
        this.mode = RerankMode.parse(mode);
        this.similarityThreshold = similarityThreshold;
        this.scoreThreshold = scoreThreshold;
        log.info("리랭커 게이트 mode={} 코사인>={} 점수>={}",
                this.mode, similarityThreshold, scoreThreshold);
    }

    /**
     * @param candidates 유사도 내림차순. 비어 있으면 호출하지 않는다.
     * @return {@link RerankDecision#accepted()} 가 null 이면 호출자는 캐시 미스로 처리한다.
     */
    public RerankDecision evaluate(String question, List<CachedResponse> candidates) {
        if (candidates.isEmpty()) {
            return RerankDecision.passthrough(null, "no_candidates");
        }
        CachedResponse cosineBest = candidates.get(0);
        if (mode == RerankMode.OFF) {
            return RerankDecision.passthrough(cosineBest, "disabled");
        }

        // 05 마이그레이션 이전 행은 question 이 비어 있어 리랭커에 넘길 원문이 없다.
        List<CachedResponse> scorable = candidates.stream()
                .filter(c -> c.question() != null && !c.question().isBlank())
                .toList();
        if (scorable.isEmpty()) {
            // 채점할 수 없는 후보뿐이다. enforce 에서도 막지 않는다 —
            // 레거시 행 때문에 캐시를 통째로 무력화하면 손해가 더 크다.
            return RerankDecision.passthrough(cosineBest, "no_scorable_candidate");
        }

        double[] scores;
        String modelVersion;
        try {
            List<String> texts = new ArrayList<>(scorable.size());
            for (CachedResponse candidate : scorable) {
                texts.add(candidate.question());
            }
            RouterRerank result = routerClient.rerank(question, texts);
            scores = result.scores();
            modelVersion = result.modelVersion();
        } catch (RouterClientException error) {
            return onFailure(cosineBest, error);
        }

        // AND 조건이다. 코사인 단독으로는 paraphrase 와 hard negative 분포가 겹쳐
        // 갈리지 않으므로 한쪽만 보면 측정 결과가 재현되지 않는다.
        int accepted = -1;
        for (int i = 0; i < scorable.size(); i++) {
            if (scorable.get(i).similarity() >= similarityThreshold && scores[i] >= scoreThreshold
                    && (accepted < 0 || scores[i] > scores[accepted])) {
                accepted = i;
            }
        }

        int rerankBest = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[rerankBest]) {
                rerankBest = i;
            }
        }
        int servedIndex = scorable.indexOf(cosineBest);
        double servedScore = servedIndex >= 0 ? scores[servedIndex] : scores[rerankBest];
        boolean wouldReject = accepted < 0;
        boolean wouldReorder = servedIndex >= 0 && rerankBest != servedIndex;

        if (mode == RerankMode.SHADOW) {
            // 관측만 한다. 서빙되는 답은 코사인 1등 그대로다.
            if (wouldReject) {
                log.info("리랭커 shadow: 적용 시 거절했을 히트 id={} similarity={} score={}",
                        cosineBest.cacheEntryId(), cosineBest.similarity(), servedScore);
            }
            return new RerankDecision(cosineBest, servedScore, modelVersion,
                    wouldReject, scorable.size(), wouldReorder, "shadow");
        }

        if (wouldReject) {
            log.info("리랭커 거부 — 캐시 미스로 처리한다 id={} similarity={} score={}",
                    cosineBest.cacheEntryId(), cosineBest.similarity(), servedScore);
            return new RerankDecision(null, servedScore, modelVersion,
                    true, scorable.size(), wouldReorder, "rejected");
        }

        CachedResponse chosen = scorable.get(accepted);
        return new RerankDecision(chosen, scores[accepted], modelVersion,
                false, scorable.size(), wouldReorder,
                accepted == servedIndex ? "accepted" : "accepted_reordered");
    }

    /**
     * 리랭커 장애 처리.
     *
     * <p>shadow 에서는 삼킨다 — 관측이 서비스를 막으면 안 된다.
     * enforce 에서는 <b>캐시 미스로 떨어뜨린다(fail-closed)</b>. 검증할 수 없는 캐시 답을
     * 내보내는 것보다 Claude 를 한 번 더 부르는 편이 싸다.
     */
    private RerankDecision onFailure(CachedResponse cosineBest, RouterClientException error) {
        if (mode == RerankMode.SHADOW) {
            log.warn("리랭커 shadow 실패 — 캐시 판정은 그대로 둔다 status={}",
                    error.statusCode(), error);
            return RerankDecision.passthrough(cosineBest, "unavailable_shadow");
        }
        log.warn("리랭커 실패 — 캐시 미스로 처리한다(fail-closed) status={}",
                error.statusCode(), error);
        return RerankDecision.passthrough(null, "unavailable_fail_closed");
    }
}
