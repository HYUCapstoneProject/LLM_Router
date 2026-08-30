package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterRerank;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.RerankShadow;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 캐시 후보를 cross-encoder 로 재검증한다 — 단, 아직 판정은 바꾸지 않는다.
 *
 * <p>왜 shadow 인가: 58쌍 오프라인 스크리닝에서 사전 gate(false hit 0건 +
 * recall 90% 이상)를 통과한 모델이 하나도 없었다. 최선 조합조차 오탐 2건이 남았다.
 * 그 임계값을 그대로 운영에 걸면 측정되지 않은 오탐률을 사용자에게 떠넘기는 것이다.
 *
 * <p>관측하는 것이 두 가지다.
 * <ul>
 *   <li><b>거부</b> — 서빙된 후보를 리랭커가 막았을지</li>
 *   <li><b>재정렬</b> — 리랭커 1등이 코사인 1등과 달랐는지. 후보를 top-k 로 넘기기
 *       전에는 잴 수 없던 값이다</li>
 * </ul>
 *
 * <p>비용도 공짜가 아니다. 리랭커는 캐시 <b>히트</b> 경로에서 돈다 — 빠르려고 만든
 * 그 경로다. FastAPI 가 다른 리전이라 왕복만 약 120ms 라서, 기본값은 꺼짐이다.
 */
@Slf4j
@Service
public class RerankShadowService {

    private final RouterClient routerClient;
    private final boolean enabled;
    private final double similarityThreshold;
    private final double scoreThreshold;

    public RerankShadowService(
            RouterClient routerClient,
            @Value("${llm.cache.rerank.shadow-enabled:false}") boolean enabled,
            @Value("${llm.cache.rerank.similarity-threshold:0.8774}") double similarityThreshold,
            @Value("${llm.cache.rerank.score-threshold:0.03581769}") double scoreThreshold) {
        this.routerClient = routerClient;
        this.enabled = enabled;
        this.similarityThreshold = similarityThreshold;
        this.scoreThreshold = scoreThreshold;
    }

    /**
     * 후보 전체에 점수를 매긴다. 실패하면 조용히 비워둔다.
     *
     * <p>shadow 의 규칙은 "관측이 동작을 바꾸지 않는다"이다. 리랭커가 죽어도
     * 사용자 응답에는 영향이 없어야 하므로 예외를 삼킨다. 실제로 판정에 쓰기
     * 시작하면 그때는 fail-closed(미스 처리)로 바꿔야 한다.
     *
     * @param candidates 유사도 내림차순. 첫 원소가 실제로 서빙되는 후보다.
     */
    public RerankShadow evaluate(String question, List<CachedResponse> candidates) {
        if (!enabled || candidates.isEmpty()) {
            return RerankShadow.skipped();
        }
        // 05 마이그레이션 이전 행은 question 이 비어 있을 수 있다. 후보에서 뺀다.
        List<CachedResponse> scorable = candidates.stream()
                .filter(c -> c.question() != null && !c.question().isBlank())
                .toList();
        CachedResponse served = candidates.get(0);
        if (scorable.isEmpty() || !scorable.contains(served)) {
            // 서빙된 후보를 채점할 수 없으면 남길 점수가 없다.
            return RerankShadow.skipped();
        }

        try {
            List<String> texts = new ArrayList<>(scorable.size());
            for (CachedResponse candidate : scorable) {
                texts.add(candidate.question());
            }
            RouterRerank result = routerClient.rerank(question, texts);

            int servedIndex = scorable.indexOf(served);
            double servedScore = result.scores()[servedIndex];

            int bestIndex = 0;
            for (int i = 1; i < result.scores().length; i++) {
                if (result.scores()[i] > result.scores()[bestIndex]) {
                    bestIndex = i;
                }
            }

            // 스크리닝이 고른 건 두 조건의 AND 였다. 코사인 단독으로는 두 분포가
            // 겹쳐서 갈리지 않았기 때문에 한쪽만 보면 재현이 안 된다.
            boolean wouldReject =
                    !(served.similarity() >= similarityThreshold && servedScore >= scoreThreshold);
            boolean wouldReorder = bestIndex != servedIndex;

            if (wouldReject) {
                log.info("리랭커 shadow: 적용 시 거절했을 히트 id={} similarity={} score={}",
                        served.cacheEntryId(), served.similarity(), servedScore);
            }
            if (wouldReorder) {
                log.info("리랭커 shadow: 적용 시 다른 후보를 골랐을 요청 "
                                + "코사인1등 id={} score={} / 리랭커1등 id={} score={}",
                        served.cacheEntryId(), servedScore,
                        scorable.get(bestIndex).cacheEntryId(), result.scores()[bestIndex]);
            }
            return new RerankShadow(servedScore, result.modelVersion(), wouldReject,
                    scorable.size(), wouldReorder);
        } catch (RouterClientException error) {
            // 관측이 응답을 막으면 안 된다. 점수만 비워두고 넘어간다.
            log.warn("리랭커 shadow 실패 — 캐시 판정은 그대로 둔다 status={}",
                    error.statusCode(), error);
            return RerankShadow.skipped();
        }
    }
}
