package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClientException;
import com.llmrouter.client.RouterRerank;
import com.llmrouter.dto.CachedResponse;
import com.llmrouter.dto.RerankShadow;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 캐시 히트를 cross-encoder 로 재검증한다 — 단, 아직 판정은 바꾸지 않는다.
 *
 * <p>왜 shadow 인가: 58쌍 오프라인 스크리닝에서 사전 gate(false hit 0건 +
 * recall 90% 이상)를 통과한 모델이 하나도 없었다. 최선 조합조차 오탐 2건이 남았다.
 * 그 임계값을 그대로 운영에 걸면 측정되지 않은 오탐률을 사용자에게 떠넘기는 것이다.
 * 실제 트래픽에서 점수 분포를 먼저 모으고, 독립 평가셋을 통과한 뒤에 켠다.
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
     * 히트한 후보에 점수를 매긴다. 실패하면 조용히 비워둔다.
     *
     * <p>shadow 의 규칙은 "관측이 동작을 바꾸지 않는다"이다. 리랭커가 죽어도
     * 사용자 응답에는 영향이 없어야 하므로 예외를 삼킨다. 실제로 판정에 쓰기
     * 시작하면 그때는 fail-closed(미스 처리)로 바꿔야 한다.
     */
    public RerankShadow evaluate(String question, CachedResponse hit) {
        if (!enabled || hit.question() == null || hit.question().isBlank()) {
            return RerankShadow.skipped();
        }
        try {
            RouterRerank result = routerClient.rerank(question, List.of(hit.question()));
            double score = result.scores()[0];
            // 스크리닝이 고른 건 두 조건의 AND 였다. 코사인 단독으로는 두 분포가
            // 겹쳐서 갈리지 않았기 때문에 한쪽만 보면 재현이 안 된다.
            boolean wouldReject =
                    !(hit.similarity() >= similarityThreshold && score >= scoreThreshold);
            if (wouldReject) {
                log.info("리랭커 shadow: 실제 적용 시 거절했을 히트 id={} similarity={} score={}",
                        hit.cacheEntryId(), hit.similarity(), score);
            }
            return new RerankShadow(score, result.modelVersion(), wouldReject);
        } catch (RouterClientException error) {
            // 관측이 응답을 막으면 안 된다. 점수만 비워두고 넘어간다.
            log.warn("리랭커 shadow 실패 — 캐시 판정은 그대로 둔다 status={}",
                    error.statusCode(), error);
            return RerankShadow.skipped();
        }
    }
}
