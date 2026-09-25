package com.llmrouter.client;

import com.llmrouter.service.RouterService;

import java.util.Map;

/** FastAPI 라우팅 결과. 상대 품질 점수를 사용하는 v2 pairwise 결과에는 confidence가 없다. */
public record RouterClassification(
        RouterService.Tier tier,
        Double confidence,
        String modelVersion,
        DecisionMetadata decisionMetadata
) {
    public RouterClassification(RouterService.Tier tier, Double confidence, String modelVersion) {
        this(tier, confidence, modelVersion, null);
    }

    /** v2 정책을 요청 로그에 보존한다. 명시적 성공 판정은 이 점수와 별도로 수집한다. */
    public record DecisionMetadata(
            String objective,
            String policyVersion,
            String policySha256,
            String candidatePoolVersion,
            Map<String, String> candidateModelIds,
            String action,
            String scoreSemantics,
            Map<String, Double> actionScores
    ) {
        public DecisionMetadata {
            candidateModelIds = Map.copyOf(candidateModelIds);
            actionScores = Map.copyOf(actionScores);
        }
    }
}
