package com.llmrouter.client;

import com.llmrouter.service.RouterService;

/** FastAPI 라우팅 응답. model 필드는 참고값이라 Spring의 tier 매핑을 우선한다. */
public record RouterClassification(
        RouterService.Tier tier,
        double confidence,
        String modelVersion
) {
}
