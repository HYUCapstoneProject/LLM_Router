package com.llmrouter.dto;

/**
 * 리랭커 shadow 관측 결과. 전부 null 이면 호출하지 않았거나 실패한 것이다.
 *
 * @param wouldReject 리랭커를 실제로 적용했다면 이 캐시 히트를 막았을지.
 *                    shadow 평가의 핵심 값이다.
 */
public record RerankShadow(Double score, String modelVersion, Boolean wouldReject) {

    private static final RerankShadow SKIPPED = new RerankShadow(null, null, null);

    public static RerankShadow skipped() {
        return SKIPPED;
    }
}
