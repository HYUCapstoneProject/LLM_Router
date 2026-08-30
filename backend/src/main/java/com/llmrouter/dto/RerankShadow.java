package com.llmrouter.dto;

/**
 * 리랭커 shadow 관측 결과. 전부 null 이면 호출하지 않았거나 실패한 것이다.
 *
 * @param score          <b>실제로 응답에 쓴</b> 후보의 점수. 다른 후보가 더 높았더라도
 *                       shadow 는 동작을 바꾸지 않으므로 서빙된 쪽을 기준으로 남긴다.
 * @param wouldReject    리랭커를 적용했다면 그 후보를 막았을지 (AND 조건 기준).
 * @param candidateCount 리랭커가 점수를 매긴 후보 수. wouldReorder 해석에 필요하다 —
 *                       후보가 1개면 재정렬은 정의상 일어날 수 없다.
 * @param wouldReorder   리랭커 1등이 코사인 1등과 달랐는지. 재정렬의 값어치를 재는 값이다.
 */
public record RerankShadow(
        Double score,
        String modelVersion,
        Boolean wouldReject,
        Integer candidateCount,
        Boolean wouldReorder
) {

    private static final RerankShadow SKIPPED = new RerankShadow(null, null, null, null, null);

    public static RerankShadow skipped() {
        return SKIPPED;
    }
}
