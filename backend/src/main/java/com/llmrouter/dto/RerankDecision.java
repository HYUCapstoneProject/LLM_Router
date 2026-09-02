package com.llmrouter.dto;

/**
 * 리랭커가 캐시 후보에 대해 내린 판단.
 *
 * <p>세 모드를 같은 타입으로 표현한다.
 * <ul>
 *   <li>{@code off} — 리랭커를 부르지 않는다. accepted 는 코사인 1등</li>
 *   <li>{@code shadow} — 점수만 기록하고 accepted 는 코사인 1등 그대로</li>
 *   <li>{@code enforce} — 리랭커 판정으로 accepted 를 정한다. 통과가 없으면 null</li>
 * </ul>
 *
 * @param accepted       실제로 응답에 쓸 캐시 항목. <b>null 이면 캐시 미스로 처리한다.</b>
 * @param score          accepted(또는 코사인 1등)의 리랭커 점수. 미호출·실패면 null
 * @param wouldReject    리랭커 기준으로 거부 대상이었는지.
 *                       shadow 에서는 "적용했다면 막았을까", enforce 에서는 실제 거부 여부
 * @param candidateCount 점수를 매긴 후보 수. 후보가 1개면 재정렬은 정의상 일어나지 않는다
 * @param wouldReorder   리랭커 1등이 코사인 1등과 달랐는지
 * @param reason         판정 사유. request_log 로 나가지 않고 로그에만 쓴다
 */
public record RerankDecision(
        CachedResponse accepted,
        Double score,
        String modelVersion,
        Boolean wouldReject,
        Integer candidateCount,
        Boolean wouldReorder,
        String reason
) {
    public static RerankDecision passthrough(CachedResponse accepted, String reason) {
        return new RerankDecision(accepted, null, null, null, null, null, reason);
    }
}
