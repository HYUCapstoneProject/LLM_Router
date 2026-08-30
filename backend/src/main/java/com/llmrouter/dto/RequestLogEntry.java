package com.llmrouter.dto;

import lombok.Builder;

/**
 * request_log 한 행. 비용 그래프의 원천 데이터다.
 *
 * <p>파라미터 목록 대신 record 로 묶은 이유: 컬럼이 16개로 늘어나면서
 * 같은 타입(String)의 nullable 인자가 여럿 붙었다. 위치 인자로 두면
 * routerModelVersion 과 embedModelVersion 을 바꿔 넣어도 컴파일이 통과한다.
 *
 * @param question        사용자 질문 원문. 개인정보 마스킹은 하지 않는다 —
 *                        형식 기반 정규식으로는 이름·주소·조합 재식별을 못 잡아
 *                        반쪽짜리 보증을 하느니 안 하기로 했다(2026-08-13).
 *                        이 값은 재학습 배치로 나가므로 그 전제를 알고 있어야 한다.
 * @param cacheEntryId    이 요청이 히트했거나 새로 만든 캐시 행. 없으면 null.
 * @param parentRequestId 에스컬레이션이면 원본 chat 의 requestId, 아니면 null.
 * @param cacheMatchType  {@code exact} 또는 {@code semantic}. 캐시 미스면 null.
 *                        둘을 합친 히트율은 "같은 질문을 다시 쳤다"까지 포함하므로
 *                        시맨틱 캐시 성능으로 제시하면 과대평가가 된다.
 * @param rerankScore     리랭커 shadow 점수. 미호출·실패면 null.
 *                        점수는 rerankModelVersion 에서만 의미가 있다.
 * @param rerankWouldReject 리랭커를 적용했다면 이 히트를 막았을지. shadow 평가의 핵심.
 * @param origin          {@code chat} 또는 {@code escalate}.
 *                        에스컬레이션도 과금되는 LLM 호출이라 정상 행으로 남기고 여기서 구분한다.
 *                        비용 집계는 전체 행, 라우터 정확도 집계는 {@code origin='chat'} 만 본다.
 */
@Builder
public record RequestLogEntry(
        String requestId,
        String question,
        String conversationId,
        String tier,
        String modelId,
        String routerModelVersion,
        Double routerConfidence,
        String embedModelVersion,
        boolean cacheHit,
        Double similarity,
        Long cacheEntryId,
        long inputTokens,
        long outputTokens,
        long latencyMs,
        String fallbackReason,
        String parentRequestId,
        String origin,
        String cacheMatchType,
        Double rerankScore,
        String rerankModelVersion,
        Boolean rerankWouldReject
) {
    public static final String ORIGIN_CHAT = "chat";
    public static final String ORIGIN_ESCALATE = "escalate";

    /** 질문이 글자까지 같아서 임베딩 없이 히트한 경우 */
    public static final String MATCH_EXACT = "exact";
    /** 임베딩 유사도로 히트한 경우 */
    public static final String MATCH_SEMANTIC = "semantic";
}
