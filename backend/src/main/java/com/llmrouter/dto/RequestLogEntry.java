package com.llmrouter.dto;

import lombok.Builder;

/**
 * request_log 한 행. 비용 그래프의 원천 데이터다.
 *
 * <p>파라미터 목록 대신 record 로 묶은 이유: 컬럼이 16개로 늘어나면서
 * 같은 타입(String)의 nullable 인자가 여럿 붙었다. 위치 인자로 두면
 * routerModelVersion 과 embedModelVersion 을 바꿔 넣어도 컴파일이 통과한다.
 *
 * @param question        <b>마스킹된</b> 질문. 원문은 저장하지 않는다.
 * @param cacheEntryId    이 요청이 히트했거나 새로 만든 캐시 행. 없으면 null.
 * @param parentRequestId 에스컬레이션이면 원본 chat 의 requestId, 아니면 null.
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
        String embedModelVersion,
        boolean cacheHit,
        Double similarity,
        Long cacheEntryId,
        long inputTokens,
        long outputTokens,
        long latencyMs,
        String fallbackReason,
        String parentRequestId,
        String origin
) {
    public static final String ORIGIN_CHAT = "chat";
    public static final String ORIGIN_ESCALATE = "escalate";
}
