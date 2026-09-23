package com.llmrouter.dto;

/**
 * 에스컬레이션의 근거가 되는 원본 요청의 맥락. request_log 에서 읽는다.
 *
 * 클라이언트가 보낸 값을 쓰지 않고 굳이 DB 를 다시 읽는 이유:
 * /api/escalate 의 존재 가치는 <b>신뢰할 수 있는</b> 정정 라벨이다.
 * 클라이언트가 주장하는 initial_tier 를 받아주면 공개 엔드포인트로
 * 재학습 데이터를 오염시킬 수 있다.
 *
 * @param tier        라우터의 최초 판정. 체인의 initial_tier 가 된다.
 * @param cacheHit    원본 답이 캐시에서 나왔는가. true 면 라우터의 과소 라우팅이 아니라
 *                    캐시 false-hit 증거다 (Decision Log 2026-08-03).
 * @param similarity  그때 매칭된 유사도. 캐시 갱신 여부 판단에 쓴다.
 */
public record EscalationContext(
        String question,
        String conversationId,
        String tier,
        boolean cacheHit,
        Long cacheEntryId,
        Double similarity,
        String routerModelVersion,
        String embedModelVersion
) {}
