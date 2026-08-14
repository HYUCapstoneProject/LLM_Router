package com.llmrouter.dto;

import lombok.Builder;

import java.util.List;

/**
 * LLM 을 부르기 <b>전에</b> escalation_log 에 꽂는 자리 예약.
 *
 * <p>순서를 이렇게 뒤집는 이유: 예약이 없으면 중복 요청이 걸러지는 시점이
 * INSERT(=LLM 호출 뒤)라 두 번째 요청도 이미 요금을 내고 나서 409 를 받는다.
 * Opus 응답이 1~3초 걸리는 동안이 정확히 사용자가 답답해서 또 누르는 창이다.
 *
 * <p>예약 행은 {@code response=NULL}, {@code user_satisfied=false} 다.
 * 두 유니크 인덱스 {@code (request_id, escalation_seq)} 와
 * {@code (request_id, final_tier)} 가 두 번째 요청을 여기서 막는다.
 *
 * <p>토큰·지연·응답·임베딩은 {@link EscalationCompletion} 에서 채운다.
 */
@Builder
public record EscalationReservation(
        String requestId,
        String question,
        String conversationId,
        String initialTier,
        List<String> escalationPath,
        String finalTier,
        int escalationSeq,
        String routerModelVersion,
        boolean wasCacheHit,
        Long cacheEntryId,
        Double similarity
) {}
