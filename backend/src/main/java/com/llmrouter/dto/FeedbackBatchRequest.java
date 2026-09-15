package com.llmrouter.dto;

import java.time.Instant;
import java.util.List;

/**
 * FastAPI {@code POST /escalate} 요청 본문 (04 API Contract).
 *
 * <p><b>정확히 4필드다.</b> 계약 문서의 Java DTO 이름은 {@code EscalateRequest} 지만
 * 그 이름은 이미 사용자용 {@code POST /api/escalate} 요청이 쓰고 있다. 둘은 방향도
 * 소유자도 다르다 — 하나는 프론트가 게이트웨이에 보내는 것이고, 이것은 게이트웨이가
 * FastAPI 에 보내는 것이다. 같은 이름을 나눠 쓰면 import 하나 잘못 고른 순간
 * 컴파일은 통과하면서 엉뚱한 곳으로 나간다. 그래서 여기서만 이름을 달리한다.
 *
 * @param batchId       immutable 멱등성 키. 1~200자, 영문·숫자·점·밑줄·하이픈
 * @param schemaVersion 상수 {@link #SCHEMA_VERSION}
 * @param requestedAt   요청 생성 시각 (UTC)
 * @param events        1개 이상, 배열 안에서 eventId 고유
 */
public record FeedbackBatchRequest(
        String batchId,
        String schemaVersion,
        Instant requestedAt,
        List<EscalationEvent> events
) {

    public static final String SCHEMA_VERSION = "feedback-inline-v1";

    public FeedbackBatchRequest(String batchId, Instant requestedAt, List<EscalationEvent> events) {
        this(batchId, SCHEMA_VERSION, requestedAt, events);
    }
}
