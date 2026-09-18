package com.llmrouter.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * FastAPI {@code POST /escalate} 의 202 응답 (04 API Contract).
 *
 * <p><b>202 는 학습 완료가 아니다.</b> GCS artifact 생성과 Vertex PipelineJob 제출이
 * 끝났다는 뜻일 뿐이라, 이 응답만 보고 "재학습 성공"으로 기록하면 거부·실패한 배치까지
 * 성공으로 집계된다. 실제 결과는 {@code statusUrl} 폴링으로만 확정한다.
 *
 * <p>응답에는 {@link JsonIgnoreProperties} 를 건다. 요청과 달리 응답은 FastAPI 가
 * 필드를 추가해도 우리가 깨질 이유가 없다 — 모르는 필드는 무시하고 아는 것만 쓴다.
 *
 * @param pipelineRunId Google resource name 이다. HTTP URL 이 아니므로
 *                      여기에 {@code https://} 를 붙이거나 직접 호출하지 않는다.
 * @param statusUrl     상대 경로. 문자열 결합이 아니라 URI builder 로 resolve 한다.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record EscalateAcceptance(
        String schemaVersion,
        String status,
        String batchId,
        String pipelineRunId,
        String logicalModelVersion,
        String candidateModelVersion,
        String trainingMode,
        String versionPolicyMode,
        String statusUrl
) {
}
