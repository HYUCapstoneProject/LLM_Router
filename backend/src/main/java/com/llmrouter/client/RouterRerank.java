package com.llmrouter.client;

/**
 * POST /router/rerank 응답.
 *
 * @param scores       요청한 candidates 와 같은 순서·같은 길이.
 *                     점수 범위는 모델마다 다르다. sigmoid 모델은 0~1 이지만 raw logit
 *                     모델은 아니다. 임계값은 반드시 modelVersion 에 묶어 해석한다.
 * @param modelVersion request_log 에 남긴다. 이 값이 바뀌면 이전 점수와 비교하면 안 되고
 *                     임계값도 다시 측정해야 한다.
 */
public record RouterRerank(double[] scores, String modelVersion) {
}
