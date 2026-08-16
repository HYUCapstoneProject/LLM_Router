package com.llmrouter.client;

/** FastAPI 임베딩 응답. 모델 버전은 요청별 로그와 캐시 항목에 함께 저장한다. */
public record RouterEmbedding(float[] vector, String modelVersion) {
}
