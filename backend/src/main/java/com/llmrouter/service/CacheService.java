package com.llmrouter.service;

import org.springframework.stereotype.Service;

import java.util.Optional;

/**
 * pgvector 기반 시맨틱 캐싱 레이어.
 *
 * 구현 계획 (7-8월):
 *   - bge-large-en-v1.5 로 질문을 임베딩
 *   - pgvector ANN 검색으로 코사인 유사도 >= 0.8 이면 캐시 히트
 *   - 캐시 미스 시 LLM 응답을 저장
 *
 * 현재: stub — 항상 캐시 미스
 */
@Service
public class CacheService {

    /**
     * 유사 질문에 대한 캐시된 응답을 반환한다.
     * @return 캐시 히트 시 응답 문자열, 미스 시 empty
     */
    public Optional<String> get(String message) {
        // TODO: pgvector 연동
        return Optional.empty();
    }

    /**
     * LLM 응답을 캐시에 저장한다.
     */
    public void put(String message, String response) {
        // TODO: 임베딩 생성 후 pgvector 저장
    }
}
