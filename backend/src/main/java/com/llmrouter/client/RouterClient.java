package com.llmrouter.client;

import com.llmrouter.service.RouterService;

/**
 * FastAPI router serving 클라이언트 (04 API Contract).
 *
 * 실제 구현은 팀원의 FastAPI 가 뜬 뒤 HTTP 로 붙인다.
 * 그때까지는 {@link StubRouterClient} 가 자리를 채운다.
 */
public interface RouterClient {

    /**
     * POST /router/embed — cache 조회 전에 항상 호출된다.
     *
     * @return L2 정규화된 1024차원 벡터.
     *         정규화 전제가 깨지면 pgvector 의 `유사도 = 1 - 거리` 가 성립하지 않는다.
     */
    RouterEmbedding embed(String message);

    /** POST /router/classify — cache miss 일 때만 호출된다. */
    RouterClassification classify(String message);

    /** request_log 에 기록할 임베딩 모델 버전 */
    String embedModelVersion();

    /** request_log 에 기록할 라우터 모델 버전. 재학습 전후 비교에 필수. */
    String routerModelVersion();
}
