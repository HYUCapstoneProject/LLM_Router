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

    /**
     * POST /router/rerank — 캐시 후보를 cross-encoder 로 재검증한다.
     *
     * <p>bi-encoder 는 두 문장을 각각 따로 압축하므로 Python 이 Java 로 바뀌어도
     * 벡터가 거의 안 움직인다. 실측에서 paraphrase 최소 0.8621 < hard_negative 최대
     * 0.9371 로 두 분포가 겹쳐 어떤 코사인 임계값도 둘을 가르지 못했다.
     *
     * @param candidates 순서가 계약이다. 응답 scores 를 인덱스로 되돌려 매핑한다.
     * @throws RouterClientException 리랭커가 없거나(503) 응답이 계약과 다를 때.
     *                               호출자는 이를 캐시 미스로 처리한다(fail-closed).
     */
    RouterRerank rerank(String query, java.util.List<String> candidates);

    /** request_log 에 기록할 임베딩 모델 버전 */
    String embedModelVersion();

    /** request_log 에 기록할 라우터 모델 버전. 재학습 전후 비교에 필수. */
    String routerModelVersion();
}
