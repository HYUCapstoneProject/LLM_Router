package com.llmrouter.client;

import com.llmrouter.service.RouterService;
import org.springframework.stereotype.Component;

import java.util.Random;

/**
 * FastAPI 가 뜨기 전까지 쓰는 임시 구현.
 *
 * ⚠️ 임베딩이 가짜다. 질문 문자열의 해시를 seed 로 한 의사난수 벡터를 만든다.
 *
 *   - 같은 질문 → 같은 벡터  ⇒ 완전히 동일한 질문은 캐시 히트한다.
 *     DB 연결·조회·저장 경로를 지금 검증할 수 있다.
 *   - 의미가 비슷한 다른 질문 → 무관한 벡터  ⇒ 히트하지 않는다.
 *     즉 시맨틱 캐싱 자체는 이걸로 검증되지 않는다.
 *
 * 캐시 히트율이나 임계값을 이 구현으로 측정하면 안 된다.
 * bge-large 가 붙기 전까지 나오는 숫자는 전부 무의미하다.
 */
@Component
public class StubRouterClient implements RouterClient {

    private static final int DIM = 1024;

    @Override
    public float[] embed(String message) {
        Random rnd = new Random(message.hashCode());

        float[] v = new float[DIM];
        double norm = 0.0;
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) rnd.nextGaussian();
            norm += (double) v[i] * v[i];
        }

        // L2 정규화 — 실제 FastAPI 도 정규화해서 준다는 계약이므로 형태를 맞춘다
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < DIM; i++) {
                v[i] /= (float) norm;
            }
        }
        return v;
    }

    @Override
    public RouterService.Tier classify(String message) {
        // HARD 가 아니라 MEDIUM 인 이유:
        // HARD 고정이면 전 요청이 Opus 로 나가 비용 측정이 왜곡된다.
        return RouterService.Tier.MEDIUM;
    }

    @Override
    public String embedModelVersion() {
        return "stub-hash-v0";
    }

    @Override
    public String routerModelVersion() {
        return "stub-medium-v0";
    }
}
