package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

/**
 * 질문 난이도를 easy / medium / hard 로 분류한다.
 *
 * 실제 분류는 FastAPI 의 POST /router/classify 가 한다 (팀원 담당).
 * 여기서는 클라이언트를 감싸고, 티어 → 모델명 매핑을 소유한다.
 *
 * 티어 → 모델 매핑의 소유자는 Gateway 다 (04 API Contract).
 * FastAPI 응답에도 model 필드가 있지만 참고용이며, 실험 중 매핑을 바꿔도
 * 라우터를 재배포하지 않는다.
 */
@Service
@RequiredArgsConstructor
public class RouterService {

    private final RouterClient routerClient;

    public enum Tier { EASY, MEDIUM, HARD }

    public Tier classify(String message) {
        return routerClient.classify(message);
    }

    /** 프론트 배지에 표시할 짧은 모델 이름 (기획서 3-2 Step 3) */
    public static String toModelName(String tier) {
        return switch (tier) {
            case "easy" -> "haiku";
            case "medium" -> "sonnet";
            case "hard" -> "opus";
            default -> tier;
        };
    }
}
