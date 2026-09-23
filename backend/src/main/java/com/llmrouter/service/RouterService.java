package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClassification;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Optional;

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

    /**
     * rank 를 명시적으로 들고 있는 이유: 에스컬레이션은 "더 높은 티어로만" 올라갈 수 있는데
     * ordinal() 에 기대면 누가 상수 순서를 바꾸는 순간 그 규칙이 조용히 뒤집힌다.
     */
    public enum Tier {
        EASY(0), MEDIUM(1), HARD(2);

        private final int rank;

        Tier(int rank) {
            this.rank = rank;
        }

        public int rank() {
            return rank;
        }

        /** DB·API 에서 쓰는 소문자 표기 (easy | medium | hard) */
        public String label() {
            return name().toLowerCase();
        }
    }

    public RouterClassification classify(String message) {
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

    /**
     * toModelName 의 역함수. /api/escalate 가 받는 currentModel / targetModel 을 해석한다.
     *
     * 모르는 이름에 예외를 던지지 않고 Optional 을 주는 이유:
     * "haiku|sonnet|opus 가 아님"과 "내려가는 방향임"은 둘 다 400 이지만
     * 사용자에게 줄 메시지가 다르다. 판단은 호출자가 한다.
     */
    public static Optional<Tier> toTier(String modelName) {
        return switch (modelName) {
            case "haiku" -> Optional.of(Tier.EASY);
            case "sonnet" -> Optional.of(Tier.MEDIUM);
            case "opus" -> Optional.of(Tier.HARD);
            default -> Optional.empty();
        };
    }
}
