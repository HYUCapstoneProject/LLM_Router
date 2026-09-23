package com.llmrouter.service;

import java.util.Locale;

/** 리랭커를 캐시 판정에 어떻게 쓸지. */
public enum RerankMode {
    /** 부르지 않는다. 코사인 임계값만으로 판정한다. */
    OFF,
    /** 점수만 기록한다. 캐시 판정은 바꾸지 않는다. */
    SHADOW,
    /** 판정에 실제로 쓴다. 통과하지 못하면 캐시 미스로 처리한다. */
    ENFORCE;

    static RerankMode parse(String value) {
        try {
            return valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException | NullPointerException error) {
            // 오타가 조용히 OFF 로 떨어지면 리랭커가 안 도는 걸 아무도 모른다.
            throw new IllegalArgumentException(
                    "llm.cache.rerank.mode 는 off|shadow|enforce 여야 한다: " + value, error);
        }
    }
}
