package com.llmrouter.dto;

import com.llmrouter.service.RouterService;

/**
 * 재학습 피드백 계약의 난이도 라벨 (04 API Contract).
 *
 * <p>상수를 소문자로 둔 것은 Java 관례에 어긋나지만 <b>계약 문서가 그렇게 정의</b>하고
 * 있고, 이러면 Jackson 기본 직렬화가 그대로 {@code "easy"} 를 만든다. 애노테이션으로
 * 표기를 바꾸는 방식은 애노테이션이 지워지는 순간 조용히 {@code "EASY"} 가 나가고
 * FastAPI 가 422 를 던진다. 표기 규칙을 타입 자체에 박아 두는 편이 안전하다.
 *
 * <p>{@link RouterService.Tier} 와 값은 같지만 역할이 다르다. 그쪽은 내부 도메인이고
 * 이쪽은 외부 계약이라, 내부 enum 에 상수를 하나 추가해도 외부 계약이 따라 움직이지 않는다.
 */
public enum RouterTier {

    easy, medium, hard;

    /** 내부 티어를 계약 표기로 옮긴다. */
    public static RouterTier from(RouterService.Tier tier) {
        return switch (tier) {
            case EASY -> easy;
            case MEDIUM -> medium;
            case HARD -> hard;
        };
    }

    /** DB 에 저장된 소문자 라벨에서 복원한다. 알 수 없는 값이면 예외다. */
    public static RouterTier fromLabel(String label) {
        if (label == null) {
            throw new IllegalArgumentException("tier 라벨이 비어 있습니다");
        }
        return valueOf(label.trim().toLowerCase(java.util.Locale.ROOT));
    }

    /** easy &lt; medium &lt; hard. 상향 여부 검증에 쓴다. */
    public int rank() {
        return ordinal();
    }
}
