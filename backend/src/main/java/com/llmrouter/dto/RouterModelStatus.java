package com.llmrouter.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * FastAPI {@code GET /router/model/current} 응답 (04 API Contract).
 *
 * <p>승인 pointer({@code desired})와 실제 메모리에 적재된 모델({@code serving})을
 * <b>구분해서</b> 본다. 재학습이 성공해도 serving 이 아직 이전 모델이면 우리가 받는
 * 분류 결과는 옛날 모델의 것이다. 이 구분 없이 "재학습 완료"를 실험 로그에 적으면
 * 재학습 전후 비교가 어긋난다.
 *
 * @param status inSync | loading | loadFailed | static | unavailable
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RouterModelStatus(
        String status,
        ModelRef desired,
        ModelRef serving,
        ModelRef previousKnownGood
) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ModelRef(String modelVersion) {
    }

    /**
     * 승인 모델이 실제로 서빙되고 있는가.
     *
     * <p>{@code loadFailed} 여도 이전 known-good 모델이 정상 응답할 수 있다.
     * 즉 이 값이 false 라고 서비스가 죽은 것은 아니다.
     */
    public boolean isInSync() {
        return "inSync".equals(status)
                && desired != null && serving != null
                && desired.modelVersion() != null
                && desired.modelVersion().equals(serving.modelVersion());
    }

    public String servingModelVersion() {
        return serving == null ? null : serving.modelVersion();
    }
}
