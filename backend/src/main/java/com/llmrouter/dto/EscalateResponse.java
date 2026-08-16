package com.llmrouter.dto;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * POST /api/escalate 응답 (04 API Contract).
 *
 * 계약의 6개 필드에 tier / modelId / inputTokens / outputTokens 를 더했다.
 * 프론트 ModelBadge 가 토큰 수와 모델 ID 를 렌더하는데, 에스컬레이션 후에만
 * 그 자리가 비면 화면이 어그러진다. 추가 필드는 계약을 깨지 않는다.
 */
@Getter
@Builder
public class EscalateResponse {

    /**
     * <b>원본 chat 의 requestId</b>. 이 HTTP 요청의 X-Request-Id 가 아니다.
     *
     * RequestIdFilter 는 /api/escalate 에도 새 UUID 를 발급하지만,
     * 프론트가 같은 말풍선에서 sonnet → opus 로 한 번 더 올릴 수 있어야 하므로
     * 체인의 기준점인 원본 id 를 돌려준다. 두 id 가 갈리는 유일한 지점이다.
     */
    private String requestId;

    private String response;

    /** 라우터가 최초에 판정했던 티어의 모델명. 직전 단계가 아니라 체인의 출발점이다. */
    private String initialModel;

    /** 이번에 답한 모델 */
    private String finalModel;

    /** 누적 경로. 예: ["haiku", "sonnet", "opus"] */
    private List<String> escalationPath;

    private long latency;

    // ── 04 API Contract 대비 추가분 (프론트 배지 렌더용) ──

    /** "easy" | "medium" | "hard" */
    private String tier;

    /** 실제 호출한 Claude 모델 ID */
    private String modelId;

    private long inputTokens;
    private long outputTokens;
}
