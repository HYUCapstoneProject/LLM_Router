package com.llmrouter.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * POST /api/escalate 요청 (04 API Contract).
 *
 * currentModel / targetModel 은 haiku | sonnet | opus 중 하나이며,
 * 반드시 더 높은 티어로만 올라갈 수 있다. 검증은 EscalationService 가 한다 —
 * "모르는 모델 이름"과 "내려가는 방향"은 둘 다 400 이지만 메시지가 달라야 한다.
 */
@Getter
@NoArgsConstructor
public class EscalateRequest {

    /**
     * /api/chat 응답으로 받은 requestId.
     *
     * UUID 형식을 여기서 막는 이유: request_log.request_id 가 UUID 타입이라
     * 형식이 깨진 문자열이 `?::uuid` 에 닿으면 DataAccessException 이 되고
     * 최후 방어선의 500 INTERNAL_ERROR 로 떨어진다. 마땅히 나가야 할 400 이 안 나간다.
     */
    @NotBlank
    @Pattern(
            regexp = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            message = "requestId 는 UUID 형식이어야 합니다."
    )
    private String requestId;

    /**
     * 원본 질문.
     *
     * 서버는 이 값을 LLM 에 보내지 않는다. request_log 에 저장된 질문을 쓴다.
     * 클라이언트가 보낸 문장을 신뢰하면 공개 엔드포인트로 재학습 라벨을 오염시킬 수 있다.
     * 계약상 필수 필드라 받기만 하고, 실제 용도는 로그 대조뿐이다.
     */
    @NotBlank
    private String message;

    /** haiku | sonnet — 지금 화면에 떠 있는 답변의 모델 */
    @NotBlank
    private String currentModel;

    /** sonnet | opus — 올라갈 모델 */
    @NotBlank
    private String targetModel;
}
