package com.llmrouter.controller;

import com.anthropic.errors.AnthropicIoException;
import com.anthropic.errors.AnthropicServiceException;
import com.anthropic.errors.BadRequestException;
import com.anthropic.errors.InternalServerException;
import com.anthropic.errors.RateLimitException;
import com.anthropic.errors.UnauthorizedException;
import com.llmrouter.client.MessageTooLongException;
import com.llmrouter.config.RequestIdFilter;
import com.llmrouter.dto.ErrorResponse;
import com.llmrouter.exception.EscalationConflictException;
import com.llmrouter.exception.EscalationInProgressException;
import com.llmrouter.exception.EscalationNotFoundException;
import com.llmrouter.exception.InvalidEscalationException;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 04 API Contract 의 에러 포맷으로 통일한다.
 *
 * 원칙: 사용자에게는 행동으로 이어지는 메시지를, 로그에는 원인 전부를 남긴다.
 * Anthropic 이 내려주는 원문 메시지를 그대로 사용자에게 노출하지 않는다
 * (내부 과금 상태나 계정 정보가 섞여 나올 수 있음).
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /** 요청 본문 검증 실패 (@NotBlank 등) */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException e) {
        String detail = e.getBindingResult().getFieldErrors().stream()
                .findFirst()
                .map(f -> f.getField() + ": " + f.getDefaultMessage())
                .orElse("잘못된 요청입니다.");

        log.warn("요청 검증 실패 - {}", detail);
        return build(HttpStatus.BAD_REQUEST, "INVALID_REQUEST", detail);
    }

    /**
     * 질문이 512 토큰을 넘는다. 캐시 임베딩 모델(BGE)의 입력 한도라
     * 자르지 않고 거절해야 캐시와 라우터가 같은 질문 전체를 본다.
     */
    @ExceptionHandler(MessageTooLongException.class)
    public ResponseEntity<ErrorResponse> handleMessageTooLong(MessageTooLongException e) {
        log.warn("입력 토큰 한도 초과 - {}", e.getMessage());
        return build(HttpStatus.BAD_REQUEST, "MESSAGE_TOO_LONG",
                "질문이 너무 깁니다. 512 토큰 이하로 줄여서 다시 보내주세요.");
    }

    /** /api/escalate 가 참조한 원본 요청이 request_log 에 없다 */
    @ExceptionHandler(EscalationNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleEscalationNotFound(EscalationNotFoundException e) {
        log.warn("에스컬레이션 대상 없음 - {}", e.getMessage());
        return build(HttpStatus.NOT_FOUND, "ESCALATION_TARGET_NOT_FOUND",
                "원본 요청을 찾을 수 없습니다. 질문을 다시 보내주세요.");
    }

    /**
     * 모르는 모델 이름이거나 더 낮은 티어로 내려가려는 요청.
     *
     * 예외 메시지를 그대로 내보낸다. "gpt-4 는 모르는 모델"과 "opus 에서 sonnet 으로는 못 간다"는
     * 고쳐야 할 것이 다른데, 한 문구로 뭉개면 RouterService.toTier 가 예외 대신
     * Optional 을 돌려주며 둘을 나눈 이유가 없어진다.
     * 내용은 요청에 실려 온 값과 허용 목록뿐이라 노출해도 안전하다.
     */
    @ExceptionHandler(InvalidEscalationException.class)
    public ResponseEntity<ErrorResponse> handleInvalidEscalation(InvalidEscalationException e) {
        log.warn("잘못된 에스컬레이션 요청 - {}", e.getMessage());
        return build(HttpStatus.BAD_REQUEST, "INVALID_ESCALATION", e.getMessage());
    }

    /**
     * 클라이언트가 낡은 화면을 보고 보낸 요청.
     *
     * "새로고침하라"고 안내하지 않는다. 대화 상태가 메모리에만 있어(useChat)
     * 새로고침하면 대화가 통째로 날아간다.
     */
    @ExceptionHandler(EscalationConflictException.class)
    public ResponseEntity<ErrorResponse> handleEscalationConflict(EscalationConflictException e) {
        log.warn("에스컬레이션 충돌 - {}", e.getMessage());
        return build(HttpStatus.CONFLICT, "ESCALATION_ALREADY_APPLIED",
                "이미 상위 모델로 답변한 요청입니다.");
    }

    /** 같은 요청이 아직 처리 중 — 기다리면 되는 상황이라 위와 안내가 달라야 한다 */
    @ExceptionHandler(EscalationInProgressException.class)
    public ResponseEntity<ErrorResponse> handleEscalationInProgress(EscalationInProgressException e) {
        log.warn("에스컬레이션 처리 중 중복 요청 - {}", e.getMessage());
        return build(HttpStatus.CONFLICT, "ESCALATION_IN_PROGRESS",
                "같은 요청을 처리하고 있습니다. 잠시 후 다시 시도해주세요.");
    }

    /** Claude API rate limit */
    @ExceptionHandler(RateLimitException.class)
    public ResponseEntity<ErrorResponse> handleRateLimit(RateLimitException e) {
        log.warn("Claude API rate limit", e);
        return build(HttpStatus.TOO_MANY_REQUESTS, "LLM_RATE_LIMITED",
                "요청이 몰리고 있습니다. 잠시 후 다시 시도해주세요.");
    }

    /** API 키 문제 — 사용자 잘못이 아니라 서버 설정 문제다 */
    @ExceptionHandler(UnauthorizedException.class)
    public ResponseEntity<ErrorResponse> handleUnauthorized(UnauthorizedException e) {
        log.error("Claude API 인증 실패 - ANTHROPIC_API_KEY 를 확인하세요", e);
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "LLM_AUTH_FAILED",
                "서버 설정 오류로 응답할 수 없습니다.");
    }

    /** Anthropic 5xx */
    @ExceptionHandler(InternalServerException.class)
    public ResponseEntity<ErrorResponse> handleUpstream5xx(InternalServerException e) {
        log.error("Claude API 서버 오류", e);
        return build(HttpStatus.SERVICE_UNAVAILABLE, "LLM_UNAVAILABLE",
                "LLM 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.");
    }

    /** 네트워크 단절 등 응답 자체를 못 받은 경우 */
    @ExceptionHandler(AnthropicIoException.class)
    public ResponseEntity<ErrorResponse> handleIo(AnthropicIoException e) {
        log.error("Claude API 연결 실패", e);
        return build(HttpStatus.SERVICE_UNAVAILABLE, "LLM_UNAVAILABLE",
                "LLM 서비스에 연결할 수 없습니다.");
    }

    /** 400 — 크레딧 부족이 여기로 들어온다 */
    @ExceptionHandler(BadRequestException.class)
    public ResponseEntity<ErrorResponse> handleBadRequest(BadRequestException e) {
        String message = String.valueOf(e.getMessage());

        // Anthropic 은 크레딧 부족도 400/invalid_request_error 로 내려준다.
        // 타입만으로는 구분이 안 돼 메시지를 본다. 문구가 바뀌면 아래 분기는 깨지고
        // LLM_ERROR 로 떨어지므로, 동작이 멈추지는 않는다.
        if (message.contains("credit balance")) {
            log.error("Claude API 크레딧 부족 - Console 의 Plans & Billing 에서 충전 필요", e);
            return build(HttpStatus.PAYMENT_REQUIRED, "LLM_QUOTA_EXCEEDED",
                    "LLM 사용 한도가 소진되었습니다. 관리자에게 문의해주세요.");
        }

        log.error("Claude API 요청 오류 type={}", e.errorType().orElse(null), e);
        return build(HttpStatus.BAD_GATEWAY, "LLM_ERROR", "LLM 요청 처리에 실패했습니다.");
    }

    /** 그 외 Anthropic 오류 (403, 404, 422 등) */
    @ExceptionHandler(AnthropicServiceException.class)
    public ResponseEntity<ErrorResponse> handleAnthropic(AnthropicServiceException e) {
        log.error("Claude API 오류 type={}", e.errorType().orElse(null), e);
        return build(HttpStatus.BAD_GATEWAY, "LLM_ERROR", "LLM 요청 처리에 실패했습니다.");
    }

    /** 최후의 방어선 */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleUnexpected(Exception e) {
        log.error("처리되지 않은 예외", e);
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR",
                "일시적인 오류가 발생했습니다.");
    }

    private ResponseEntity<ErrorResponse> build(HttpStatus status, String code, String message) {
        return ResponseEntity.status(status).body(
                ErrorResponse.builder()
                        .error(message)
                        .code(code)
                        .requestId(MDC.get(RequestIdFilter.REQUEST_ID))
                        .build());
    }
}
