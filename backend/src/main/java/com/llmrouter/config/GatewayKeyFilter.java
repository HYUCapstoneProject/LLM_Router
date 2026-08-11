package com.llmrouter.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * 프론트 nginx 가 붙인 공유 시크릿을 검증한다.
 *
 * 왜 필요한가:
 *   데모는 누구나 접속할 수 있어야 하므로 Cloud Run 에 GCP 인증을 걸 수 없다.
 *   그런데 백엔드 뒤에는 Anthropic 크레딧이 있다. 인증 없이 열어두면
 *   누구나 /api/chat 을 반복 호출해 프로젝트 예산을 소진시킬 수 있다.
 *
 *   nginx 가 X-Gateway-Key 를 붙여 보내고 여기서 검증하면,
 *   브라우저에는 시크릿이 노출되지 않으면서 GCE 8080 직접 호출은 막힌다.
 *
 * 로컬 개발에서는 llm.gateway.key 를 비워두면 검증을 건너뛴다.
 * 배포 시에는 반드시 설정해야 한다.
 */
@Slf4j
@Order(2)   // RequestIdFilter(1) 다음. 거부 응답에도 requestId 를 담기 위함이다.
@Component
public class GatewayKeyFilter extends OncePerRequestFilter {

    private static final String HEADER = "X-Gateway-Key";

    @Value("${llm.gateway.key:}")
    private String expectedKey;

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        // 검증 대상은 /api/** 뿐이다. 헬스체크 등은 통과시킨다.
        return !request.getRequestURI().startsWith("/api/");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        if (expectedKey == null || expectedKey.isBlank()) {
            // 로컬 개발: 키 미설정이면 검증하지 않는다
            filterChain.doFilter(request, response);
            return;
        }

        String provided = request.getHeader(HEADER);
        if (!matches(provided)) {
            log.warn("게이트웨이 키 불일치 - 차단 uri={} ip={}",
                    request.getRequestURI(), request.getRemoteAddr());
            writeUnauthorized(response);
            return;
        }

        filterChain.doFilter(request, response);
    }

    /** 타이밍 공격을 피하기 위해 상수 시간 비교를 쓴다 */
    private boolean matches(String provided) {
        if (provided == null) {
            return false;
        }
        return MessageDigest.isEqual(
                provided.getBytes(StandardCharsets.UTF_8),
                expectedKey.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * 필터는 @RestControllerAdvice 를 타지 않으므로 직접 쓴다.
     * 형태는 04 API Contract 의 Error Format 과 맞춘다.
     */
    private void writeUnauthorized(HttpServletResponse response) throws IOException {
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.getWriter().write("""
                {"error":"허용되지 않은 요청입니다.","code":"GATEWAY_KEY_INVALID","requestId":"%s"}"""
                .formatted(String.valueOf(MDC.get(RequestIdFilter.REQUEST_ID))));
    }
}
