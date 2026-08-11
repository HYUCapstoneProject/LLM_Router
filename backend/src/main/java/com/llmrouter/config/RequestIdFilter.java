package com.llmrouter.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.UUID;

/**
 * 요청마다 requestId 를 만들어 MDC 에 넣는다.
 *
 * 이렇게 해두면 라우팅 → 캐시 조회 → LLM 호출 로그가 전부 같은 id 를 달고 나온다.
 * 나중에 FastAPI 까지 붙으면 요청 하나가 여러 서비스를 오가므로,
 * 그때 이 id 를 헤더로 전파하면 추적이 가능해진다.
 */
@Order(1)   // 가장 먼저. 이후 필터의 로그와 에러 응답이 requestId 를 쓸 수 있어야 한다.
@Component
public class RequestIdFilter extends OncePerRequestFilter {

    public static final String REQUEST_ID = "requestId";
    private static final String HEADER = "X-Request-Id";

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        String requestId = UUID.randomUUID().toString();
        MDC.put(REQUEST_ID, requestId);
        response.setHeader(HEADER, requestId);
        try {
            filterChain.doFilter(request, response);
        } finally {
            // 톰캣은 스레드를 재사용하므로 반드시 지운다. 안 지우면 다음 요청에 남는다.
            MDC.remove(REQUEST_ID);
        }
    }
}
