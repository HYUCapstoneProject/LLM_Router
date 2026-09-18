package com.llmrouter.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.time.Clock;

/**
 * 재학습 피드백 경로의 스케줄링 설정.
 *
 * <p>{@link EnableScheduling} 은 여기서 처음 켠다. 지금까지 이 서비스에는 주기 작업이
 * 없었다 — 모든 처리가 사용자 요청에 붙어 있었다. 재학습 제출과 상태 폴링은 요청과
 * 무관하게 돌아야 하는 첫 작업이다.
 */
@Configuration
@EnableScheduling
public class RetrainingConfig {

    /**
     * 시각 생성을 주입 가능하게 만든다.
     *
     * <p>batchId 에 UTC 날짜가 들어가므로({@code feedback-20260915-0001}) 테스트가
     * 시각을 고정할 수 있어야 한다. {@code Instant.now()} 를 직접 부르면 날짜 경계에서만
     * 깨지는 테스트가 되는데, 그건 하필 자정 근처에 돌릴 때만 빨개진다.
     */
    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }
}
