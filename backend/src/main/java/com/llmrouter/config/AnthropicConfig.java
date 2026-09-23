package com.llmrouter.config;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AnthropicConfig {

    /**
     * ANTHROPIC_API_KEY 환경변수에서 키를 읽는다.
     * 키를 코드나 application.yml 에 적지 않는다 (팀 규칙: secret commit 금지).
     */
    @Bean
    public AnthropicClient anthropicClient() {
        return AnthropicOkHttpClient.fromEnv();
    }
}
