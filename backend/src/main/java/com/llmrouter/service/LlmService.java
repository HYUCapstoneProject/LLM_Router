package com.llmrouter.service;

import com.anthropic.client.AnthropicClient;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.llmrouter.dto.LlmResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.stream.Collectors;

/**
 * Anthropic Claude API 호출 레이어.
 *
 * 티어 → 모델 매핑은 application.yml 에서 주입받는다.
 * 실험 중 모델을 바꿔가며 비용/정확도를 재려면 재컴파일 없이 바꿀 수 있어야 한다.
 *
 * thinking 파라미터는 의도적으로 설정하지 않는다.
 * 3티어 비용 비교가 목적인데 티어마다 thinking 설정이 다르면
 * 토큰 수가 티어 자체의 특성인지 thinking 때문인지 분리되지 않는다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LlmService {

    private final AnthropicClient anthropicClient;

    @Value("${llm.models.easy}")
    private String easyModel;

    @Value("${llm.models.medium}")
    private String mediumModel;

    @Value("${llm.models.hard}")
    private String hardModel;

    @Value("${llm.max-tokens}")
    private long maxTokens;

    public LlmResult call(RouterService.Tier tier, String message) {
        String modelId = resolveModelId(tier);

        MessageCreateParams params = MessageCreateParams.builder()
                .model(modelId)
                .maxTokens(maxTokens)
                .addUserMessage(message)
                .build();

        // 예외는 잡지 않는다. GlobalExceptionHandler 가 타입별로 처리하며,
        // 여기서 또 잡아 로그를 남기면 같은 오류가 두 번 찍힌다.
        // 실패 시에도 이 줄이 남아 있어 어느 모델에서 터졌는지 알 수 있다.
        log.info("LLM 호출 시작 tier={} model={}", tier, modelId);

        Message response = anthropicClient.messages().create(params);

        String text = response.content().stream()
                .flatMap(block -> block.text().stream())
                .map(textBlock -> textBlock.text())
                .collect(Collectors.joining());

        long inputTokens = response.usage().inputTokens();
        long outputTokens = response.usage().outputTokens();

        log.info("LLM 호출 완료 tier={} model={} in={} out={}",
                tier, modelId, inputTokens, outputTokens);

        return new LlmResult(text, tier.name().toLowerCase(), modelId, inputTokens, outputTokens);
    }

    private String resolveModelId(RouterService.Tier tier) {
        return switch (tier) {
            case EASY -> easyModel;
            case MEDIUM -> mediumModel;
            case HARD -> hardModel;
        };
    }
}
