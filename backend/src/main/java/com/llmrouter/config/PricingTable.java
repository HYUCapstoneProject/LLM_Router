package com.llmrouter.config;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * Claude 모델 단가표.
 *
 * 비용을 코드에 박지 않는 이유: 가격은 개정되고, 실험 결과는 어느 단가로 계산했는지
 * 함께 기록해야 재현이 된다. version 을 request 응답과 실험 결과에 같이 남긴다.
 */
@Component
public class PricingTable {

    @Getter
    private final String version;

    private final double haikuInput;
    private final double haikuOutput;
    private final double sonnetInput;
    private final double sonnetOutput;
    private final double opusInput;
    private final double opusOutput;

    public PricingTable(
            @Value("${llm.pricing.version}") String version,
            @Value("${llm.pricing.haiku.input}") double haikuInput,
            @Value("${llm.pricing.haiku.output}") double haikuOutput,
            @Value("${llm.pricing.sonnet.input}") double sonnetInput,
            @Value("${llm.pricing.sonnet.output}") double sonnetOutput,
            @Value("${llm.pricing.opus.input}") double opusInput,
            @Value("${llm.pricing.opus.output}") double opusOutput) {
        this.version = version;
        this.haikuInput = haikuInput;
        this.haikuOutput = haikuOutput;
        this.sonnetInput = sonnetInput;
        this.sonnetOutput = sonnetOutput;
        this.opusInput = opusInput;
        this.opusOutput = opusOutput;
    }

    /** 티어가 실제로 쓴 모델 단가로 계산한 USD */
    public double cost(String tier, long inputTokens, long outputTokens) {
        return switch (tier == null ? "" : tier) {
            case "easy" -> usd(inputTokens, outputTokens, haikuInput, haikuOutput);
            case "hard" -> usd(inputTokens, outputTokens, opusInput, opusOutput);
            default -> usd(inputTokens, outputTokens, sonnetInput, sonnetOutput);
        };
    }

    /**
     * 같은 토큰을 전부 Opus 로 보냈다면 얼마였을지.
     *
     * 주의: 실제로 Opus 에 물었을 때 출력 토큰이 같다는 보장은 없다.
     * 이 값은 "동일 토큰 기준 단가 차이"이지 실측 비교가 아니다.
     * 정확한 비교는 고정 실험에서 Opus 를 실제로 돌려 얻는다.
     */
    public double alwaysOpusCost(long inputTokens, long outputTokens) {
        return usd(inputTokens, outputTokens, opusInput, opusOutput);
    }

    private double usd(long inputTokens, long outputTokens, double inPrice, double outPrice) {
        return (inputTokens * inPrice + outputTokens * outPrice) / 1_000_000.0;
    }
}
