package com.llmrouter.service;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class QuestionHashTest {

    @Test
    void sameQuestionGivesSameHash() {
        assertThat(QuestionHash.of("What is the capital of Australia?"))
                .isEqualTo(QuestionHash.of("What is the capital of Australia?"))
                .hasSize(64);
    }

    /** 사용자가 같은 질문을 다시 칠 때 실제로 달라지는 건 대개 공백뿐이다. */
    @Test
    void whitespaceDifferencesAreNormalizedAway() {
        String base = QuestionHash.of("What is the capital of Australia?");
        assertThat(QuestionHash.of("  What is the capital of Australia?  ")).isEqualTo(base);
        assertThat(QuestionHash.of("What  is\tthe\ncapital of Australia?")).isEqualTo(base);
    }

    /**
     * 대소문자는 정규화하지 않는다.
     *
     * exact match 의 존재 이유가 "오탐 0" 이라 문자열 동일성 이상으로 넓히지 않는다.
     * 대소문자만 다른 질문은 임베딩 경로로 넘어가 0.99 근처에서 히트한다.
     */
    @Test
    void caseIsNotNormalized() {
        assertThat(QuestionHash.of("What does HTTP stand for?"))
                .isNotEqualTo(QuestionHash.of("what does http stand for?"));
    }

    @Test
    void differentQuestionsGiveDifferentHashes() {
        assertThat(QuestionHash.of("Write a Python function to reverse a list."))
                .isNotEqualTo(QuestionHash.of("Write a Java function to reverse a list."));
    }

    @Test
    void nullIsTreatedAsEmptyInsteadOfThrowing() {
        assertThat(QuestionHash.of(null)).isEqualTo(QuestionHash.of(""));
    }
}
