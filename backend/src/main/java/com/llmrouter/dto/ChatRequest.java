package com.llmrouter.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class ChatRequest {

    @NotBlank
    private String message;

    @NotBlank
    private String conversationId;
}
