package com.llmrouter.controller;

import com.llmrouter.dto.EscalateRequest;
import com.llmrouter.dto.EscalateResponse;
import com.llmrouter.service.EscalationService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class EscalationController {

    private final EscalationService escalationService;

    @PostMapping("/escalate")
    public EscalateResponse escalate(@Valid @RequestBody EscalateRequest request) {
        return escalationService.escalate(request);
    }
}
