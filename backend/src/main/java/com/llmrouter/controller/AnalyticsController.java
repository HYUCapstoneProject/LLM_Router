package com.llmrouter.controller;

import com.llmrouter.dto.AnalyticsSummary;
import com.llmrouter.service.AnalyticsService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class AnalyticsController {

    private final AnalyticsService analyticsService;

    /** period: today | 7d | all */
    @GetMapping("/analytics/summary")
    public AnalyticsSummary summary(@RequestParam(defaultValue = "all") String period) {
        return analyticsService.summary(period);
    }
}
