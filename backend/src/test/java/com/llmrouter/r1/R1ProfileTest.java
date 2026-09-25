package com.llmrouter.r1;

import com.llmrouter.client.RetrainingClient;
import com.llmrouter.repository.FeedbackBatchRepository;
import com.llmrouter.service.FeedbackBatchService;
import com.llmrouter.service.R1ObservationService;
import com.llmrouter.service.RetrainingScheduler;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.time.Clock;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

class R1ProfileTest {
    @Test
    void actualR1ProfilePreservesClassifyV1AndBlocksLegacyManualSubmission() {
        var repository = mock(FeedbackBatchRepository.class);
        var client = mock(RetrainingClient.class);
        new ApplicationContextRunner()
                .withInitializer(new ConfigDataApplicationContextInitializer())
                .withPropertyValues("spring.profiles.active=r1")
                .withBean(FeedbackBatchRepository.class, () -> repository)
                .withBean(RetrainingClient.class, () -> client)
                .withBean(Clock.class, Clock::systemUTC)
                .withUserConfiguration(FeedbackBatchService.class, R1ObservationService.class, RetrainingScheduler.class)
                .run(context -> {
                    assertThat(context).hasNotFailed().doesNotHaveBean(R1ObservationService.class);
                    assertThat(context.getEnvironment().getProperty("llm.router.classify-contract")).isEqualTo("v1");
                    assertThat(context.getEnvironment().getProperty("llm.retraining.auto-submit")).isEqualTo("false");
                    var service = context.getBean(FeedbackBatchService.class);
                    assertThatThrownBy(() -> service.submitPending("manual"))
                            .isInstanceOf(IllegalStateException.class).hasMessageContaining("R1 profile disables");
                    assertThatThrownBy(() -> service.retrySubmission("old-batch"))
                            .isInstanceOf(IllegalStateException.class).hasMessageContaining("R1 profile disables");
                    var scheduler = context.getBean(RetrainingScheduler.class);
                    scheduler.autoSubmit();
                    scheduler.retryUnaccepted();
                    verifyNoInteractions(repository, client);
                });
    }
}
