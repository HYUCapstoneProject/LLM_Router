package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import com.llmrouter.client.RouterClassification;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class RouterServiceTest {
    @Test void delegatesClassification() {
        RouterClient client = mock(RouterClient.class);
        RouterClassification result = new RouterClassification(RouterService.Tier.HARD, 0.91, "router-v1");
        when(client.classify("q")).thenReturn(result);
        assertThat(new RouterService(client).classify("q")).isSameAs(result);
    }

    @Test void mapsKnownAndUnknownTierNames() {
        assertThat(RouterService.toModelName("easy")).isEqualTo("haiku");
        assertThat(RouterService.toModelName("medium")).isEqualTo("sonnet");
        assertThat(RouterService.toModelName("hard")).isEqualTo("opus");
        assertThat(RouterService.toModelName("future")).isEqualTo("future");
    }
}
