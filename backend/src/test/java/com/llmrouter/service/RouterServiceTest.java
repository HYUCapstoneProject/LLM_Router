package com.llmrouter.service;

import com.llmrouter.client.RouterClient;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class RouterServiceTest {
    @Test void delegatesClassification() {
        RouterClient client = mock(RouterClient.class);
        when(client.classify("q")).thenReturn(RouterService.Tier.HARD);
        assertThat(new RouterService(client).classify("q")).isEqualTo(RouterService.Tier.HARD);
    }

    @Test void mapsKnownAndUnknownTierNames() {
        assertThat(RouterService.toModelName("easy")).isEqualTo("haiku");
        assertThat(RouterService.toModelName("medium")).isEqualTo("sonnet");
        assertThat(RouterService.toModelName("hard")).isEqualTo("opus");
        assertThat(RouterService.toModelName("future")).isEqualTo("future");
    }
}
