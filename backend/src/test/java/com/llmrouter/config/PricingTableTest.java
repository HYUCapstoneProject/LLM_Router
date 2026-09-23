package com.llmrouter.config;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class PricingTableTest {

    /** application.yml 과 같은 값 (100만 토큰당 USD) */
    private final PricingTable pricing =
            new PricingTable("test", 1.0, 5.0, 3.0, 15.0, 5.0, 25.0);

    @Test
    void tierPicksTheModelThatActuallyServedTheRequest() {
        assertThat(pricing.cost("easy", 1_000_000, 1_000_000)).isEqualTo(6.0);
        assertThat(pricing.cost("medium", 1_000_000, 1_000_000)).isEqualTo(18.0);
        assertThat(pricing.cost("hard", 1_000_000, 1_000_000)).isEqualTo(30.0);
    }

    /**
     * tier 가 비어 있으면 medium 으로 본다.
     *
     * 라우터가 실패하면 ChatService 가 MEDIUM 으로 폴백하므로, 값이 없을 때
     * 가장 싼 haiku 로 계산하면 실제보다 절감률이 좋아 보인다. 그쪽으로 틀리지 않게 한다.
     */
    @Test
    void unknownTierIsPricedAsMediumNotAsTheCheapest() {
        assertThat(pricing.cost(null, 1_000_000, 0)).isEqualTo(3.0);
        assertThat(pricing.cost("unexpected", 1_000_000, 0)).isEqualTo(3.0);
    }

    @Test
    void alwaysOpusUsesOpusPricesRegardlessOfWhatRan() {
        assertThat(pricing.alwaysOpusCost(2_000, 1_000))
                .isCloseTo(2_000 * 5.0 / 1e6 + 1_000 * 25.0 / 1e6, within(1e-12));
    }
}
