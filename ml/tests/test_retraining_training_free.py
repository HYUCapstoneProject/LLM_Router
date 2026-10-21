"""Contract tests for the F6 training-free score update; synthetic rows only."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "notebooks/retraining_effect"
sys.path.insert(0, str(HELPERS))
spec = importlib.util.spec_from_file_location("retraining_training_free", HELPERS / "training_free.py")
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)


def stream(n=3000, seed=0, bias=1.0):
    """Needed tier follows an ordinal latent; frozen logits see it but are shifted toward easy."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    need = np.digitize(latent + rng.logistic(scale=.3, size=n), [-.3, .9])
    noisy = latent + rng.normal(scale=.3, size=n)
    logits = np.column_stack([np.zeros(n), noisy - bias, 2 * noisy - 2 * bias - 1])
    data = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "duplicate_group_id": [f"g{i}" for i in range(n)],
                         "source_dataset": np.where(np.arange(n) % 2, "a", "b"),
                         "difficulty_label": np.asarray(tf.LABELS)[need], "window": np.repeat(np.arange(1, 7), n // 6)})
    for i, name in enumerate(tf.LABELS):
        data[f"logit_{name}"] = logits[:, i]
    for i, (tier, cost) in enumerate(zip(("low", "medium", "high"), (.001, .003, .005))):
        data[f"success_{tier}"] = (need <= i).astype(float)
        data[f"quality_{tier}"] = data[f"success_{tier}"]
        data[f"actual_cost_{tier}"] = cost
    data["initial_label"] = np.asarray(tf.LABELS)[logits.argmax(1)]
    return data, need


class TrainingFreeContracts(unittest.TestCase):
    def test_observations_are_what_the_service_sees(self):
        sim = pd.DataFrame({"initial_tier": [0, 0, 1, 1, 2, 0], "resolved_tier": [0, 1, 1, 2, -1, -1]})
        lower, upper = tf.observations(sim)
        self.assertEqual(lower.tolist(), [0, 1, 0, 2, 0, 2])
        self.assertEqual(upper.tolist(), [0, 1, 1, 2, 2, 2])

    def test_fit_corrects_a_biased_frozen_router(self):
        data, need = stream()
        frozen = data[tf.LOGIT_COLUMNS].to_numpy().argmax(1)
        lower, upper = tf.observations(tf.simulate(data, frozen))
        params, info = tf.fit(tf.features(data), lower, upper)
        self.assertTrue(info["converged"])
        adjusted = tf.decide(params, tf.features(data))
        self.assertGreater((adjusted == need).mean(), (frozen == need).mean() + .05)

    def test_prequential_window_one_is_the_frozen_router(self):
        data, _ = stream()
        frozen = data[tf.LOGIT_COLUMNS].to_numpy().argmax(1)
        for mode in ("fixed_logger", "on_policy", "escalation_only"):
            decisions, params, fits = tf.run_stream(data, 6, mode)
            first = data.window.eq(1).to_numpy()
            self.assertTrue(np.array_equal(decisions[first], frozen[first]), mode)
            self.assertEqual(len(fits), 5)
            self.assertTrue((decisions >= 0).all())

    def test_escalation_only_drifts_up(self):
        data, _ = stream()
        full, _, _ = tf.run_stream(data, 6, "fixed_logger")
        events, _, _ = tf.run_stream(data, 6, "escalation_only")
        later = data.window.ge(2).to_numpy()
        self.assertGreater(events[later].mean(), full[later].mean())


if __name__ == "__main__":
    unittest.main()
