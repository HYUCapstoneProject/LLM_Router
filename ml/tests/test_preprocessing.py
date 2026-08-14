from __future__ import annotations

import unittest

import pandas as pd

from llm_router_labeling.training.routerbench_preprocessing import (
    TRAINING_COLUMNS,
    assign_duplicate_groups,
    assign_group_splits,
    clean_model_input_text,
    prepare_training_dataset,
    select_balanced_train_groups,
)


PREPROCESSING_CONFIG = {
    "split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
    "seed": 42,
    "confidence_weights": {"high": 1.0, "medium": 0.5, "low": 0.0},
    "include_retrieval": True,
    "near_duplicate_jaccard_threshold": 0.8,
    "near_duplicate_min_tokens": 5,
    "near_duplicate_max_bucket_size": 200,
}


class RouterBenchPreprocessingTest(unittest.TestCase):
    def test_balanced_selection_keeps_rare_mixed_label_group_atomically(self) -> None:
        rows = [
            {
                "sample_id": "easy-code",
                "duplicate_group_id": "mixed-code-group",
                "difficulty_label": "easy",
                "domain": "code",
                "high_success_rate": 1.0,
            },
            {
                "sample_id": "hard-code",
                "duplicate_group_id": "mixed-code-group",
                "difficulty_label": "hard",
                "domain": "code",
                "high_success_rate": 1.0,
            },
        ]
        for label in ("easy", "medium", "hard"):
            for index in range(6):
                rows.append(
                    {
                        "sample_id": f"{label}-{index}",
                        "duplicate_group_id": f"{label}-group-{index}",
                        "difficulty_label": label,
                        "domain": "knowledge",
                        "high_success_rate": 1.0,
                    }
                )
        train = pd.DataFrame(rows)
        selected, metrics = select_balanced_train_groups(
            train,
            {
                "enabled": True,
                "target_per_class": 4,
                "label_order": ["easy", "medium", "hard"],
                "rare_domains": ["code", "conversation"],
            },
            seed=42,
        )
        chosen = train.loc[selected]
        self.assertEqual(
            chosen["difficulty_label"].value_counts().to_dict(),
            {"easy": 4, "medium": 4, "hard": 4},
        )
        self.assertEqual(
            set(chosen.loc[chosen["duplicate_group_id"].eq("mixed-code-group"), "sample_id"]),
            {"easy-code", "hard-code"},
        )
        self.assertEqual(metrics["mixed_label_group_count_selected"], 1)

    def test_balanced_selection_is_exact_and_preserves_rare_domains(self) -> None:
        rows = []
        for label in ("easy", "medium", "hard"):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{label}-{index}",
                        "duplicate_group_id": f"{label}-group-{index}",
                        "difficulty_label": label,
                        "domain": "code" if index < 2 else "knowledge",
                        "high_success_rate": 1.0 if index % 2 else 1.0 / 3.0,
                    }
                )
        train = pd.DataFrame(rows)
        selected, metrics = select_balanced_train_groups(
            train,
            {
                "enabled": True,
                "target_per_class": 6,
                "label_order": ["easy", "medium", "hard"],
                "rare_domains": ["code", "conversation"],
                "hard_strong_success_rate_threshold": 2.0 / 3.0,
            },
            seed=42,
        )
        chosen = train.loc[selected]
        self.assertEqual(
            chosen["difficulty_label"].value_counts().to_dict(),
            {"easy": 6, "medium": 6, "hard": 6},
        )
        self.assertEqual(
            chosen.loc[chosen["domain"].eq("code")]
            .groupby("difficulty_label")
            .size()
            .to_dict(),
            {"easy": 2, "medium": 2, "hard": 2},
        )
        self.assertEqual(metrics["target_per_class"], 6)

    def test_removes_only_benchmark_choice_wrappers(self) -> None:
        query = (
            "Please answer with the letter of the correct answer.\n\n"
            "What is 2 + 2?\nA) 3\nB) 4\n"
            'Print only a single choice  from "A" or "B" without explanation.'
        )
        cleaned, removed = clean_model_input_text(query)
        self.assertTrue(removed)
        self.assertEqual(cleaned, "What is 2 + 2?\nA) 3\nB) 4")

    def test_preserves_general_task_instructions_and_context(self) -> None:
        query = "Write a Python function.\n\nCan you parallelize it?"
        cleaned, removed = clean_model_input_text(query)
        self.assertFalse(removed)
        self.assertEqual(cleaned, query)

    def test_exact_and_near_duplicates_share_a_group(self) -> None:
        texts = pd.Series(
            [
                "Explain why the sky appears blue during a clear afternoon today",
                "Explain why the sky appears blue during a clear afternoon today!",
                "Explain why the sky appears blue during a clear afternoon",
                "Implement a binary search tree in Python",
            ]
        )
        groups, metrics = assign_duplicate_groups(
            texts,
            jaccard_threshold=0.8,
            minimum_tokens=5,
            maximum_bucket_size=200,
        )
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertEqual(groups.iloc[0], groups.iloc[2])
        self.assertNotEqual(groups.iloc[0], groups.iloc[3])
        self.assertGreaterEqual(metrics["duplicate_row_count"], 2)

    def test_group_split_is_reproducible_and_has_no_leakage(self) -> None:
        rows = []
        for index in range(30):
            rows.append(
                {
                    "duplicate_group_id": f"group-{index // 2}",
                    "difficulty_label": ("easy", "medium", "hard")[index % 3],
                    "domain": ("knowledge", "code")[index % 2],
                }
            )
        dataframe = pd.DataFrame(rows)
        first = assign_group_splits(
            dataframe, PREPROCESSING_CONFIG["split_fractions"], 42
        )
        second = assign_group_splits(
            dataframe, PREPROCESSING_CONFIG["split_fractions"], 42
        )
        self.assertEqual(first.tolist(), second.tolist())
        check = dataframe.assign(split=first).groupby("duplicate_group_id")[
            "split"
        ].nunique()
        self.assertTrue(check.eq(1).all())
        self.assertEqual(set(first), {"train", "validation", "test"})

    def test_prepare_separates_training_projection_and_metadata(self) -> None:
        rows = []
        confidence = ("high", "medium", "low")
        for index in range(15):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "query_text": f"Question number {index} with enough unique words for grouping",
                    "eval_name": "mmlu-test",
                    "domain": "knowledge" if index % 2 else "code",
                    "difficulty_label": ("easy", "medium", "hard")[index % 3],
                    "label_reason": "low_tier_consensus",
                    "label_confidence": confidence[index % 3],
                    "needs_retrieval": False,
                    "low_success_rate": 1.0,
                }
            )
        prepared = prepare_training_dataset(
            pd.DataFrame(rows),
            PREPROCESSING_CONFIG,
            preprocessing_version="test-v1",
        )
        self.assertEqual(tuple(prepared.training.columns), TRAINING_COLUMNS)
        self.assertNotIn("low_success_rate", prepared.training.columns)
        self.assertEqual(len(prepared.training), 10)
        self.assertEqual(len(prepared.metadata), 15)
        self.assertEqual(set(prepared.training["sample_weight"]), {0.5, 1.0})
        self.assertEqual(prepared.metrics["excluded_row_count"], 5)


if __name__ == "__main__":
    unittest.main()
