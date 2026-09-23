from __future__ import annotations

import unittest

import pandas as pd

from llm_router_labeling.evaluation.routerarena_preprocessing import (
    EVALUATION_COLUMNS,
    RouterArenaError,
    build_routerarena_model_input,
    prepare_routerarena_evaluation,
)


PREPROCESSING_CONFIG = {
    "exclude_exact_training_overlap": True,
    "exclude_exact_source_duplicates": True,
}


def routerarena_frame(row_count: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "Category": f"category-{index % 3}",
                "Domain": f"domain-{index % 2}",
                "Dataset name": f"dataset-{index % 4}",
                "Global Index": f"source-{index}",
                "Context": "Shared &amp; useful context" if index % 2 else "",
                "Question": f"What is the answer to unique question number {index}?",
                "Options": ["first", "second", "third"],
                "Answer": "B",
                "Metadata": "{}",
                "Keywords": "",
                "Difficulty": ("easy", "medium", "hard")[index % 3],
            }
        )
    return pd.DataFrame(rows)


class RouterArenaPreprocessingTest(unittest.TestCase):
    def test_builds_context_question_and_options_without_answer(self) -> None:
        model_input = build_routerarena_model_input(
            "Birds &amp; flight", "Which bird cannot fly?", ["Eagle", "Penguin"]
        )
        self.assertEqual(
            model_input,
            "Context:\nBirds & flight\n\nWhich bird cannot fly?\n\n"
            "A) Eagle\nB) Penguin",
        )
        self.assertNotIn("Answer", model_input)

    def test_full_split_maps_to_evaluation_contract(self) -> None:
        prepared = prepare_routerarena_evaluation(
            routerarena_frame(),
            PREPROCESSING_CONFIG,
            source_split="full",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
        )
        self.assertEqual(tuple(prepared.evaluation.columns), EVALUATION_COLUMNS)
        self.assertEqual(len(prepared.evaluation), 12)
        self.assertEqual(
            set(prepared.evaluation["difficulty_label"]),
            {"easy", "medium", "hard"},
        )
        self.assertTrue(
            prepared.metadata["label_source"]
            .eq("routerarena_official_difficulty")
            .all()
        )
        self.assertNotIn("answer", prepared.evaluation.columns)

    def test_rejects_non_official_difficulty_label(self) -> None:
        dataframe = routerarena_frame(3)
        dataframe.loc[1, "Difficulty"] = "remember"
        with self.assertRaises(RouterArenaError):
            prepare_routerarena_evaluation(
                dataframe,
                PREPROCESSING_CONFIG,
                source_split="full",
                dataset_version="test-data-v1",
                preprocessing_version="test-preprocess-v2",
            )

    def test_missing_question_is_preserved_in_metadata_and_excluded(self) -> None:
        dataframe = routerarena_frame(3)
        dataframe.loc[1, "Question"] = ""
        prepared = prepare_routerarena_evaluation(
            dataframe,
            PREPROCESSING_CONFIG,
            source_split="full",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
        )
        self.assertEqual(len(prepared.evaluation), 2)
        excluded = prepared.metadata.loc[
            prepared.metadata["source_sample_id"].eq("source-1")
        ].iloc[0]
        self.assertEqual(excluded["exclusion_reason"], "missing_question")

    def test_refusal_placeholder_and_embedded_answer_are_excluded(self) -> None:
        dataframe = routerarena_frame(4)
        dataframe.loc[1, "Question"] = "I'm sorry, I can't assist with that request."
        dataframe.loc[2, "Question"] = (
            "What is two plus two?\n\nReturn a boxed answer.\n\n\\boxed{{4}}"
        )
        prepared = prepare_routerarena_evaluation(
            dataframe,
            PREPROCESSING_CONFIG,
            source_split="full",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
        )
        self.assertEqual(len(prepared.evaluation), 2)
        reasons = prepared.metadata.set_index("source_sample_id")[
            "exclusion_reason"
        ].to_dict()
        self.assertEqual(reasons["source-1"], "refusal_placeholder")
        self.assertEqual(reasons["source-2"], "embedded_answer")

    def test_robustness_inherits_label_and_reference_from_full(self) -> None:
        full = routerarena_frame(4)
        robustness = full.copy()
        robustness["Question"] = robustness["Question"].map(
            lambda question: f"Paraphrased: {question}"
        )
        robustness["Difficulty"] = ""
        robustness["Domain"] = ""
        robustness["Options"] = [[] for _ in range(len(robustness))]
        prepared = prepare_routerarena_evaluation(
            robustness,
            PREPROCESSING_CONFIG,
            source_split="robustness",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
            reference_full=full,
        )
        self.assertEqual(
            prepared.evaluation["difficulty_label"].tolist(),
            full["Difficulty"].tolist(),
        )
        self.assertTrue(
            prepared.metadata["reference_model_input_text"].notna().all()
        )
        self.assertTrue(
            prepared.metadata["label_source"]
            .eq("routerarena_official_difficulty_full_join")
            .all()
        )

    def test_robustness_rejects_missing_reference(self) -> None:
        robustness = routerarena_frame(2)
        robustness["Difficulty"] = ""
        with self.assertRaises(RouterArenaError):
            prepare_routerarena_evaluation(
                robustness,
                PREPROCESSING_CONFIG,
                source_split="robustness",
                dataset_version="test-data-v1",
                preprocessing_version="test-preprocess-v2",
            )

    def test_exact_training_overlap_is_excluded(self) -> None:
        dataframe = routerarena_frame(4)
        overlapping_text = build_routerarena_model_input(
            dataframe.loc[0, "Context"],
            dataframe.loc[0, "Question"],
            dataframe.loc[0, "Options"],
        )
        training = pd.DataFrame({"model_input_text": [overlapping_text]})
        prepared = prepare_routerarena_evaluation(
            dataframe,
            PREPROCESSING_CONFIG,
            source_split="full",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
            training=training,
        )
        self.assertEqual(len(prepared.evaluation), 3)
        self.assertEqual(prepared.metrics["training_overlap_count"], 1)
        self.assertEqual(
            prepared.metadata.loc[
                prepared.metadata["source_sample_id"].eq("source-0"),
                "exclusion_reason",
            ].iloc[0],
            "exact_training_overlap",
        )

    def test_exact_source_duplicate_keeps_one_canonical_row(self) -> None:
        dataframe = routerarena_frame(4)
        dataframe.loc[1, ["Context", "Question", "Options"]] = [
            dataframe.loc[0, "Context"],
            dataframe.loc[0, "Question"],
            dataframe.loc[0, "Options"],
        ]
        prepared = prepare_routerarena_evaluation(
            dataframe,
            PREPROCESSING_CONFIG,
            source_split="full",
            dataset_version="test-data-v1",
            preprocessing_version="test-preprocess-v2",
        )
        self.assertEqual(len(prepared.evaluation), 3)
        self.assertEqual(
            prepared.metadata["exclusion_reason"].eq("duplicate_content").sum(), 1
        )


if __name__ == "__main__":
    unittest.main()
