"""External evaluation preparation contracts; synthetic rows only, no downloads."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from llm_router_labeling.evaluation import external_data as x


class WrapperContracts(unittest.TestCase):
    def test_known_wrappers_are_removed_and_question_kept(self):
        mmlu = ("Answer the following law question. The last line of your response should be of the following format: "
                "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n\nWhat is a tort?\n\nA. x\nB. y\n\nLet's think step by step.")
        self.assertEqual(x.strip_wrapper("mmlupro", mmlu), "What is a tort?\n\nA. x\nB. y")
        boxed = ("Solve the following math problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.\n\n"
                 "Find 2+2.\n\nRemember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.")
        self.assertEqual(x.strip_wrapper("boxed_math", boxed), "Find 2+2.")
        hle = ("Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\n"
               "Confidence: {your confidence score between 0% and 100% for your answer}. Which view?\n\nA. one")
        self.assertEqual(x.strip_wrapper("hle_head", hle), "Which view?\n\nA. one")
        lcb = "You are an expert.\n### Question:\nSort a list.\n\n### Format: stdin\n\n### Answer: (use the provided format with backticks)\n\n"
        self.assertEqual(x.strip_wrapper("livecodebench", lcb), "Sort a list.\n\n### Format: stdin")

    def test_unmatched_wrapper_is_reported_not_guessed(self):
        self.assertIsNone(x.strip_wrapper("mmlupro", "What is a tort?"))
        self.assertIsNone(x.strip_wrapper("livecodebench", "no markers"))
        with self.assertRaises(x.ExternalDataError):
            x.strip_wrapper("unknown", "text")


class OverlapContracts(unittest.TestCase):
    STEM = "Which of the following principles is not part of the governance code of practice in the United Kingdom?"

    def test_stem_ignores_option_order_and_format(self):
        a = self.STEM + "\nA) Openness\nB) Integrity"
        b = self.STEM + "\n\nOptions:\n\nA. Integrity\n\nB. Openness\nC. None"
        self.assertEqual(x.stem_key(a), x.stem_key(b))
        self.assertIsNone(x.stem_key("What is two plus two?\nA) 3\nB) 4"))

    def frames(self):
        reference = pd.DataFrame({"sample_id": ["r1", "r2"],
                                  "model_input_text": [self.STEM + "\nA) Openness\nB) Integrity",
                                                       "Explain photosynthesis in plants with a detailed example of the light reactions."]})
        rows = []
        for sid, bench, text in [
            ("lrb:1", "llmrouterbench", self.STEM + "\n\nA. Integrity\nB. Openness\nC. Other"),
            ("lrb:2", "llmrouterbench", "Describe how a compiler turns source code into machine instructions step by step for a beginner."),
            ("sp:1", "sprout", "Describe how a compiler turns source code into machine instructions step by step for a beginner.\n"),
            ("sp:2", "sprout", "Write a short poem about autumn leaves falling in a quiet park during the evening light."),
        ]:
            rows.append({"sample_id": sid, "benchmark": bench, "source_family": "f", "model_input_text": text, "exclusion_reason": None})
        return reference, pd.DataFrame(rows)

    def test_project_overlap_and_secondary_duplicates_are_excluded(self):
        reference, external = self.frames()
        dedup = {"near_duplicate_jaccard_threshold": .9, "near_duplicate_min_tokens": 5, "near_duplicate_max_bucket_size": 200}
        out, report = x.mark_overlaps(external, reference, dedup)
        reasons = dict(zip(out.sample_id, out.exclusion_reason))
        self.assertEqual(reasons["lrb:1"], "overlaps_project_data")
        self.assertIsNone(reasons["lrb:2"])
        self.assertEqual(reasons["sp:1"], "duplicate_of_primary_external")
        self.assertIsNone(reasons["sp:2"])
        self.assertEqual(report["secondary_duplicate_rows"], 1)

    def test_shared_passage_stays_in_one_split(self):
        story = "Story: " + " ".join(f"word{i}" for i in range(400))
        frame = pd.DataFrame({"duplicate_group_id": ["g1", "g2", "g3"], "stem_key": [None, None, None],
                              "model_input_text": [story + " Question: who?", story + " Question: where?", "unrelated short text"]})
        groups = x.split_groups(frame)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])
        split = x.assign_split(frame.assign(split_group_id=groups), .5, 1)
        self.assertEqual(split.iloc[0], split.iloc[1])


class OutcomeContracts(unittest.TestCase):
    def test_failed_calls_are_missing_and_labels_follow_project_rule(self):
        pool = ["m-a", "m-b", "m-c"]
        config = {"pool_prices_usd_per_million_tokens": {m: [1, 1] for m in pool},
                  "datasets": {"simpleqa": {"split": "test", "slice": "main_new_family", "wrapper": "none"}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = {"m-a": (1.0, .01, 5), "m-b": (0.0, 0.0, 0), "m-c": (0.0, .02, 7)}
            for model, (score, cost, tokens) in outcomes.items():
                path = root / "simpleqa" / "test" / model / "run.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"records": [{"index": 1, "prompt": "Who won the award in 2010?", "score": score,
                                                         "cost": cost, "completion_tokens": tokens}]}))
            wide = x.load_llmrouterbench(root, config)
        self.assertTrue(np.isnan(wide.loc[0, "success_m_b"]))
        self.assertFalse(wide.loc[0, "valid_m_b"])
        labelled = x.add_labels(wide, [x.model_column(m) for m in pool], {"hard": .27, "easy": .64}, 2)
        self.assertEqual(int(labelled.loc[0, "valid_model_count"]), 2)
        self.assertAlmostEqual(labelled.loc[0, "overall_solve_rate"], .5)
        self.assertEqual(labelled.loc[0, "difficulty_label"], "medium")
        too_strict = x.add_labels(wide, [x.model_column(m) for m in pool], {"hard": .27, "easy": .64}, 3)
        self.assertEqual(too_strict.loc[0, "exclusion_reason"], "too_few_valid_models")

    def test_inconsistent_prompts_are_excluded(self):
        config = {"pool_prices_usd_per_million_tokens": {"m-a": [1, 1], "m-b": [1, 1]},
                  "datasets": {"simpleqa": {"split": "test", "slice": "main_new_family", "wrapper": "none"}}}
        with tempfile.TemporaryDirectory() as tmp:
            for model, prompt in (("m-a", "one question"), ("m-b", "another question")):
                path = Path(tmp) / "simpleqa" / "test" / model / "run.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"records": [{"index": 7, "prompt": prompt, "score": 1.0, "cost": .1, "completion_tokens": 1}]}))
            wide = x.load_llmrouterbench(Path(tmp), config)
        self.assertEqual(wide.loc[0, "exclusion_reason"], "inconsistent_prompt_across_models")


if __name__ == "__main__":
    unittest.main()
