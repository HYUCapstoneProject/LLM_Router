from __future__ import annotations

import unittest

import pandas as pd

from llm_router_labeling.routing_data.outcome_audit import (
    _mcq_target,
    apply_routerbench_audit,
    apply_xroute_audit,
    audit_xroute,
    final_numbers,
    math_answer_key,
    squad_contains_gold,
    strict_choice_letter,
)
from llm_router_labeling.routing_data.preprocessing import _validate_xroute_rows


class FinalAnswerTest(unittest.TestCase):
    def test_verbose_answer_restating_inputs_uses_first_number(self) -> None:
        response = "['360 / 60 = 6\\n\\nBarry would need to buy 6 bottles to last for 180 days.']"
        self.assertIn(6.0, final_numbers(response))

    def test_equation_line_uses_last_number(self) -> None:
        self.assertIn(7.0, final_numbers("Bags: 3 + 4 = 7"))

    def test_middle_numbers_never_count(self) -> None:
        # A truncated response must not pass because the gold value appears mid-line.
        self.assertNotIn(1000.0, final_numbers("80 then 1000 then 1001 then 80"))

    def test_boxed_and_latex_thousands(self) -> None:
        self.assertEqual(final_numbers("so \\boxed{\\$24{,}960}"), [24960.0])
        self.assertEqual(final_numbers("x\n#### 1,200"), [1200.0])


class ChoiceLetterTest(unittest.TestCase):
    def test_unambiguous_forms(self) -> None:
        for response in ("A", "(B)", "**C**", "D. You must send it", "The correct answer is (E)."):
            self.assertIsNotNone(strict_choice_letter(response), response)

    def test_sentence_starting_with_article_is_not_a_letter(self) -> None:
        self.assertIsNone(strict_choice_letter("A chemical change because the vinegar reacts."))


class MathAndSquadTest(unittest.TestCase):
    def test_format_equivalent_math_answers(self) -> None:
        self.assertEqual(math_answer_key("\\text{odd}"), math_answer_key("odd"))
        self.assertEqual(math_answer_key("(-4,1)"), math_answer_key("(-4,\\,1)"))
        self.assertEqual(math_answer_key("5,\\!866,\\!608"), math_answer_key("5866608"))
        self.assertNotEqual(math_answer_key("-9"), math_answer_key("9"))

    def test_squad_containment(self) -> None:
        self.assertTrue(squad_contains_gold('The term "prehensio" comes from Latin.', "Latin"))
        self.assertFalse(squad_contains_gold("It is Greek.", "Latin"))
        self.assertFalse(squad_contains_gold("Platinum", "Latin"))


class RouterBenchPromptTest(unittest.TestCase):
    def test_few_shot_prompt_keeps_only_the_target_block(self) -> None:
        prompt = (
            "['Q1\\nA) 0\\nB) 3\\nPrint only a single choice. Answer:\\nA\\n\\n"
            "Please answer with the letter of the correct answer.\\n\\nSolve x.\\nA) -71\\nB) 71\\n"
            "Print only a single choice. Answer:']"
        )
        stem, options = _mcq_target(prompt)
        self.assertIn("Solve x.", stem)
        self.assertEqual([letter for letter, _ in options], ["A", "B"])


class ApplyAuditTest(unittest.TestCase):
    def test_routerbench_cells_are_overwritten_and_exclusions_returned(self) -> None:
        labeled = pd.DataFrame({"sample_id": ["gsm.1", "mmlu.2"]})
        accepted = pd.DataFrame({"m1": [False, True], "m2": [False, False]}, dtype="boolean")
        performance = pd.DataFrame({"m1": [0.5, 1.0], "m2": [0.25, 0.0]})
        audit = {
            "routerbench_corrections": pd.DataFrame(
                [{"sample_id": "gsm.1", "model": "m1", "verified_performance": 1.0, "verified_accepted": True}]
            ),
            "routerbench_exclusions": pd.DataFrame({"sample_id": ["mmlu.2"], "rule": ["mmlu_redux_expert"]}),
        }
        accepted, excluded = apply_routerbench_audit(accepted, labeled, audit, performance)
        self.assertTrue(bool(accepted.loc[0, "m1"]))
        self.assertFalse(bool(accepted.loc[0, "m2"]))
        self.assertEqual(performance.loc[0, "m1"], 1.0)
        self.assertEqual(excluded, {"mmlu.2"})

    def test_xroute_false_negatives_flip_but_successes_are_kept(self) -> None:
        rows = pd.DataFrame(
            [
                # bare correct letter recorded as failure
                {"task_name": "mmlu", "query": "q1", "metric": "em_mc", "task_id": None, "embedding_id": 1,
                 "model_name": "a", "performance": 0.0, "ground_truth": "C", "response": "C"},
                # wrong letter stays a failure
                {"task_name": "mmlu", "query": "q1", "metric": "em_mc", "task_id": None, "embedding_id": 1,
                 "model_name": "b", "performance": 0.0, "ground_truth": "C", "response": "(B)"},
                # recorded success is never revoked
                {"task_name": "mmlu", "query": "q1", "metric": "em_mc", "task_id": None, "embedding_id": 1,
                 "model_name": "c", "performance": 1.0, "ground_truth": "C", "response": "(B) (C)"},
                {"task_name": "gsm8k", "query": "q2", "metric": "GSM8K", "task_id": None, "embedding_id": 2,
                 "model_name": "a", "performance": 0.0, "ground_truth": "...\n#### 720",
                 "response": "Greta has $720.00 left to spend."},
                {"task_name": "squad", "query": "q3", "metric": "f1_score", "task_id": None, "embedding_id": 3,
                 "model_name": "a", "performance": 0.33, "ground_truth": "Latin",
                 "response": "It comes from Latin."},
            ]
        )
        corrections, summary = audit_xroute(rows)
        self.assertEqual(
            summary,
            {"em_mc_format_false_negative": 1, "gsm8k_format_false_negative": 1, "squad_contains_gold_answer": 1},
        )
        validated = _validate_xroute_rows(
            rows,
            split="train",
            candidates=("a", "b", "c"),
            exact_metrics={"em_mc", "GSM8K"},
            continuous_metrics={"f1_score"},
            outcome_audit={"xroute_corrections": corrections},
        )
        self.assertEqual(validated["accepted"].astype(bool).tolist(), [True, False, True, True, True])
        self.assertEqual(validated.loc[4, "performance"], 1.0)
        self.assertEqual(validated["outcome_verified"].tolist(), [True, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
