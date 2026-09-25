from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from llm_router_labeling.routing_data.continual import assign_continual_splits
from llm_router_labeling.routing_data.preprocessing import _routerbench_accepted, _apply_selected_label
from llm_router_labeling.routing_data.text import clean_text, clean_xroute_text, structured_duplicate_key, compatible_near_duplicate
from llm_router_labeling.training.routerbench_preprocessing import assign_duplicate_groups


class InitialAnchorDataTest(unittest.TestCase):
    def test_code_scope_and_operators_are_not_conflated(self):
        inputs = pd.Series([
            'def f(x):\n    if x > 0:\n        return x + 1\n    return 0',
            'def f(x):\n    if x > 0:\n        return x - 1\n    return 0',
            'def f(x):\n    if x > 0:\n        return x + 1\n        return 0',
        ])
        self.assertEqual(clean_text(inputs.iloc[0]), inputs.iloc[0])
        groups, _ = assign_duplicate_groups(inputs, jaccard_threshold=.9, minimum_tokens=5, maximum_bucket_size=200, normalizer=structured_duplicate_key, near_duplicate_guard=compatible_near_duplicate)
        self.assertEqual(groups.nunique(), 3)

    def test_unknown_system_and_code_body_instruction_survive(self):
        prompt = '[System Instruction]\nImplement the function body only. Do not use recursion.\n\n[User Query]\ndef f(n):\n    """Compute n!"""'
        cleaned = clean_xroute_text(prompt)
        self.assertIn('Do not use recursion.', cleaned)
        self.assertIn('    """Compute n!"""', cleaned)
        self.assertNotIn('[User Query]', cleaned)

    def test_missing_and_unknown_outcomes_are_not_failures(self):
        frame = pd.DataFrame({'sample_id':['mmlu.1','mbpp.2','new-task.3'], 'eval_name':['mmlu','mbpp','new-task'], 'a':[1,.7,1], 'b':[None,.69,0]})
        accepted = _routerbench_accepted(frame, ['a','b'])
        self.assertEqual(accepted.notna().sum(axis=1).tolist(), [1,2,0])
        self.assertEqual(accepted.astype('boolean').sum(axis=1).tolist(), [1,1,0])

    def test_continual_split_protects_groups_and_official_tests(self):
        config = json.loads((Path(__file__).parents[1] / 'config/modernbert_initial_v1_data.json').read_text())
        rows = []
        for i in range(1000):
            for _ in range(2):
                rows.append({'sample_id':str(len(rows)), 'duplicate_group_id':f'g{i}', 'split':'train', 'source_dataset':'rb' if i % 2 else 'xrb', 'category':'math', 'overall_solve_rate':(i % 11)/10})
        rows += [{'sample_id':'sealed', 'duplicate_group_id':'sealed', 'split':'external_test', 'source_dataset':'xrb','category':'math','overall_solve_rate':.5}]
        original = pd.DataFrame(rows)
        output = assign_continual_splits(original, config)
        self.assertEqual(output.groupby('duplicate_group_id').split.nunique().max(), 1)
        self.assertEqual(output.iloc[-1]['split'], 'external_test')
        shuffled = assign_continual_splits(original.sample(frac=1, random_state=19), config)
        self.assertEqual(output.set_index('sample_id').split.to_dict(), shuffled.set_index('sample_id').split.to_dict())
        for split, fraction in config['continual_split']['fractions'].items():
            self.assertLess(abs((output.split == split).sum()/2000-fraction), .015)


if __name__ == '__main__':
    unittest.main()
