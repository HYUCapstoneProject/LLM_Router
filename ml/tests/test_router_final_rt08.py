import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import rt08  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BuildTest(unittest.TestCase):
    def test_uses_v2_old_labels_drops_exclusions_and_keeps_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prep = tmp / "prepare"
            prep.mkdir()
            rows = {
                "train": [("routerbench:a", "routerbench", False, "easy", "medium"),
                          ("routerbench:b", "routerbench", True, "hard", "hard"),
                          ("xroutebench:c", "xroutebench", False, "medium", "medium"),
                          ("sprout:d", "sprout", False, None, None)],
                "dev": [("routerbench:e", "routerbench", False, "hard", "easy")],
            }
            outputs = {}
            for role, items in rows.items():
                frame = pd.DataFrame(items, columns=["sample_id", "source_dataset", "excluded_v2", "old_label_v2", "old_label_v1"])
                frame["model_input_text"] = "q " + frame.sample_id
                path = prep / f"{role}.parquet"
                frame.to_parquet(path, index=False)
                outputs[path.name] = {"sha256": _sha(path), "rows": len(frame)}
            (prep / "manifest.json").write_text(json.dumps({"outputs": outputs}))
            original = pd.DataFrame({"sample_id": ["routerbench:a", "routerbench:b", "xroutebench:c", "routerbench:e"],
                                     "duplicate_group_id": ["g1", "g2", "g3", "g4"], "source_split": "x",
                                     "category": "c", "source_task": "t"})
            original.iloc[:3].to_parquet(tmp / "a_train.parquet", index=False)
            original.iloc[3:].to_parquet(tmp / "a_dev.parquet", index=False)
            manifest = rt08.build(prep, tmp / "a_train.parquet", tmp / "a_dev.parquet", tmp / "out")
            data = pd.read_parquet(tmp / "out/training.parquet").set_index("sample_id")
            self.assertEqual(sorted(data.index), ["routerbench:a", "routerbench:e", "xroutebench:c"])
            self.assertEqual(data.loc["routerbench:a", "difficulty_label"], "easy")
            self.assertEqual(data.loc["routerbench:e", "split"], "validation")
            self.assertEqual(manifest["dropped_v2_exclusions"], 1)
            self.assertEqual(manifest["max_train_steps"], 2007)


if __name__ == "__main__":
    unittest.main()
