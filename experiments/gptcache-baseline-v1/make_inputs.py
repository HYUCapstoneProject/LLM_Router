"""GPTCache 실행 입력을 만든다. 캐시 항목과 질문(텍스트 + ID)만 보내고, 정답·분할은 로컬에 둔다.

    python make_inputs.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAT, D2 = HERE.parent / "cache-natural-v1", HERE.parent / "d2-replica-v1"

nat = [json.loads(l) for l in open(NAT / "data/dataset.jsonl", encoding="utf-8")]
d2 = [json.loads(l) for l in open(D2 / "data/dataset.jsonl", encoding="utf-8")]
background = json.load(open(D2 / "data/background.json", encoding="utf-8"))

inputs = {
    "chat": {"entries": [[g["id"], g["original"]] for g in nat],
             "queries": [[f"{g['id']}|{k}", g[k]] for g in nat for k in ("para_light", "para_heavy", "trap")]},
    "d2": {"entries": [[f"o:{g['gid']}", g["original"]] for g in d2] + [[f"b:{i}", t] for i, t in enumerate(background)],
           "queries": [[f"{g['gid']}|{k}", g[k]] for g in d2 for k in ("para", "trap")]},
}
meta = {}
for g in nat:
    split = "calib" if hashlib.sha256(g["id"].encode()).digest()[0] % 3 == 0 else "test"
    for k in ("para_light", "para_heavy", "trap"):
        meta[f"chat|{g['id']}|{k}"] = {"own": g["id"], "is_para": k != "trap", "split": split, "stratum": g["domain"]}
for g in d2:
    for k in ("para", "trap"):
        meta[f"d2|{g['gid']}|{k}"] = {"own": f"o:{g['gid']}", "is_para": k == "para",
                                      "split": "calib" if g["split"] == "calibration" else "test", "stratum": g["stratum"]}
json.dump(inputs, open(HERE / "cloud/inputs.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(meta, open(HERE / "data/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
print({ds: {k: len(v) for k, v in d.items()} for ds, d in inputs.items()})
