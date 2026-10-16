"""채점할 쌍을 모은다. 대화형·D2 재현본의 (변형, 후보) 쌍마다 원문과 차이 문장 두 형태를 만든다.

    python make_pairs.py
"""
import json
from pathlib import Path

from diffpair import reduce_pair

HERE = Path(__file__).resolve().parent
NAT, D2 = HERE.parent / "cache-natural-v1", HERE.parent / "d2-replica-v1"
SEP = "␟"

pairs = []                                       # {key, a(질문), b(캐시 원질문)}
for p in json.load(open(NAT / "cloud/pairs.json", encoding="utf-8")):
    pairs.append({"key": "chat|" + p["pid"], "a": p["a"], "b": p["b"]})

groups = {json.loads(l)["gid"]: json.loads(l) for l in open(D2 / "data/dataset.jsonl", encoding="utf-8")}
background = json.load(open(D2 / "data/background.json", encoding="utf-8"))
out = json.load(open(D2 / "results/out.json", encoding="utf-8"))
for k in out["pair_scores"]:
    text, cid = k.split(SEP)
    kind, ref = cid.split(":", 1)
    pairs.append({"key": "d2|" + k, "a": text, "b": groups[ref]["original"] if kind == "o" else background[int(ref)]})

texts = {}                                       # (a, b) -> 인덱스
rows = []
for p in pairs:
    da, db = reduce_pair(p["a"], p["b"])
    p["full"], p["diff"] = [], []
    for form, (a, b) in (("full", (p["a"], p["b"])), ("diff", (da, db))):
        if (a, b) not in texts:
            texts[(a, b)] = len(rows)
            rows.append([a, b])
        p[form] = texts[(a, b)]
index = {p["key"]: {"full": p["full"], "diff": p["diff"]} for p in pairs}
json.dump(rows, open(HERE / "cloud/pairs.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(index, open(HERE / "data/index.json", "w", encoding="utf-8"), ensure_ascii=False)
print("pairs", len(pairs), "unique rows", len(rows))
