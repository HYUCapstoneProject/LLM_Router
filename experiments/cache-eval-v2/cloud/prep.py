"""평가 세트를 만든다 (PLAN.md). Cloud Build 안에서 실행하고 원문은 빌드 밖으로 나가지 않는다.

    SET=lmarena|search|classification|dstream|pairs|service python prep.py
출력: items.json
"""
import json
import os
import random
import subprocess

import pyarrow.parquet as pq

SEED = 20261013
SET = os.environ["SET"]
REV = {
    "vCache/SemBenchmarkLmArena": "c6f36229bef6a5f7936e5b9d982bd5766ef2f862",
    "vCache/SemBenchmarkSearchQueries": "5159bff8ccf5d7c57dd82cbb24a76e2e5d24f1c0",
    "vCache/SemBenchmarkClassification": "373719c21898ba0914b1934e88e0486de2420c02",
    "redis/langcache-sentencepairs-v1": "fcf7322a21d28643896a6e808fe069bf1417fe77",
}


def hf(repo, path, columns):
    local = "/tmp/" + path.replace("/", "_")
    if not os.path.exists(local):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{REV[repo]}/{path}"
        subprocess.run(["curl", "-sSL", "--fail", "-o", local, url], check=True)
    return pq.read_table(local, columns=columns).to_pylist()


def lmarena_rows():
    rows = [r for r in hf("vCache/SemBenchmarkLmArena", "train.parquet", ["prompt", "ID_Set"])
            if r["prompt"] and r["prompt"].strip()]
    classes = sorted({r["ID_Set"] for r in rows})
    random.Random(SEED).shuffle(classes)
    return rows, classes


out = {"set": SET, "seed": SEED}
if SET == "lmarena":
    rows, classes = lmarena_rows()
    chosen = set(classes[:700])
    req = [{"cls": str(r["ID_Set"]), "text": r["prompt"]} for r in rows if r["ID_Set"] in chosen]
    random.Random(SEED + 1).shuffle(req)
    out.update(kind="stream", requests=req, n_classes_total=len(classes))
elif SET == "search":
    rows = [r for r in hf("vCache/SemBenchmarkSearchQueries", "train.parquet", ["prompt", "id_set", "cluster_id"])
            if r["prompt"] and r["prompt"].strip()]
    by = {}
    for r in rows:
        by.setdefault(r["cluster_id"], []).append(r)
    clusters = sorted(by)
    random.Random(SEED).shuffle(clusters)
    req = []
    for c in clusters:
        if len(req) >= 10000:
            break
        req += [{"cls": str(r["id_set"]), "text": r["prompt"], "cluster": str(c)} for r in by[c]]
    random.Random(SEED + 1).shuffle(req)
    out.update(kind="stream", requests=req)
elif SET == "classification":
    rows = [r for r in hf("vCache/SemBenchmarkClassification", "train.parquet",
                          ["prompt", "dataset_name", "response_llama_3_8b"])
            if r["prompt"] and r["prompt"].strip() and r["response_llama_3_8b"] is not None]
    rows = random.Random(SEED).sample(rows, 8000)
    req = [{"cls": f"{r['dataset_name']}::{str(r['response_llama_3_8b']).strip().lower()}", "text": r["prompt"],
            "dataset": r["dataset_name"]} for r in rows]
    out.update(kind="stream", requests=req)
elif SET == "dstream":
    rows = pq.read_table("stream.parquet", columns=["sample_id", "query_text"]).to_pylist()
    out.update(kind="dstream", pool=[{"cls": r["sample_id"], "text": r["query_text"]} for r in rows])
elif SET == "pairs":
    subsets = {}
    for cfg, n in (("pit2015", None), ("qqp", 4000), ("paws", 4000), ("mrpc", None)):
        rows = hf("redis/langcache-sentencepairs-v1", f"{cfg}/test-00000-of-00001.parquet",
                  ["sentence1", "sentence2", "label"])
        rows = [r for r in rows if r["sentence1"] and r["sentence2"] and r["sentence1"].strip()
                and r["sentence2"].strip() and r["label"] is not None]
        if n is not None and len(rows) > n:
            rows = random.Random(SEED).sample(rows, n)
        cache = sorted({r["sentence1"] for r in rows})
        subsets[cfg] = {"cache": cache,
                        "queries": [{"text": r["sentence2"], "own": r["sentence1"], "label": int(r["label"])}
                                    for r in rows]}
    out.update(kind="pairs", subsets=subsets)
elif SET == "service":
    groups = [json.loads(l) for l in open("service.jsonl", encoding="utf-8")]
    rows, classes = lmarena_rows()
    first = {}
    for r in rows:
        first.setdefault(r["ID_Set"], r["prompt"])
    background = [first[c] for c in classes[700:3700]]          # lmarena 표본(앞 700개)과 겹치지 않는 클래스
    queries = []
    for g in groups:
        for k, label in (("para_light", 1), ("para_heavy", 1), ("trap", 0), ("trap2", 0)):
            if k in g:
                queries.append({"text": g[k], "own": g["original"], "label": label, "kind": k,
                                "category": g["category"], "gid": g["gid"]})
    out.update(kind="pairs", subsets={"service": {"cache": [g["original"] for g in groups] + background,
                                                  "queries": queries, "n_background": len(background)}})
json.dump(out, open("items.json", "w", encoding="utf-8"), ensure_ascii=False)
print("prep", SET, {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}, flush=True)
