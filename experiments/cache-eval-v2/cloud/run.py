"""C1~C4 캐시 판정 재생 (PLAN.md). 점수는 배포한 서빙 엔진으로 낸다.

    python run.py            # items.json → 결과를 gzip+base64 로 로그에 출력
"""
import base64
import gzip
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, "serving/src")
from huggingface_hub import snapshot_download  # noqa: E402

from llm_router_serving.inference import (  # noqa: E402
    TransformersBgeEmbedder,
    TransformersCrossEncoderReranker,
    TransformersNliReranker,
)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
items = json.load(open("items.json", encoding="utf-8"))

C2_COS = 0.80
C3_COS, C3_Q = 0.8774, 0.03581769
C4_COS, C4_NLI = 0.80, 0.28154424

bge = TransformersBgeEmbedder.load(model_id="BAAI/bge-large-en-v1.5",
                                   revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                                   model_version="bge-large-en-v1.5", expected_dimension=1024,
                                   max_length=512, device_name="cpu")
quora = TransformersCrossEncoderReranker.load(
    model_id=snapshot_download("cross-encoder/quora-distilroberta-base",
                               revision="f62e7a4b20b97195c2868e53ec59126df5eac743"),
    model_version="quora", max_length=512, device_name="cpu", local_files_only=True)
nli = TransformersNliReranker.load(
    model_id=snapshot_download("tasksource/deberta-base-long-nli",
                               revision="04dcf11f844b07bc57015169fca2b7d6df8299d5"),
    model_version="nli", max_length=1280, device_name="cpu", local_files_only=True)
torch = bge._torch
log("models loaded")


def embed_all(texts):
    """TransformersBgeEmbedder.embed 와 같은 CLS + L2 (512 토큰 절단). 길이순 배치로 묶는다."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        inputs = bge._tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
        with torch.inference_mode():
            cls = bge._model(**inputs).last_hidden_state[:, 0]
            vec = torch.nn.functional.normalize(cls, p=2, dim=1).numpy()
        out[idx] = vec
    return out


def bge_tokens(text):
    return len(bge._tokenizer(text)["input_ids"])


norm = lambda s: " ".join(s.split())                 # QuestionHash.normalize 와 같다
_nli, _quora = {}, {}


def nli_score(q, c):
    key = (q, c)
    if key not in _nli:
        _nli[key] = nli.rerank(q, [c]).scores[0]
    return _nli[key]


def quora_scores(q, cands):
    need = [c for c in cands if (q, c) not in _quora]
    if need:
        for c, s in zip(need, quora.rerank(q, need).scores, strict=True):
            _quora[(q, c)] = s
    return [_quora[(q, c)] for c in cands]


def semantic(policy, q, sims, cand_texts):
    """sims 는 저장된 항목 전체와의 코사인. 재사용할 항목의 위치 또는 None."""
    if policy == "C1" or sims.size == 0:
        return None
    if policy in ("C2", "C4"):
        j = int(np.argmax(sims))
        if sims[j] < (C2_COS if policy == "C2" else C4_COS):
            return None
        if policy == "C2":
            return j
        return j if nli_score(q, cand_texts(j)) >= C4_NLI else None
    k = min(5, sims.size)                           # C3: 상위 5개, 코사인 AND quora, 통과 중 quora 최대
    top = np.argpartition(-sims, k - 1)[:k]
    top = [int(j) for j in top if sims[j] >= C3_COS]
    if not top:
        return None
    scores = quora_scores(q, [cand_texts(j) for j in top])
    ok = [(s, j) for s, j in zip(scores, top, strict=True) if s >= C3_Q]
    return max(ok)[1] if ok else None


POLICIES = ("C1", "C2", "C3", "C4")


def run_stream(texts, classes, emb):
    """운영 방식 재생: exact → 의미 판정, 미스일 때만 저장."""
    res = {}
    for p in POLICIES:
        stored, stored_cls, exact = [], [], {}
        mat = np.zeros((len(texts), 1024), dtype=np.float32)
        seen_cls = set()
        rec = []
        for i, (t, c) in enumerate(zip(texts, classes, strict=True)):
            available = c in seen_cls
            key = norm(t)
            if key in exact:
                j = exact[key]
                rec.append(("exact", stored_cls[j] == c, available))
                continue
            n = len(stored)
            sims = mat[:n] @ emb[i] if n else np.zeros(0, dtype=np.float32)
            j = semantic(p, t, sims, lambda j: texts[stored[j]])
            if j is not None:
                rec.append(("semantic", stored_cls[j] == c, available))
                continue
            rec.append(("miss", None, available))
            mat[n] = emb[i]
            stored.append(i)
            stored_cls.append(c)
            exact[key] = n
            seen_cls.add(c)
        res[p] = summarize_stream(rec)
    return res


def summarize_stream(rec):
    n = len(rec)
    hits = [r for r in rec if r[0] != "miss"]
    correct = sum(1 for r in hits if r[1])
    wrong = len(hits) - correct
    available = sum(1 for r in rec if r[2])
    sem = [r for r in hits if r[0] == "semantic"]
    return {"requests": n, "hits": len(hits), "exact_hits": len(hits) - len(sem), "semantic_hits": len(sem),
            "correct": correct, "wrong": wrong, "semantic_wrong": sum(1 for r in sem if not r[1]),
            "available": available, "correct_when_available": sum(1 for r in hits if r[1] and r[2])}


def run_pairs(cache, queries, emb_cache, emb_q):
    res = {p: [] for p in POLICIES}
    exact = {}
    for j, t in enumerate(cache):
        exact.setdefault(norm(t), j)
    own_idx = {t: j for j, t in enumerate(cache)}
    sims_all = emb_q @ emb_cache.T
    for qi, q in enumerate(queries):
        own = own_idx.get(q["own"])
        for p in POLICIES:
            key = norm(q["text"])
            if key in exact:
                j = exact[key]
            else:
                j = semantic(p, q["text"], sims_all[qi], lambda j: cache[j])
            res[p].append(None if j is None else ("own" if j == own else "other"))
    out = {}
    for p, dec in res.items():
        pos = [d for d, q in zip(dec, queries, strict=True) if q["label"] == 1]
        neg = [d for d, q in zip(dec, queries, strict=True) if q["label"] == 0]
        out[p] = {"n_pos": len(pos), "n_neg": len(neg),
                  "pos_own": sum(1 for d in pos if d == "own"), "pos_other": sum(1 for d in pos if d == "other"),
                  "neg_hit": sum(1 for d in neg if d is not None), "neg_own": sum(1 for d in neg if d == "own"),
                  "decisions": dec}
    return out


result = {"set": items["set"], "meta": {}}
kind = items["kind"]
if kind == "stream":
    req = items["requests"]
    texts, classes = [r["text"] for r in req], [r["cls"] for r in req]
    result["meta"] = {"requests": len(texts), "classes": len(set(classes)),
                      "bge_over_512": sum(1 for t in texts if bge_tokens(t) > 512)}
    emb = embed_all(texts)
    log(f"embedded {len(texts)}")
    result["policies"] = run_stream(texts, classes, emb)
    if items["set"] == "classification":                # 데이터셋별로도 본다
        result["by_dataset"] = {}
        for ds in sorted({r["dataset"] for r in req}):
            idx = [i for i, r in enumerate(req) if r["dataset"] == ds]
            result["by_dataset"][ds] = run_stream([texts[i] for i in idx], [classes[i] for i in idx], emb[idx])
elif kind == "dstream":
    pool = [r for r in items["pool"] if bge_tokens(r["text"]) <= 512]
    texts = [r["text"] for r in pool]
    emb = embed_all(texts)
    log(f"embedded pool {len(texts)} (excluded {len(items['pool']) - len(texts)} over 512 tokens)")
    result["meta"] = {"pool": len(texts), "excluded_over_512": len(items["pool"]) - len(texts)}
    result["traces"] = {}
    for r in (0.0, 0.1, 0.3, 0.5):
        for seed in range(3):
            rng = random.Random(items["seed"] + 1000 * seed + int(r * 100))
            order = list(range(len(texts)))
            rng.shuffle(order)
            trace, issued, nxt = [], [], 0
            for _ in range(4000):
                if issued and rng.random() < r:
                    i = rng.choice(issued)
                else:
                    i = order[nxt]
                    nxt += 1
                    issued.append(i)
                trace.append(i)
            res = run_stream([texts[i] for i in trace], [pool[i]["cls"] for i in trace], emb[trace])
            result["traces"][f"r{int(r * 100)}_s{seed}"] = res
            log(f"trace r={r} seed={seed} done")
elif kind == "pairs":
    result["subsets"] = {}
    for name, s in items["subsets"].items():
        cache, queries = s["cache"], s["queries"]
        emb_c = embed_all(cache)
        emb_q = embed_all([q["text"] for q in queries])
        log(f"{name}: embedded cache {len(cache)} queries {len(queries)}")
        r = run_pairs(cache, queries, emb_c, emb_q)
        r["meta"] = {"cache": len(cache), "queries": len(queries), "n_background": s.get("n_background", 0),
                     "query_kinds": [q.get("kind") for q in queries], "query_gid": [q.get("gid") for q in queries],
                     "query_category": [q.get("category") for q in queries]}
        result["subsets"][name] = r
result["meta"]["nli_pairs_scored"] = len(_nli)
result["meta"]["quora_pairs_scored"] = len(_quora)
result["meta"]["elapsed_s"] = round(time.time() - T0)
log(json.dumps(result["meta"])[:500])

blob = base64.b64encode(gzip.compress(json.dumps(result).encode())).decode()
print("===OUT-BEGIN===", flush=True)
for k in range(0, len(blob), 3000):
    print(blob[k:k + 3000], flush=True)
print("===OUT-END===", flush=True)
