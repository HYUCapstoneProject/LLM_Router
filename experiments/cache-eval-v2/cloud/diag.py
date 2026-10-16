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



POLICIES = ("C3", "C4")
SCORE = {}


def semantic_logged(policy, q, sims, cand_texts):
    j = semantic(policy, q, sims, cand_texts)
    if j is not None:
        c = cand_texts(j)
        SCORE[(policy, q, c)] = (float(sims[j]), _quora.get((q, c)), _nli.get((q, c)))
    return j


def stream_hits(texts, classes, emb, policy):
    stored, stored_cls, exact = [], [], {}
    mat = np.zeros((len(texts), 1024), dtype=np.float32)
    hits = []
    for i, (t, c) in enumerate(zip(texts, classes, strict=True)):
        key = norm(t)
        if key in exact:
            continue
        n = len(stored)
        sims = mat[:n] @ emb[i] if n else np.zeros(0, dtype=np.float32)
        j = semantic_logged(policy, t, sims, lambda j: texts[stored[j]])
        if j is not None:
            m = texts[stored[j]]
            cos, qs, ns = SCORE[(policy, t, m)]
            hits.append({"q": t, "m": m, "ok": stored_cls[j] == c, "cos": cos, "quora": qs, "nli": ns})
            continue
        mat[n] = emb[i]; stored.append(i); stored_cls.append(c); exact[key] = n
    return hits


out = {}
if items["kind"] == "stream":
    req = items["requests"]
    texts, classes = [r["text"] for r in req], [r["cls"] for r in req]
    emb = embed_all(texts)
else:
    pool = [r for r in items["pool"] if bge_tokens(r["text"]) <= 512]
    alltexts = [r["text"] for r in pool]
    allemb = embed_all(alltexts)
    rng = random.Random(items["seed"] + 0)
    order = list(range(len(alltexts))); rng.shuffle(order)
    trace = order[:4000]
    texts = [alltexts[i] for i in trace]; classes = [pool[i]["cls"] for i in trace]; emb = allemb[trace]
log("embedded")
for p in POLICIES:
    h = stream_hits(texts, classes, emb, p)
    wrong = [x for x in h if not x["ok"]]
    rnd = random.Random(7)
    out[p] = {"hits": len(h), "wrong": len(wrong), "wrong_sample": rnd.sample(wrong, min(120, len(wrong))),
              "right_sample": rnd.sample([x for x in h if x["ok"]], min(40, sum(1 for x in h if x["ok"])))}
    log(f"{p} hits={len(h)} wrong={len(wrong)}")
blob = base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
print("===OUT-BEGIN===", flush=True)
for k in range(0, len(blob), 3000):
    print(blob[k:k + 3000], flush=True)
print("===OUT-END===", flush=True)
