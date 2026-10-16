"""Cloud Build 안에서 NLI·LangCache 점수를 계산한다. 로컬 디스크를 쓰지 않기 위한 실행 경로다.

입력 pairs.json: [{"pid": str, "a": 변형, "b": 원질문}]
출력 out/scores.json: {pid: {"nli_ab": {...}, "nli_ba": {...}, "langcache_cos": float}}
"""
import json
import os
import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
LANGCACHE_MODEL = "redis/langcache-embed-v3-small"

pairs = json.load(open("pairs.json", encoding="utf-8"))
os.makedirs("out", exist_ok=True)
t0 = time.time()

nli = CrossEncoder(NLI_MODEL, device="cpu")
labels = [nli.model.config.id2label[i].lower() for i in range(len(nli.model.config.id2label))]
ab = nli.predict([(p["a"], p["b"]) for p in pairs], apply_softmax=True, batch_size=32, show_progress_bar=False)
ba = nli.predict([(p["b"], p["a"]) for p in pairs], apply_softmax=True, batch_size=32, show_progress_bar=False)
print(f"nli done {time.time() - t0:.0f}s labels={labels}", flush=True)

emb_model = SentenceTransformer(LANGCACHE_MODEL, device="cpu")
texts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
vecs = emb_model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
index = {t: v for t, v in zip(texts, vecs)}
print(f"langcache done {time.time() - t0:.0f}s dim={vecs.shape[1]}", flush=True)

out = {}
for p, x, y in zip(pairs, ab, ba):
    out[p["pid"]] = {
        "nli_ab": {lab: float(v) for lab, v in zip(labels, x)},
        "nli_ba": {lab: float(v) for lab, v in zip(labels, y)},
        "langcache_cos": float(np.dot(index[p["a"]], index[p["b"]])),
    }
meta = {"nli_model": NLI_MODEL, "langcache_model": LANGCACHE_MODEL, "labels": labels,
        "n_pairs": len(pairs), "elapsed_s": round(time.time() - t0, 1)}
json.dump({"meta": meta, "scores": out}, open("out/scores.json", "w", encoding="utf-8"))
print(json.dumps(meta), flush=True)

# 버킷 없이 결과를 받기 위해 압축·인코딩해 로그로 내보낸다.
import base64
import gzip

blob = base64.b64encode(gzip.compress(open("out/scores.json", "rb").read())).decode()
print("===SCORES-BEGIN===", flush=True)
for i in range(0, len(blob), 3000):
    print(blob[i:i + 3000], flush=True)
print("===SCORES-END===", flush=True)
