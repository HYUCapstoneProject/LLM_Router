"""D2 재현본 채점 (Cloud Build). 운영 검색을 재현하고 후보마다 판정 점수를 계산한다.

1) 패리티: 대화형 데이터 300쌍에서 운영 FastAPI 점수(BGE 코사인, quora)와 이 환경 점수를 비교한다.
2) 캐시 = 원질문 300 + 배경 4,986. 변형 600개마다 BGE 상위 5개 중 코사인 >= 0.80 후보를 뽑는다.
3) 후보와 자기 원질문 쌍마다 quora, NLI base·xsmall(양방향), LangCache 코사인을 계산한다.
결과는 버킷 없이 받기 위해 gzip+base64 로 로그에 출력한다.
"""
import base64
import gzip
import json
import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

BGE = ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09")
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

groups = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]
background = json.load(open("background.json", encoding="utf-8"))
parity = json.load(open("parity.json", encoding="utf-8"))

# ── BGE 임베딩 ──
bge = SentenceTransformer(BGE[0], revision=BGE[1], device="cpu")
bge.max_seq_length = 512
cache_ids = [f"o:{g['gid']}" for g in groups] + [f"b:{i}" for i in range(len(background))]
cache_texts = [g["original"] for g in groups] + background
cache_vec = bge.encode(cache_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
log(f"cache embedded {len(cache_texts)}")
var_items = [(g["gid"], k, g[k]) for g in groups for k in ("para", "trap")]
var_vec = bge.encode([t for _, _, t in var_items], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
log("variants embedded")
pa = bge.encode([p["a"] for p in parity], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
pb = bge.encode([p["b"] for p in parity], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
par_cos = (pa * pb).sum(1)
bge_tok = bge.tokenizer
over512 = sum(1 for t in cache_texts[:len(groups)] + [t for _, _, t in var_items]
              if len(bge_tok(t)["input_ids"]) > 512)

# ── 검색 ──
text_of = dict(zip(cache_ids, cache_texts))
sims = var_vec @ cache_vec.T
records, pairs = [], {}
for i, (gid, kind, text) in enumerate(var_items):
    order = np.argsort(-sims[i])[:5]
    cands = [(cache_ids[j], float(sims[i, j])) for j in order if sims[i, j] >= 0.80]
    own = f"o:{gid}"
    own_cos = float(sims[i, cache_ids.index(own)])
    rank = int((sims[i] > own_cos).sum()) + 1
    records.append({"gid": gid, "kind": kind, "own_cos": own_cos, "own_rank": rank, "candidates": cands})
    for cid, _ in cands + [(own, own_cos)]:
        pairs[(text, cid)] = None
pair_list = list(pairs)
log(f"retrieval done, pairs {len(pair_list)}")

# ── 판정 점수 ──
def ce_scores(name, softmax):
    m = CrossEncoder(name, device="cpu")
    ab = m.predict([(q, text_of[c]) for q, c in pair_list], batch_size=16, show_progress_bar=False,
                   apply_softmax=softmax)
    out = {"ab": ab}
    if softmax:
        out["ba"] = m.predict([(text_of[c], q) for q, c in pair_list], batch_size=16, show_progress_bar=False,
                              apply_softmax=True)
        out["labels"] = [m.model.config.id2label[k].lower() for k in range(len(m.model.config.id2label))]
    tok = m.tokenizer
    out["truncated"] = sum(1 for q, c in pair_list if len(tok(q, text_of[c])["input_ids"]) > 512)
    return m, out

quora_model, quora = ce_scores("cross-encoder/quora-distilroberta-base", False)
par_quora = quora_model.predict([(p["a"], p["b"]) for p in parity], batch_size=32, show_progress_bar=False)
log("quora done")
_, nli_b = ce_scores("cross-encoder/nli-deberta-v3-base", True)
log("nli base done")
_, nli_x = ce_scores("cross-encoder/nli-deberta-v3-xsmall", True)
log("nli xsmall done")
lc = SentenceTransformer("redis/langcache-embed-v3-small", device="cpu")
u = lc.encode([q for q, _ in pair_list], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
v = lc.encode([text_of[c] for _, c in pair_list], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
lc_cos = (u * v).sum(1)
log("langcache done")

def nli_dict(store, i):
    lab = store["labels"]
    ab, ba = dict(zip(lab, map(float, store["ab"][i]))), dict(zip(lab, map(float, store["ba"][i])))
    return {"ent_ab": ab["entailment"], "ent_ba": ba["entailment"], "con_ab": ab["contradiction"], "con_ba": ba["contradiction"]}

pair_scores = {f"{q}␟{c}": {"quora": float(quora["ab"][i]), "nli_b": nli_dict(nli_b, i), "nli_x": nli_dict(nli_x, i),
                                 "lc": float(lc_cos[i])} for i, (q, c) in enumerate(pair_list)}
for (gid, kind, text), r in zip(var_items, records):
    r["text"] = text
out = {
    "meta": {"bge": BGE, "n_cache": len(cache_texts), "n_pairs": len(pair_list), "bge_over_512": over512,
             "truncated_pairs": {"quora": quora["truncated"], "nli_base": nli_b["truncated"], "nli_xsmall": nli_x["truncated"]},
             "elapsed_s": round(time.time() - t0)},
    "parity": {"cos_max_abs_diff": float(np.max(np.abs(par_cos - np.array([p["prod_cos"] for p in parity])))),
               "quora_max_abs_diff": float(np.max(np.abs(par_quora - np.array([p["prod_quora"] for p in parity])))),
               "n": len(parity)},
    "records": records, "pair_scores": pair_scores,
}
log(json.dumps(out["meta"]) + " " + json.dumps(out["parity"]))
blob = base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
print("===OUT-BEGIN===", flush=True)
for k in range(0, len(blob), 3000):
    print(blob[k:k + 3000], flush=True)
print("===OUT-END===", flush=True)
