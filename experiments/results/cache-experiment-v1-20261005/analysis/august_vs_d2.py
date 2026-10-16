"""Compare the August reranker evidence with run001 D2 (report section 7).

1. Input length: BGE tokens of every D2 original and variant, and the reranker's pair
   length (variant + original, which the cross-encoder reads as one input truncated
   at 512). For truncated pairs, whether the changed span survives truncation.
2. Changed share: changed reranker tokens / tokens of the longer question, for the
   August 58 pairs (experiments/results/reranker-20260830, quora scores) and the D2
   calibration 200 pairs (scores from the sealed 03 run). Pass = operational C3 rule.

Only token counts and scores are written; no question text. Test-split items are read
only for their lengths (section 1); scores come from the calibration split.
Usage: python august_vs_d2.py <runs/run001> <repo root> <out.json>
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

EMBED = ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09")
RERANK = ("cross-encoder/quora-distilroberta-base", "f62e7a4b20b97195c2868e53ec59126df5eac743")
COS_GATE, SCORE_GATE, LIMIT = 0.8774, 0.03581769, 512
SHARE_BINS = [0, 0.05, 0.10, 0.20, 0.40, float("inf")]  # share can exceed 1 when replaced spans differ in length

etok = AutoTokenizer.from_pretrained(EMBED[0], revision=EMBED[1])
rtok = AutoTokenizer.from_pretrained(RERANK[0], revision=RERANK[1])


def ids(text):
    return rtok(text, add_special_tokens=False)["input_ids"]


def kept(x, y, limit=LIMIT):
    """Segment lengths after transformers' longest_first truncation (4 special tokens)."""
    x, y = len(x), len(y)
    while x + y > limit - 4:
        if x > y:
            x -= 1
        else:
            y -= 1
    return x, y


def edit_visibility(x, y):
    kx, ky = kept(x, y)
    ops = [o for o in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not ops:
        return "identical"
    start_x, start_y = ops[0][1], ops[0][3]
    end_x, end_y = ops[-1][2], ops[-1][4]
    hidden_x, hidden_y = start_x >= kx, start_y >= ky
    if hidden_x and hidden_y:
        return "hidden_both"
    if hidden_x or hidden_y:
        return "hidden_one"
    return "partly_cut" if end_x > kx or end_y > ky else "visible"


def changed_share(a, b):
    x, y = ids(a), ids(b)
    ops = difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    return changed / max(len(x), len(y)), changed, max(len(x), len(y))


def sealed_calibration(run, items):
    """Cosine and reranker score of each calibration variant against its own original (sealed 03 run)."""
    pool = pd.read_parquet(run / "prepare/pool.parquet")
    vectors = np.load(run / "prepare/embeddings.npy")
    originals = pd.read_parquet(run / "d2/originals.parquet")
    d1 = pool[~pool.sample_id.isin(set(originals.sample_id))]
    cache = pd.concat([originals[originals.d2_split.eq("calibration")][["sample_id"]], d1[["sample_id"]]],
                      ignore_index=True)
    row = {s: i for i, s in enumerate(pool.sample_id)}
    sims = np.load(run / "d2/calibration-variant-embeddings.npy") @ vectors[[row[s] for s in cache.sample_id]].T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :16]
    scores = np.load(run / "d2/calibration-rerank-scores.npy")
    position = {s: i for i, s in enumerate(cache.sample_id)}
    cos = [float(sims[i, position[s]]) for i, s in enumerate(items.original_sample_id)]
    rerank = [float(scores[i, np.flatnonzero(order[i] == position[s])[0]]) for i, s in enumerate(items.original_sample_id)]
    return cos, rerank


def main():
    run, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    originals = pd.read_parquet(run / "d2/originals.parquet").set_index("sample_id")
    labels = pd.read_parquet(run / "d2/annotations.parquet")
    labels["original_text"] = originals.request_text.reindex(labels.original_sample_id).values

    # 1. Lengths
    bge_variant = [len(etok(t, truncation=False)["input_ids"]) for t in labels.variant_text]
    bge_original = [len(etok(t, truncation=False)["input_ids"]) for t in originals.request_text]
    pairs = [(ids(v), ids(o)) for v, o in zip(labels.variant_text, labels.original_text)]
    labels["pair_tokens"] = [len(x) + len(y) + 4 for x, y in pairs]
    labels["truncated"] = labels.pair_tokens > LIMIT
    labels["edit"] = [edit_visibility(x, y) for x, y in pairs]
    lengths = {
        "bge": {"variant_max": int(max(bge_variant)), "variant_over_512": int(sum(n > LIMIT for n in bge_variant)),
                "original_max": int(max(bge_original)), "original_over_512": int(sum(n > LIMIT for n in bge_original)),
                "variant_median": float(np.median(bge_variant))},
        "reranker_pair": {
            "pairs": int(len(labels)), "over_512": int(labels.truncated.sum()),
            "median_tokens": float(labels.pair_tokens.median()), "max_tokens": int(labels.pair_tokens.max()),
            "over_512_by_stratum": labels[labels.truncated].groupby("stratum").size().to_dict(),
            "edit_visibility_by_role": pd.crosstab(labels.role, labels.edit).to_dict("index")},
    }

    # 2. Changed share
    august = pd.read_csv(repo / "experiments/results/reranker-20260830/quora-distilroberta/"
                                "cross-encoder--quora-distilroberta-base.csv")
    # The cache holds the base ("query" column); the incoming request is the variant.
    aug = pd.DataFrame({"dataset": "august", "label": august.label.map({"positive": "same", "negative": "trap"}),
                        "a": august.candidate, "b": august["query"], "cosine": august.bi_encoder_similarity,
                        "rerank": august.reranker_score})
    cal = labels[labels.d2_split.eq("calibration")].reset_index(drop=True)
    cos, rerank = sealed_calibration(run, cal)
    d2 = pd.DataFrame({"dataset": "d2", "label": cal.role.map({"paraphrase": "same", "hard_negative": "trap"}),
                       "a": cal.variant_text, "b": cal.original_text, "cosine": cos, "rerank": rerank,
                       "truncated": cal.truncated})
    frame = pd.concat([aug, d2], ignore_index=True)
    shares = [changed_share(a, b) for a, b in zip(frame.a, frame.b)]
    frame["share"] = [s[0] for s in shares]
    frame["changed_tokens"] = [s[1] for s in shares]
    frame["tokens"] = [s[2] for s in shares]
    frame["passes"] = (frame.cosine >= COS_GATE) & (frame.rerank >= SCORE_GATE)

    summary = frame.groupby(["dataset", "label"]).agg(
        pairs=("share", "size"), tokens_median=("tokens", "median"), changed_tokens_median=("changed_tokens", "median"),
        share_median=("share", "median"), share_q25=("share", lambda s: s.quantile(.25)),
        share_q75=("share", lambda s: s.quantile(.75)), rerank_median=("rerank", "median"),
        pass_rate=("passes", "mean")).reset_index()
    traps = frame[frame.label.eq("trap")].copy()
    traps["share_bin"] = pd.cut(traps.share, SHARE_BINS, include_lowest=True).astype(str)
    by_bin = traps.groupby(["share_bin", "dataset"]).agg(
        pairs=("passes", "size"), pass_rate=("passes", "mean"), rerank_median=("rerank", "median")).reset_index()
    d2_traps = traps[traps.dataset.eq("d2")]
    by_truncation = d2_traps.groupby("truncated").agg(
        pairs=("passes", "size"), pass_rate=("passes", "mean"), rerank_median=("rerank", "median")).reset_index()

    result = {"rule": {"cosine": COS_GATE, "rerank": SCORE_GATE, "limit": LIMIT},
              "models": {"embed": EMBED, "rerank": RERANK},
              "lengths": lengths,
              "share_summary": summary.round(4).to_dict("records"),
              "traps_by_share_bin": by_bin.round(4).to_dict("records"),
              "d2_calibration_traps_by_pair_truncation": by_truncation.round(4).to_dict("records")}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
