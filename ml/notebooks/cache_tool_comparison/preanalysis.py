"""Pre-analysis behind the cache tool comparison plan (docs/ml/CACHE_TOOL_COMPARISON.md section 2).

Reads only sealed run001 artifacts of the D2 *calibration* split; the test split is
not touched. No model is loaded: cosine scores come from the sealed BGE embeddings
and reranker scores from calibration-rerank-scores.npy (top-16 candidates).

Usage: python preanalysis.py <runs/run001 directory> [--out preanalysis.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Token classes whose change can flip the answer. Deliberately crude: this is the
# cheapest possible guard, measured to see whether rules alone could be enough.
CRITICAL = {
    "number": lambda t: bool(re.fullmatch(r"\d+(?:[.,]\d+)?", t)) or t in set(
        "zero one two three four five six seven eight nine ten eleven twelve twenty hundred thousand "
        "million half twice double triple percent dozen".split()),
    "negation": lambda t: t in set(
        "not no never none nothing nor without except n't cannot isn't aren't doesn't don't didn't "
        "won't wasn't weren't false incorrect least fewest unless neither".split()),
    "order": lambda t: t in set(
        "first second third last before after earlier later earliest latest largest smallest biggest "
        "highest lowest most fewer more less greater ascending descending reverse previous next final initial".split()),
    "time": lambda t: t in set(
        "year years month months week weeks day days today yesterday tomorrow hour hours minute minutes ago "
        "future past now currently century decade january february march april may june july august "
        "september october november december monday tuesday wednesday thursday friday saturday sunday "
        "morning evening night".split()),
    "code_language": lambda t: t in set(
        "python java javascript typescript c cpp rust go golang ruby php swift kotlin scala sql bash "
        "haskell perl r matlab".split()),
    "output_format": lambda t: t in set(
        "json csv table list bullet bullets yaml xml markdown sentence sentences word words paragraph "
        "paragraphs uppercase lowercase letter letters".split()),
}


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?|[^\sa-z\d]", text.lower())


def changed_classes(a, b):
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    diff = (ca - cb) + (cb - ca)
    return sorted({name for name, test in CRITICAL.items() for t in diff if test(t)})


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    r = args.run_dir

    pool = pd.read_parquet(r / "prepare/pool.parquet")
    vectors = np.load(r / "prepare/embeddings.npy")
    originals = pd.read_parquet(r / "d2/originals.parquet")
    labels = pd.read_parquet(r / "d2/annotations.parquet")
    items = labels[labels.d2_split.eq("calibration")].reset_index(drop=True)

    # Same cache as pipeline._d2_split_inputs: the split's originals, then the D1 pool.
    taken = set(originals.sample_id)
    d1_pool = pool[~pool.sample_id.isin(taken)].reset_index(drop=True)
    cache = pd.concat([originals[originals.d2_split.eq("calibration")][["sample_id"]], d1_pool[["sample_id"]]],
                      ignore_index=True)
    row = {s: i for i, s in enumerate(pool.sample_id)}
    cache_vectors = vectors[[row[s] for s in cache.sample_id]]
    query_vectors = np.load(r / "d2/calibration-variant-embeddings.npy")
    rerank = np.load(r / "d2/calibration-rerank-scores.npy")
    sims = query_vectors @ cache_vectors.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :16]
    position = {s: i for i, s in enumerate(cache.sample_id)}

    own_cos, own_rerank, own_rank = [], [], []
    for i, item in enumerate(items.itertuples(index=False)):
        j = position[item.original_sample_id]
        own_cos.append(float(sims[i, j]))
        found = np.flatnonzero(order[i] == j)
        own_rank.append(int(found[0]) if found.size else None)
        own_rerank.append(float(rerank[i, found[0]]) if found.size else float("nan"))
    items = items.assign(cos=own_cos, rerank=own_rerank, rank=own_rank)
    request_text = originals.set_index("sample_id").request_text
    items["changed"] = [changed_classes(request_text[s], v) for s, v in zip(items.original_sample_id, items.variant_text)]
    items["guard_blocks"] = items.changed.str.len() > 0

    equivalent = items.semantic_equivalent.eq("Y")
    reusable = equivalent & items.cached_answer_correct_for_target.eq("Y")
    wide = items.pivot(index="group_id", columns="role", values=["cos", "rerank", "semantic_equivalent"])
    groups = wide[wide["semantic_equivalent"]["paraphrase"].eq("Y")]
    traps = items[items.role.eq("hard_negative")]

    result = {
        "split": "calibration",
        "items": int(len(items)),
        "retrieval": {"own_original_top1": float(items["rank"].eq(0).mean()),
                      "own_original_in_top16": float(items["rank"].notna().mean())},
        "separability": {
            "auc_cosine_equivalent_vs_not": auc(items.cos[equivalent], items.cos[~equivalent]),
            "auc_rerank_equivalent_vs_not": auc(items.rerank[equivalent].fillna(0), items.rerank[~equivalent].fillna(0)),
            "groups_with_equivalent_paraphrase": int(len(groups)),
            "groups_trap_cosine_ge_paraphrase": float((groups["cos"]["hard_negative"] >= groups["cos"]["paraphrase"]).mean()),
            "groups_trap_rerank_ge_paraphrase": float((groups["rerank"]["hard_negative"] >= groups["rerank"]["paraphrase"]).mean()),
            "traps_cosine_ge_0_99": float((traps.cos >= 0.99).mean()),
            "paraphrases_cosine_ge_0_99": float((items[items.role.eq("paraphrase")].cos >= 0.99).mean()),
            "by_role": items.groupby("role")[["cos", "rerank"]].median().round(4).to_dict(),
        },
        "keyword_guard": {
            "blocks_traps": float(traps.guard_blocks.mean()),
            "blocks_equivalent_paraphrases": float(items[items.role.eq("paraphrase") & equivalent].guard_blocks.mean()),
            "blocks_reusable": float(items[reusable].guard_blocks.mean()),
            "blocks_traps_by_type": traps.groupby("negative_type").guard_blocks.mean().round(3).to_dict(),
            "paraphrase_block_reasons": dict(Counter(c for cl in items[items.role.eq("paraphrase")].changed for c in cl)),
        },
        # EscalationService.refreshCache overwrites the hit row's answer when the hit's
        # cosine >= llm.escalation.cache-refresh.overwrite-similarity (0.95).
        "escalation_overwrite": {
            "overwrite_similarity": 0.95,
            "traps_c3_hit_and_ge_overwrite": float(((traps.cos >= 0.95) & (traps.cos >= 0.8774)
                                                    & (traps.rerank >= 0.03581769)).mean()),
            "traps_ge_overwrite": float((traps.cos >= 0.95).mean()),
        },
        "most_similar_traps": [
            {"group_id": t.group_id, "negative_type": t.negative_type, "cosine": round(t.cos, 5), "rerank": round(t.rerank, 4)}
            for t in traps.sort_values("cos", ascending=False).head(3).itertuples()],
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
