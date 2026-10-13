"""cache-natural-v1 결과 집계. PLAN.md 4절의 지표를 계산해 results/summary.json 에 남긴다.

    python analyze.py results/raw.jsonl
"""
import collections
import json
import math
import random
import sys

D2 = {"C3_trap": 0.72, "C2_trap": 0.98, "C3_para": 0.89, "C2_para": 1.00,
      "auc_cos": 0.683, "auc_rerank": 0.686}


def _pmf(i, n, p):
    return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * math.log(p) + (n - i) * math.log(1 - p))


def cp(k, n, alpha=0.05):
    """Clopper-Pearson 95% 구간. scipy 없이 이항 꼬리확률을 이분 탐색한다."""
    if n == 0:
        return (None, None)

    def bisect(f, increasing, target):
        lo, hi = 1e-12, 1 - 1e-12
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    tail_ge = lambda p: sum(_pmf(i, n, p) for i in range(k, n + 1))   # p 에 대해 증가
    cdf_le = lambda p: sum(_pmf(i, n, p) for i in range(0, k + 1))    # p 에 대해 감소
    lower = 0.0 if k == 0 else bisect(tail_ge, True, alpha / 2)
    upper = 1.0 if k == n else bisect(cdf_le, False, alpha / 2)
    return (lower, upper)


def rate(k, n):
    lo, hi = cp(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def auc(pos, neg):
    """pos(같은 뜻)가 neg(함정)보다 높을 확률. 동점은 0.5."""
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for t in ranked[i:j] if t[1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    variants = [r for r in rows if r["record"] == "variant"]
    contam = {r["id"]: r for r in rows if r["record"] == "contamination"}
    meta = next(r for r in rows if r["record"] == "meta")
    flagged = {gid for gid, r in contam.items() if r["max_cos"] is not None and r["max_cos"] >= 0.95}

    def summarize(vs):
        out = {}
        paras = [v for v in vs if v["kind"] != "trap"]
        traps = [v for v in vs if v["kind"] == "trap"]
        for policy in ("c2", "c3"):
            key = policy + "_hit"
            correct = sum(1 for v in paras if v[key] == v["id"])
            para_wrong = sum(1 for v in paras if v[key] and v[key] != v["id"])
            trap_hit = sum(1 for v in traps if v[key])
            hits = correct + para_wrong + trap_hit
            by_type = {}
            for t, group in sorted(collections.defaultdict(list, {
                    t: [v for v in traps if v["trap_type"] == t]
                    for t in {v["trap_type"] for v in traps}}).items()):
                by_type[t] = rate(sum(1 for v in group if v[key]), len(group))
            by_domain = {}
            for d in sorted({v["domain"] for v in vs}):
                tg = [v for v in traps if v["domain"] == d]
                pg = [v for v in paras if v["domain"] == d]
                by_domain[d] = {"trap_pass": rate(sum(1 for v in tg if v[key]), len(tg)),
                                "para_correct": rate(sum(1 for v in pg if v[key] == v["id"]), len(pg))}
            by_kind = {k: rate(sum(1 for v in paras if v["kind"] == k and v[key] == v["id"]),
                               sum(1 for v in paras if v["kind"] == k))
                       for k in ("para_light", "para_heavy")}
            pc, pw, th = correct / len(paras), para_wrong / len(paras), trap_hit / len(traps)
            reweighted = {}
            for p in (0.05, 0.10, 0.20, 1 / 3, 0.5):
                good = (1 - p) * pc
                bad = (1 - p) * pw + p * th
                reweighted[f"{p:.3f}"] = bad / (good + bad) if good + bad else None
            out[policy] = {
                "para_correct_hit": rate(correct, len(paras)),
                "para_correct_by_kind": by_kind,
                "para_wrong_hit": rate(para_wrong, len(paras)),
                "trap_pass": rate(trap_hit, len(traps)),
                "trap_pass_type_macro": sum(r["rate"] for r in by_type.values()) / len(by_type),
                "trap_pass_by_type": by_type,
                "by_domain": by_domain,
                "hits": hits,
                "false_reuse_among_hits_dataset_mix": (para_wrong + trap_hit) / hits if hits else None,
                "false_reuse_among_hits_by_trap_share": reweighted,
            }
        return out

    def auc_block(vs, field):
        paras = [v[field] for v in vs if v["kind"] != "trap"]
        traps = [v[field] for v in vs if v["kind"] == "trap"]
        return auc(paras, traps)

    def boot(vs, field, reps=2000, seed=20261010):
        by_group = collections.defaultdict(list)
        for v in vs:
            by_group[v["id"]].append(v)
        ids = sorted(by_group)
        rng = random.Random(seed)
        vals = []
        for _ in range(reps):
            sample = [v for gid in (rng.choice(ids) for _ in ids) for v in by_group[gid]]
            vals.append(auc_block(sample, field))
        vals.sort()
        return [vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]]

    separability = {}
    for field, name in (("own_sim", "cosine"), ("own_score", "rerank")):
        separability[name] = {
            "auc_all": auc_block(variants, field), "ci95": boot(variants, field),
            "auc_light_vs_trap": auc_block([v for v in variants if v["kind"] != "para_heavy"], field),
            "auc_heavy_vs_trap": auc_block([v for v in variants if v["kind"] != "para_light"], field),
        }
    traps = [v for v in variants if v["kind"] == "trap"]
    paras = [v for v in variants if v["kind"] != "trap"]
    separability["trap_closer_than_both_paras"] = {
        name: sum(1 for t in traps
                  if all(t[field] >= p[field] for p in paras if p["id"] == t["id"])) / len(traps)
        for field, name in (("own_sim", "cosine"), ("own_score", "rerank"))}
    retrieval = {"own_original_rank1": sum(1 for v in variants if v["own_rank"] == 1) / len(variants),
                 "para_own_rank1": sum(1 for v in paras if v["own_rank"] == 1) / len(paras)}

    summary = {
        "meta": meta,
        "n_groups": len({v["id"] for v in variants}), "n_variants": len(variants),
        "contamination": {"flagged_ge_0.95": sorted(flagged),
                          "max_cos_quantiles": sorted(r["max_cos"] for r in contam.values())[::30]},
        "all": summarize(variants),
        "excluding_flagged": summarize([v for v in variants if v["id"] not in flagged]),
        "separability": separability,
        "retrieval": retrieval,
        "d2_reference": D2,
    }
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("n_groups", "n_variants", "contamination", "separability", "retrieval")},
                     indent=1, ensure_ascii=False))
    for policy in ("c2", "c3"):
        s = summary["all"][policy]
        print(policy, "para_correct", round(s["para_correct_hit"]["rate"], 3), s["para_correct_hit"]["ci95"],
              "| para_wrong", round(s["para_wrong_hit"]["rate"], 3),
              "| trap_pass", round(s["trap_pass"]["rate"], 3), s["trap_pass"]["ci95"],
              "| macro", round(s["trap_pass_type_macro"], 3),
              "| among_hits", round(s["false_reuse_among_hits_dataset_mix"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
