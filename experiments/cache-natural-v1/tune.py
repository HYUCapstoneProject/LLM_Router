"""PLAN2.md 의 판정기 비교. calibration 에서 기준을 고르고 test 에서 한 번 평가한다.

    python tune.py results/raw.jsonl results/scores.json
"""
import hashlib
import json
import math
import sys

import numpy as np

from analyze import cp
from guard import RULES, check

TRAP_LIMIT = 0.05
PROD_COS, PROD_Q = 0.8774, 0.03581769

groups = {json.loads(l)["id"]: json.loads(l) for l in open("data/dataset.jsonl", encoding="utf-8")}
raw = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
variants = [v for v in raw if v["record"] == "variant"]
scores = json.load(open(sys.argv[2], encoding="utf-8"))
score_meta, scores = scores["meta"], scores["scores"]


def is_calib(gid):
    return hashlib.sha256(gid.encode()).digest()[0] % 3 == 0


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def features(v, oid, cos, quora):
    s = scores[f"{v['id']}|{v['kind']}|{oid}"]
    ent = min(s["nli_ab"]["entailment"], s["nli_ba"]["entailment"])
    con = max(s["nli_ab"]["contradiction"], s["nli_ba"]["contradiction"])
    return {"oid": oid, "cos": cos, "quora": quora, "nli": ent, "contra": con, "lc": s["langcache_cos"],
            "guard": check(groups[v["id"]][v["kind"]], groups[oid]["original"])}


for v in variants:
    v["calib"] = is_calib(v["id"])
    v["is_para"] = v["kind"] != "trap"
    v["cands"] = [features(v, oid, cos, q) for oid, cos, q in v["candidates"]]
    v["own"] = features(v, v["id"], v["own_sim"], v["own_score"])

calib = [v for v in variants if v["calib"]]
test = [v for v in variants if not v["calib"]]

# 1. 규칙 선택: calibration 의 자기 원질문 쌍에서 함정 차단이 같은 뜻 차단의 3배 이상인 규칙만 켠다
rule_stats = {}
for r in RULES:
    tb = sum(1 for v in calib if not v["is_para"] and v["own"]["guard"][r])
    pb = sum(1 for v in calib if v["is_para"] and v["own"]["guard"][r])
    rule_stats[r] = {"traps_blocked": tb, "paras_blocked": pb, "enabled": tb >= 1 and tb >= 3 * pb}
ENABLED = [r for r in RULES if rule_stats[r]["enabled"]]


def blocked(f):
    return any(f["guard"][r] for r in ENABLED)


def decide(v, accept, score):
    ok = [f for f in v["cands"] if accept(f)]
    return max(ok, key=score)["oid"] if ok else None


def evaluate(vs, accept, score):
    pc = pw = th = 0
    n_para = sum(1 for v in vs if v["is_para"])
    n_trap = len(vs) - n_para
    for v in vs:
        hit = decide(v, accept, score)
        if v["is_para"]:
            pc += hit == v["id"]
            pw += hit is not None and hit != v["id"]
        else:
            th += hit is not None
    return {"para_correct": pc / n_para, "para_wrong": pw / n_para, "trap_pass": th / n_trap,
            "counts": {"pc": pc, "pw": pw, "th": th, "n_para": n_para, "n_trap": n_trap}}


def select(make, grid):
    """함정 통과율 ≤ 5% 안에서 같은 뜻 재사용 최대(calibration)."""
    best = None
    for params in grid:
        acc, sc = make(*params)
        m = evaluate(calib, acc, sc)
        if m["trap_pass"] > TRAP_LIMIT:
            continue
        key = (m["para_correct"], -m["trap_pass"], -m["para_wrong"])
        if best is None or key > best[0]:
            best = (key, params, m)
    return best


def quantile_grid(field, n=120):
    vals = sorted({f[field] for v in calib for f in v["cands"]})
    if len(vals) <= n:
        return vals
    return [vals[int(i * (len(vals) - 1) / (n - 1))] for i in range(n)]


methods = {}
methods["M0 운영 C3"] = ((lambda: (lambda f: f["cos"] >= PROD_COS and f["quora"] >= PROD_Q, lambda f: f["quora"])), [()])
methods["M0+G 운영 C3 + 조건 규칙"] = ((lambda: (lambda f: f["cos"] >= PROD_COS and f["quora"] >= PROD_Q and not blocked(f),
                                             lambda f: f["quora"])), [()])
cos_grid = [round(0.80 + 0.0025 * i, 4) for i in range(77)]
q_grid = [0.0] + quantile_grid("quora", 60)
methods["M1 C3 재튜닝"] = ((lambda c, q: (lambda f: f["cos"] >= c and f["quora"] >= q, lambda f: f["quora"])),
                       [(c, q) for c in cos_grid for q in q_grid])
methods["M2 NLI 양방향 함의"] = ((lambda t: (lambda f: f["nli"] >= t, lambda f: f["nli"])),
                           [(t,) for t in quantile_grid("nli")])
methods["M3 LangCache 임베딩"] = ((lambda t: (lambda f: f["lc"] >= t, lambda f: f["lc"])),
                              [(t,) for t in quantile_grid("lc")])
methods["M4 조건 규칙 + C3 재튜닝"] = ((lambda c, q: (lambda f: f["cos"] >= c and f["quora"] >= q and not blocked(f),
                                                lambda f: f["quora"])), [(c, q) for c in cos_grid for q in q_grid])
methods["M5 조건 규칙 + NLI"] = ((lambda t: (lambda f: f["nli"] >= t and not blocked(f), lambda f: f["nli"])),
                           [(t,) for t in quantile_grid("nli")])
methods["M6 조건 규칙 + LangCache"] = ((lambda t: (lambda f: f["lc"] >= t and not blocked(f), lambda f: f["lc"])),
                                  [(t,) for t in quantile_grid("lc")])

# M7: calibration 자기 원질문 쌍으로 로지스틱 회귀
def vec(f):
    return [f["cos"], logit(f["quora"]), logit(f["nli"]), logit(f["contra"]), f["lc"], 1.0 if blocked(f) else 0.0]


X = np.array([vec(v["own"]) for v in calib])
y = np.array([1.0 if v["is_para"] else 0.0 for v in calib])
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
w, b = np.zeros(Z.shape[1]), 0.0
for _ in range(5000):
    p = 1 / (1 + np.exp(-(Z @ w + b)))
    w -= 0.1 * (Z.T @ (p - y) / len(y) + 0.01 * w)
    b -= 0.1 * float(np.mean(p - y))


def lr_prob(f):
    z = (np.array(vec(f)) - mu) / sd
    return float(1 / (1 + np.exp(-(z @ w + b))))


for v in variants:
    for f in v["cands"]:
        f["lr"] = lr_prob(f)
methods["M7 로지스틱 결합"] = ((lambda t: (lambda f: f["lr"] >= t, lambda f: f["lr"])),
                         [(t,) for t in quantile_grid("lr", 200)])

results = {}
for name, (make, grid) in methods.items():
    if grid == [()]:
        # 고정 기준(운영 그대로)은 고르지 않으므로 함정 상한을 적용하지 않는다
        acc0, sc0 = make()
        best = (None, (), evaluate(calib, acc0, sc0))
    else:
        best = select(make, grid)
    if best is None:
        results[name] = {"selected": None, "note": "calibration 에서 함정 ≤ 5% 를 만족하는 기준 없음"}
        continue
    _, params, cal = best
    acc, sc = make(*params)
    te = evaluate(test, acc, sc)
    c = te["counts"]
    pc, th, pw = te["para_correct"], te["trap_pass"], te["para_wrong"]
    good, bad = 0.9 * pc, 0.9 * pw + 0.1 * th
    results[name] = {
        "params": params, "calib": {k: cal[k] for k in ("para_correct", "para_wrong", "trap_pass")},
        "test": {"para_correct": pc, "para_correct_ci": cp(c["pc"], c["n_para"]),
                 "para_wrong": pw, "trap_pass": th, "trap_pass_ci": cp(c["th"], c["n_trap"]),
                 "false_reuse_among_hits_at_10pct_traps": bad / (good + bad) if good + bad else None,
                 "counts": c}}

out = {"split": {"calib_groups": len({v["id"] for v in calib}), "test_groups": len({v["id"] for v in test})},
       "trap_limit": TRAP_LIMIT, "rules": rule_stats, "enabled_rules": ENABLED,
       "lr": {"features": ["cos", "logit_quora", "logit_nli_entail", "logit_nli_contra", "langcache_cos", "guard"],
              "weights": w.round(3).tolist(), "bias": round(b, 3)},
       "score_models": score_meta, "methods": results}
json.dump(out, open("results/tuning.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

print("split", out["split"], "| enabled rules:", ENABLED)
for r, s in rule_stats.items():
    print(f"  rule {r:12s} traps {s['traps_blocked']:3d} paras {s['paras_blocked']:3d} {'ON' if s['enabled'] else ''}")
print(f"\n{'method':28s} {'params':22s} | calib para/trap | test para [CI]          wrong  trap [CI]            hit중오재사용@10%")
for name, r in results.items():
    if r.get("params") is None and "test" not in r:
        print(f"{name:28s} {r['note']}")
        continue
    t = r["test"]
    ps = ",".join(f"{p:.4g}" for p in r["params"]) or "-"
    print(f"{name:28s} {ps:22s} | {r['calib']['para_correct']:.2f}/{r['calib']['trap_pass']:.2f}   "
          f"| {t['para_correct']:.3f} [{t['para_correct_ci'][0]:.2f},{t['para_correct_ci'][1]:.2f}] "
          f"{t['para_wrong']:.3f}  {t['trap_pass']:.3f} [{t['trap_pass_ci'][0]:.2f},{t['trap_pass_ci'][1]:.2f}]  "
          f"{t['false_reuse_among_hits_at_10pct_traps']:.3f}")
