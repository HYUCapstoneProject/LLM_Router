"""NLI 모델 크기별 정확도 비교와 '1위 후보만 검증' 변형. tune.py 와 같은 분할·선택 규칙을 쓴다.

    python tune_small.py
"""
import contextlib
import io
import json
import runpy
import sys

import numpy as np

sys.argv = ["tune.py", "results/raw.jsonl", "results/scores.json"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("tune.py")

variants, calib, test = g["variants"], g["calib"], g["test"]
select, evaluate, quantile_grid, blocked, logit, cp = (g[k] for k in
                                                       ("select", "evaluate", "quantile_grid", "blocked", "logit", "cp"))
small = json.load(open("results/scores_small_nli.json", encoding="utf-8"))
base = json.load(open("results/scores.json", encoding="utf-8"))["scores"]


def nli_from(store, pid, kind):
    s = store[pid]
    if kind == "base":
        ab, ba = s["nli_ab"], s["nli_ba"]
    else:
        ab, ba = s["ab"], s["ba"]
    return min(ab["entailment"], ba["entailment"]), max(ab["contradiction"], ba["contradiction"])


def set_model(name):
    store, kind = (base, "base") if name == "base" else (small[name], "small")
    for v in variants:
        for f in v["cands"] + [v["own"]]:
            f["nli"], f["contra"] = nli_from(store, f"{v['id']}|{v['kind']}|{f['oid']}", kind)


def fit_lr():
    vec = lambda f: [f["cos"], logit(f["quora"]), logit(f["nli"]), logit(f["contra"]), f["lc"], 1.0 if blocked(f) else 0.0]
    X = np.array([vec(v["own"]) for v in calib])
    y = np.array([1.0 if v["is_para"] else 0.0 for v in calib])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.1 * (Z.T @ (p - y) / len(y) + 0.01 * w)
        b -= 0.1 * float(np.mean(p - y))
    for v in variants:
        for f in v["cands"]:
            z = (np.array(vec(f)) - mu) / sd
            f["lr"] = float(1 / (1 + np.exp(-(z @ w + b))))


def run(label, top1=False):
    if top1:
        saved = {id(v): v["cands"] for v in variants}
        for v in variants:
            v["cands"] = v["cands"][:1]          # 코사인 1위 후보만 검증
    out = {}
    for mname, (make, grid) in {
        "NLI": (lambda t: (lambda f: f["nli"] >= t, lambda f: f["nli"]), [(t,) for t in quantile_grid("nli")]),
        "NLI+LR": (lambda t: (lambda f: f["lr"] >= t, lambda f: f["lr"]), [(t,) for t in quantile_grid("lr", 200)]),
    }.items():
        best = select(make, grid)
        if best is None:
            out[mname] = None
            continue
        _, params, cal = best
        te = evaluate(test, *make(*params))
        c = te["counts"]
        out[mname] = {"param": params[0], "para": te["para_correct"], "para_ci": cp(c["pc"], c["n_para"]),
                      "trap": te["trap_pass"], "trap_ci": cp(c["th"], c["n_trap"]), "para_wrong": te["para_wrong"]}
    if top1:
        for v in variants:
            v["cands"] = saved[id(v)]
    return out


results = {}
for name in ["base"] + list(small):
    set_model(name)
    fit_lr()
    results[name] = run(name)
set_model("base")
fit_lr()
results["base (1위 후보만)"] = run("base", top1=True)
set_model("cross-encoder/nli-deberta-v3-xsmall")
fit_lr()
results["xsmall (1위 후보만)"] = run("xsmall", top1=True)

json.dump(results, open("results/tuning_small.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print(f"{'model':40s} | {'NLI 단독: 같은뜻 / 함정':28s} | NLI+LR: 같은뜻 / 함정")
for name, r in results.items():
    cells = []
    for m in ("NLI", "NLI+LR"):
        x = r[m]
        cells.append("기준 없음" if x is None else
                     f"{x['para']:.3f} [{x['para_ci'][0]:.2f},{x['para_ci'][1]:.2f}] / {x['trap']:.3f}")
    print(f"{name.split('/')[-1]:40s} | {cells[0]:28s} | {cells[1]}")
