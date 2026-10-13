"""D2 재현본 평가. 대화형 데이터(cache-natural-v1)에서 고른 판정 기준을 고정 적용한다(주 결과).
보조로 재현본 calibration 100그룹에서 기준을 다시 고르고 test 200그룹에서 평가한다(D2와 같은 절차).

    python evaluate.py results/out.json
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "../cache-natural-v1")
from analyze import auc, cp          # noqa: E402
from guard import check              # noqa: E402

out = json.load(open(sys.argv[1], encoding="utf-8"))
groups = {json.loads(l)["gid"]: json.loads(l) for l in open("data/dataset.jsonl", encoding="utf-8")}
background = json.load(open("data/background.json", encoding="utf-8"))
F = json.load(open("frozen_judges.json", encoding="utf-8"))
RULES = F["guard_rules"]
SEP = "␟"


def text_of(cid):
    kind, key = cid.split(":", 1)
    return groups[key]["original"] if kind == "o" else background[int(key)]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def feat(text, cid, cos):
    s = out["pair_scores"][f"{text}{SEP}{cid}"]
    f = {"cid": cid, "cos": cos, "quora": s["quora"], "lc": s["lc"],
         "guard": any(check(text, text_of(cid))[r] for r in RULES)}
    for key in ("nli_b", "nli_x"):
        n = s[key]
        f[key] = min(n["ent_ab"], n["ent_ba"])
        f[key + "_con"] = max(n["con_ab"], n["con_ba"])
    for key, mdl in (("lr_b", "base"), ("lr_x", "xsmall")):
        p = F[mdl]["lr"]
        nli = "nli_b" if mdl == "base" else "nli_x"
        x = np.array([f["cos"], logit(f["quora"]), logit(f[nli]), logit(f[nli + "_con"]), f["lc"], 1.0 if f["guard"] else 0.0])
        z = (x - np.array(p["mu"])) / np.array(p["sd"])
        f[key] = float(1 / (1 + np.exp(-(z @ np.array(p["w"]) + p["b"]))))
    return f


V = []
for r in out["records"]:
    g = groups[r["gid"]]
    v = {**r, "split": g["split"], "stratum": g["stratum"], "trap_type": g["trap_type"], "is_para": r["kind"] == "para"}
    v["cands"] = [feat(r["text"], cid, cos) for cid, cos in r["candidates"]]
    v["own"] = feat(r["text"], f"o:{r['gid']}", r["own_cos"])
    V.append(v)

JUDGES = {
    "C2 코사인 0.80 (GPTCache 기본과 같음)": (lambda f: f["cos"] >= F["c2_cos"], "cos", False),
    "C3 운영 (코사인+quora)": (lambda f: f["cos"] >= 0.8774 and f["quora"] >= 0.03581769, "quora", False),
    "NLI base": (lambda f: f["nli_b"] >= F["base"]["nli_all"], "nli_b", False),
    "NLI base, 1위만": (lambda f: f["nli_b"] >= F["base"]["nli_top1"], "nli_b", True),
    "NLI xsmall": (lambda f: f["nli_x"] >= F["xsmall"]["nli_all"], "nli_x", False),
    "NLI xsmall, 1위만": (lambda f: f["nli_x"] >= F["xsmall"]["nli_top1"], "nli_x", True),
    "LR+NLI base": (lambda f: f["lr_b"] >= F["base"]["lr_all"], "lr_b", False),
    "LR+NLI base, 1위만": (lambda f: f["lr_b"] >= F["base"]["lr_top1"], "lr_b", True),
    "LR+NLI xsmall, 1위만": (lambda f: f["lr_x"] >= F["xsmall"]["lr_top1"], "lr_x", True),
}


def decide(v, accept, key, top1):
    cands = v["cands"][:1] if top1 else v["cands"]
    ok = [f for f in cands if accept(f)]
    return max(ok, key=lambda f: f[key])["cid"] if ok else None


def evaluate(vs, accept, key, top1):
    P = [v for v in vs if v["is_para"]]
    T = [v for v in vs if not v["is_para"]]
    pc = sum(1 for v in P if decide(v, accept, key, top1) == f"o:{v['gid']}")
    pw = sum(1 for v in P if decide(v, accept, key, top1) not in (None, f"o:{v['gid']}"))
    th = sum(1 for v in T if decide(v, accept, key, top1) is not None)
    hits = pc + pw + th
    return {"para": pc / len(P), "para_ci": cp(pc, len(P)), "para_wrong": pw / len(P),
            "trap": th / len(T), "trap_ci": cp(th, len(T)), "hits": hits,
            "false_reuse_among_hits": (pw + th) / hits if hits else None, "n": (len(P), len(T))}


test = [v for v in V if v["split"] == "test"]
calib = [v for v in V if v["split"] == "calibration"]
result = {"meta": out["meta"], "parity": out["parity"], "frozen": {}, "recalibrated": {}, "by_stratum": {}, "auc": {}}

for name, (acc, key, top1) in JUDGES.items():
    result["frozen"][name] = {"test": evaluate(test, acc, key, top1), "all": evaluate(V, acc, key, top1)}
for name in ("C3 운영 (코사인+quora)", "NLI xsmall, 1위만", "LR+NLI base, 1위만"):
    acc, key, top1 = JUDGES[name]
    result["by_stratum"][name] = {s: evaluate([v for v in test if v["stratum"] == s], acc, key, top1)
                                  for s in sorted({v["stratum"] for v in V})}
for key in ("cos", "quora", "nli_b", "nli_x", "lr_b", "lr_x"):
    result["auc"][key] = auc([v["own"][key] for v in V if v["is_para"]], [v["own"][key] for v in V if not v["is_para"]])

# 보조: 재현본 calibration 에서 다시 고르기 (함정 통과 5% 이하에서 같은 뜻 재사용 최대)
for name, key, top1 in (("코사인", "cos", False), ("quora", "quora", False), ("NLI base", "nli_b", True),
                        ("NLI xsmall", "nli_x", True), ("LR+NLI base", "lr_b", True)):
    grid = sorted({f[key] for v in calib for f in v["cands"]})
    best = None
    for t in grid:
        m = evaluate(calib, lambda f, t=t, key=key: f[key] >= t, key, top1)
        if m["trap"] <= 0.05 and (best is None or m["para"] > best[1]["para"]):
            best = (t, m)
    if best:
        t = best[0]
        result["recalibrated"][name] = {"threshold": t, "calib": best[1],
                                        "test": evaluate(test, lambda f, t=t, key=key: f[key] >= t, key, top1)}
    else:
        result["recalibrated"][name] = None

json.dump(result, open("results/evaluation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print("meta", json.dumps(out["meta"]), "\nparity", json.dumps(out["parity"]))
print(f"\n[고정 기준, test 200그룹]  {'판정':34s} 같은뜻 [95%]          함정통과 [95%]        hit중오재사용")
for name, r in result["frozen"].items():
    t = r["test"]
    print(f"  {name:36s} {t['para']:.3f} [{t['para_ci'][0]:.2f},{t['para_ci'][1]:.2f}]  {t['trap']:.3f} [{t['trap_ci'][0]:.2f},{t['trap_ci'][1]:.2f}]  "
          f"{t['false_reuse_among_hits'] if t['false_reuse_among_hits'] is None else round(t['false_reuse_among_hits'], 3)}")
print("\nAUC(자기 원질문, 같은 뜻 vs 함정):", {k: round(v, 3) for k, v in result["auc"].items()})
print("\n[재현본 calibration에서 다시 고른 기준 → test]")
for name, r in result["recalibrated"].items():
    print(f"  {name:12s}", "기준 없음" if r is None else f"t={r['threshold']:.4g}  같은뜻 {r['test']['para']:.3f}  함정 {r['test']['trap']:.3f}")
print("\n[층별 test: 같은뜻 / 함정]")
for name, d in result["by_stratum"].items():
    print(f"  {name}: " + "  ".join(f"{s} {m['para']:.2f}/{m['trap']:.2f}" for s, m in d.items()))
