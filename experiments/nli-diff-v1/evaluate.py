"""PLAN.md 비교. 빌드 로그에서 꺼낸 results/scores_*.json 과 기존 점수를 합쳐 평가한다.

    python fetch.py      # 빌드 로그 → results/scores_<build>.json
    python evaluate.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAT, D2 = HERE.parent / "cache-natural-v1", HERE.parent / "d2-replica-v1"
sys.path.insert(0, str(NAT))
from analyze import cp  # noqa: E402

TRAP_LIMIT = 0.05
SEP = "␟"
LS = list(range(40, 801, 10))

index = json.load(open(HERE / "data/index.json", encoding="utf-8"))
new = {}
for f in sorted((HERE / "results").glob("scores_*.json")):
    new.update(json.load(open(f, encoding="utf-8")))

# (모델, 형태) -> {키: 양방향 함의 최소}
S = {}
for name, r in new.items():
    by_row = {row: min(a, b) for row, a, b in zip(r["rows"], r["ent_ab"], r["ent_ba"])}
    short = name.split("/")[-1]
    for form in ("full", "diff"):
        d = {k: by_row[v[form]] for k, v in index.items() if v[form] in by_row}
        if len(d) == len(index):
            S[(short, form)] = d

# 기존 원문 점수(base, xsmall)
nat_base = json.load(open(NAT / "results/scores.json", encoding="utf-8"))["scores"]
nat_x = json.load(open(NAT / "results/scores_small_nli.json", encoding="utf-8"))["cross-encoder/nli-deberta-v3-xsmall"]
out = json.load(open(D2 / "results/out.json", encoding="utf-8"))
old = {("nli-deberta-v3-base", "full"): {}, ("nli-deberta-v3-xsmall", "full"): {}}
for pid, s in nat_base.items():
    old[("nli-deberta-v3-base", "full")]["chat|" + pid] = min(s["nli_ab"]["entailment"], s["nli_ba"]["entailment"])
for pid, s in nat_x.items():
    old[("nli-deberta-v3-xsmall", "full")]["chat|" + pid] = min(s["ab"]["entailment"], s["ba"]["entailment"])
for k, s in out["pair_scores"].items():
    old[("nli-deberta-v3-base", "full")]["d2|" + k] = min(s["nli_b"]["ent_ab"], s["nli_b"]["ent_ba"])
    old[("nli-deberta-v3-xsmall", "full")]["d2|" + k] = min(s["nli_x"]["ent_ab"], s["nli_x"]["ent_ba"])
parity = {}
for key, d in old.items():
    if key in S:                                   # 새로 잰 원문 점수가 있으면 차이만 기록
        parity[key[0]] = max(abs(S[key][k] - d[k]) for k in d)
    else:
        S[key] = d
# 대화형은 차이 문장이 원문과 같은 경우가 많다 → 같은 행끼리 비교해 환경 차이를 잰다
for m in ("nli-deberta-v3-base", "nli-deberta-v3-xsmall"):
    if (m, "diff") in S:
        same = [k for k, v in index.items() if v["full"] == v["diff"]]
        parity[m + " (diff=full 행, 새 환경 vs 기존)"] = max(abs(S[(m, "diff")][k] - S[(m, "full")][k]) for k in same)

# ── 항목 ──
nat_groups = {json.loads(l)["id"]: json.loads(l) for l in open(NAT / "data/dataset.jsonl", encoding="utf-8")}
d2_groups = {json.loads(l)["gid"]: json.loads(l) for l in open(D2 / "data/dataset.jsonl", encoding="utf-8")}
items = []
for line in open(NAT / "results/raw.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["record"] != "variant":
        continue
    items.append({"ds": "chat", "own": r["id"], "is_para": r["kind"] != "trap",
                  "split": "calib" if hashlib.sha256(r["id"].encode()).digest()[0] % 3 == 0 else "test",
                  "len": len(nat_groups[r["id"]][r["kind"]]), "stratum": nat_groups[r["id"]]["domain"],
                  "top1": None if not r["candidates"] else
                  {"cid": r["candidates"][0][0], "key": f"chat|{r['id']}|{r['kind']}|{r['candidates'][0][0]}"}})
for r in out["records"]:
    g = d2_groups[r["gid"]]
    items.append({"ds": "d2", "own": f"o:{r['gid']}", "is_para": r["kind"] == "para",
                  "split": "calib" if g["split"] == "calibration" else "test", "len": len(r["text"]),
                  "stratum": g["stratum"],
                  "top1": None if not r["candidates"] else
                  {"cid": r["candidates"][0][0], "key": f"d2|{r['text']}{SEP}{r['candidates'][0][0]}"}})
calib = [i for i in items if i["split"] == "calib"]
test = [i for i in items if i["split"] == "test"]


def scorer(model, short_form, long_form, L):
    a, b = S[(model, short_form)], S[(model, long_form)]
    return lambda it: (a if it["len"] <= L else b)[it["top1"]["key"]]


def choose(its, sc):
    traps = sorted((sc(i) for i in its if not i["is_para"] and i["top1"]), reverse=True)
    n_trap = sum(1 for i in its if not i["is_para"])
    k = int(TRAP_LIMIT * n_trap)
    return float("-inf") if k >= len(traps) else traps[k] + 1e-12


def decide(it, sc, t):
    if it["top1"] is None or t is None:
        return None
    return it["top1"]["cid"] if sc(it) >= t else None


def metrics(its, rule):
    P = [i for i in its if i["is_para"]]
    T = [i for i in its if not i["is_para"]]
    d = {id(i): rule(i) for i in its}
    pc = sum(1 for i in P if d[id(i)] == i["own"])
    pw = sum(1 for i in P if d[id(i)] not in (None, i["own"]))
    th = sum(1 for i in T if d[id(i)] is not None)
    return {"para": pc / len(P), "para_ci": cp(pc, len(P)), "para_wrong": pw / len(P),
            "trap": th / len(T), "trap_ci": cp(th, len(T)), "n_para": len(P), "n_trap": len(T)}


def report(rule):
    r = {ds: metrics([i for i in test if i["ds"] == ds], rule) for ds in ("chat", "d2")}
    r["d2_long"] = metrics([i for i in test if i["ds"] == "d2" and i["len"] > 200], rule)
    r["d2_by_stratum"] = {s: metrics([i for i in test if i["ds"] == "d2" and i["stratum"] == s], rule)
                          for s in sorted({i["stratum"] for i in items if i["ds"] == "d2"})}
    return r


def fit_single(model, form):
    sc = scorer(model, form, form, 10 ** 9)
    t = choose(calib, sc)
    return {"t": t}, (lambda it: decide(it, sc, t))


def fit_split(model, long_form):
    best = None
    for L in LS:
        sc = scorer(model, "full", long_form, L)
        ts = choose([i for i in calib if i["len"] <= L], sc)
        tl = choose([i for i in calib if i["len"] > L], sc)
        rule = (lambda it, sc=sc, L=L, ts=ts, tl=tl: decide(it, sc, ts if it["len"] <= L else tl))
        n = sum(1 for i in calib if i["is_para"] and rule(i) == i["own"])
        if best is None or n > best[0]:
            best = (n, {"L": L, "t_short": ts, "t_long": tl}, rule)
    return best[1], best[2]


def fixed(model, form, t=0.5):
    sc = scorer(model, form, form, 10 ** 9)
    return {"t": t}, (lambda it: decide(it, sc, t))


models = sorted({m for m, _ in S})
results = {"parity": parity, "models": {m: sorted(f for mm, f in S if mm == m) for m in models}, "methods": {}}
for m in models:
    forms = [f for f in ("full", "diff") if (m, f) in S]
    for f in forms:
        results["methods"][f"{m} | A 기준 하나 | {f}"] = fit_single(m, f)
        results["methods"][f"{m} | 고정 0.5 | {f}"] = fixed(m, f)
    results["methods"][f"{m} | B 길이별 | full"] = fit_split(m, "full")
    if "diff" in forms:
        results["methods"][f"{m} | B 길이별 | 짧음 full + 김 diff"] = fit_split(m, "diff")
for k, (params, rule) in list(results["methods"].items()):
    results["methods"][k] = {"params": params, "test": report(rule)}

(HERE / "results").mkdir(exist_ok=True)
json.dump(results, open(HERE / "results/evaluation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False,
          default=float)

f = lambda m: f"{m['para']:.3f}/{m['trap']:.3f}"
print("parity", json.dumps(parity, ensure_ascii=False))
print(f"{'방법':62s} {'기준':40s} | 대화형 | D2 | D2 긴(>200) | D2 오재사용")
for k, r in results["methods"].items():
    p = ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in r["params"].items())
    t = r["test"]
    print(f"{k:62s} {p:40s} | {f(t['chat'])} | {f(t['d2'])} | {f(t['d2_long'])} | {t['d2']['para_wrong']:.3f}")
