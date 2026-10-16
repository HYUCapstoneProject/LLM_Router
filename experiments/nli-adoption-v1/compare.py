"""NLI 도입 방식 비교 (PLAN.md). 기존 점수만 쓴다.

    python compare.py
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
PROD_COS, PROD_Q = 0.8774, 0.03581769
SEP = "␟"


def ent(ab, ba):
    return min(ab, ba)


# ── 대화형 ──
nat_groups = {json.loads(l)["id"]: json.loads(l) for l in open(NAT / "data/dataset.jsonl", encoding="utf-8")}
nat_base = json.load(open(NAT / "results/scores.json", encoding="utf-8"))["scores"]
nat_x = json.load(open(NAT / "results/scores_small_nli.json", encoding="utf-8"))["cross-encoder/nli-deberta-v3-xsmall"]
items = []
for line in open(NAT / "results/raw.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["record"] != "variant":
        continue
    cands = []
    for oid, cos, quora in r["candidates"]:
        pid = f"{r['id']}|{r['kind']}|{oid}"
        b, x = nat_base[pid], nat_x[pid]
        cands.append({"cid": oid, "cos": cos, "quora": quora,
                      "base": ent(b["nli_ab"]["entailment"], b["nli_ba"]["entailment"]),
                      "xsmall": ent(x["ab"]["entailment"], x["ba"]["entailment"])})
    items.append({"ds": "chat", "gid": r["id"], "own": r["id"], "is_para": r["kind"] != "trap",
                  "split": "calib" if hashlib.sha256(r["id"].encode()).digest()[0] % 3 == 0 else "test",
                  "len": len(nat_groups[r["id"]][r["kind"]]), "stratum": nat_groups[r["id"]]["domain"], "cands": cands})

# ── D2 재현본 ──
d2_groups = {json.loads(l)["gid"]: json.loads(l) for l in open(D2 / "data/dataset.jsonl", encoding="utf-8")}
out = json.load(open(D2 / "results/out.json", encoding="utf-8"))
for r in out["records"]:
    g = d2_groups[r["gid"]]
    cands = []
    for cid, cos in r["candidates"]:
        s = out["pair_scores"][f"{r['text']}{SEP}{cid}"]
        cands.append({"cid": cid, "cos": cos, "quora": s["quora"],
                      "base": ent(s["nli_b"]["ent_ab"], s["nli_b"]["ent_ba"]),
                      "xsmall": ent(s["nli_x"]["ent_ab"], s["nli_x"]["ent_ba"])})
    items.append({"ds": "d2", "gid": r["gid"], "own": f"o:{r['gid']}", "is_para": r["kind"] == "para",
                  "split": "calib" if g["split"] == "calibration" else "test",
                  "len": len(r["text"]), "stratum": g["stratum"], "cands": cands})


def pick(it, model, top1):
    """후보 하나로 줄인다. 기준 하나로 거르고 점수 최대를 고르면, 통과 여부는 최대 점수 후보로 결정된다."""
    cs = it["cands"][:1] if top1 else it["cands"]
    if not cs:
        return None, float("-inf")
    best = max(cs, key=lambda c: c[model])
    return best["cid"], best[model]


def choose(its, model, top1):
    """함정 통과 ≤ 5% 에서 같은 뜻 재사용이 최대인 가장 낮은 기준."""
    traps = sorted((pick(i, model, top1)[1] for i in its if not i["is_para"]), reverse=True)
    if not traps:
        return float("-inf")
    k = int(TRAP_LIMIT * len(traps))           # 통과시켜도 되는 함정 수
    if k >= len(traps):
        return float("-inf")
    edge = traps[k]
    return float("-inf") if edge == float("-inf") else edge + 1e-12


def judge(rule, model, top1):
    """rule(it) -> 기준 또는 None(의미 캐시 안 씀). 반환: it -> 재사용한 cid 또는 None."""
    def f(it):
        t = rule(it)
        if t is None:
            return None
        cid, s = pick(it, model, top1)
        return cid if cid is not None and s >= t else None
    return f


def c3(it):
    ok = [c for c in it["cands"] if c["cos"] >= PROD_COS and c["quora"] >= PROD_Q]
    return max(ok, key=lambda c: c["quora"])["cid"] if ok else None


def metrics(its, decide):
    P = [i for i in its if i["is_para"]]
    T = [i for i in its if not i["is_para"]]
    d = {id(i): decide(i) for i in its}
    pc = sum(1 for i in P if d[id(i)] == i["own"])
    pw = sum(1 for i in P if d[id(i)] not in (None, i["own"]))
    th = sum(1 for i in T if d[id(i)] is not None)
    return {"para": pc / len(P) if P else None, "para_ci": cp(pc, len(P)) if P else None,
            "para_wrong": pw / len(P) if P else None,
            "trap": th / len(T) if T else None, "trap_ci": cp(th, len(T)) if T else None,
            "n_para": len(P), "n_trap": len(T), "pc": pc, "th": th}


calib = [i for i in items if i["split"] == "calib"]
test = [i for i in items if i["split"] == "test"]
LS = list(range(40, 801, 10))


def para_count(its, decide):
    return sum(1 for i in its if i["is_para"] and decide(i) == i["own"])


def fit(method, model, top1):
    if method == "A":
        t = choose(calib, model, top1)
        return {"t": t}, (lambda it: t)
    best = None
    for L in LS:
        short = [i for i in calib if i["len"] <= L]
        long_ = [i for i in calib if i["len"] > L]
        ts = choose(short, model, top1)
        tl = choose(long_, model, top1) if method == "B" else None
        rule = (lambda it, L=L, ts=ts, tl=tl: ts if it["len"] <= L else tl)
        score = para_count(calib, judge(rule, model, top1))
        if best is None or score > best[0]:
            best = (score, {"L": L, "t_short": ts, "t_long": tl}, rule)
    return best[1], best[2]


def oracle(model, top1):
    ts = {ds: choose([i for i in calib if i["ds"] == ds], model, top1) for ds in ("chat", "d2")}
    return {"t_chat": ts["chat"], "t_d2": ts["d2"]}, (lambda it: ts[it["ds"]])


def report(decide):
    res = {}
    for ds in ("chat", "d2"):
        res[ds] = metrics([i for i in test if i["ds"] == ds], decide)
    res["d2_short_le100"] = metrics([i for i in test if i["ds"] == "d2" and i["len"] <= 100], decide)
    res["d2_by_stratum"] = {s: metrics([i for i in test if i["ds"] == "d2" and i["stratum"] == s], decide)
                            for s in sorted({i["stratum"] for i in items if i["ds"] == "d2"})}
    return res


results = {"split": {ds: {sp: sum(1 for i in items if i["ds"] == ds and i["split"] == sp) for sp in ("calib", "test")}
                     for ds in ("chat", "d2")},
           "methods": {"C3 운영": {"params": {}, "test": report(c3)}}}
for model in ("xsmall", "base"):
    for top1 in (True, False):
        tag = f"{model}{', 1위만' if top1 else ', 후보 전체'}"
        for name, fitter in (("A 기준 하나", lambda: fit("A", model, top1)), ("B 길이별 기준", lambda: fit("B", model, top1)),
                             ("C 짧은 질문만 NLI", lambda: fit("C", model, top1)), ("참고 데이터셋별 상한", lambda: oracle(model, top1))):
            params, rule = fitter()
            results["methods"][f"{name} | {tag}"] = {"params": params, "test": report(judge(rule, model, top1))}

# B 의 L 민감도: xsmall, 1위만
sens = []
for L in (100, 150, 200, 300, 400, 600):
    short = [i for i in calib if i["len"] <= L]
    long_ = [i for i in calib if i["len"] > L]
    ts, tl = choose(short, "xsmall", True), choose(long_, "xsmall", True)
    rule = lambda it, L=L, ts=ts, tl=tl: ts if it["len"] <= L else tl
    r = report(judge(rule, "xsmall", True))
    sens.append({"L": L, "t_short": ts, "t_long": tl, "chat": r["chat"], "d2": r["d2"]})
results["L_sensitivity_xsmall_top1"] = sens

HERE.joinpath("results").mkdir(exist_ok=True)
json.dump(results, open(HERE / "results/comparison.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False,
          default=float)

f = lambda m: f"{m['para']:.3f} [{m['para_ci'][0]:.2f},{m['para_ci'][1]:.2f}] / {m['trap']:.3f} [{m['trap_ci'][0]:.2f},{m['trap_ci'][1]:.2f}]"
print("split", results["split"])
print(f"{'method':38s} {'params':44s} | chat 같은뜻/함정 | d2 같은뜻/함정 | d2≤100자 | d2 오재사용")
for name, r in results["methods"].items():
    p = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["params"].items())
    t = r["test"]
    s = t["d2_short_le100"]
    print(f"{name:38s} {p:44s} | {f(t['chat'])} | {f(t['d2'])} | {s['para']:.2f}/{s['trap']:.2f} (n={s['n_para']},{s['n_trap']}) | {t['d2']['para_wrong']:.3f}")
print("\nL 민감도 (B, xsmall 1위만):")
for s in sens:
    print(f"  L={s['L']:4d} ts={s['t_short']:.4g} tl={s['t_long']:.4g} | chat {s['chat']['para']:.3f}/{s['chat']['trap']:.3f} | d2 {s['d2']['para']:.3f}/{s['d2']['trap']:.3f}")
