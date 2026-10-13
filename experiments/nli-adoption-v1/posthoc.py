"""사후 분석(PLAN 밖). compare.py 의 함수·데이터를 그대로 쓴다.

    python posthoc.py
"""
import contextlib
import io
import json
import runpy

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("compare.py")
calib, test, choose, judge, report, pick = (g[k] for k in ("calib", "test", "choose", "judge", "report", "pick"))


def split_rule(L, ms, ml, top1=True, long_off=False):
    short = [i for i in calib if i["len"] <= L]
    long_ = [i for i in calib if i["len"] > L]
    ts, tl = choose(short, ms, top1), (None if long_off else choose(long_, ml, top1))

    def decide(it):
        m, t = (ms, ts) if it["len"] <= L else (ml, tl)
        if t is None:
            return None
        cid, s = pick(it, m, top1)
        return cid if cid is not None and s >= t else None
    return {"L": L, "short": ms, "long": None if long_off else ml, "t_short": ts, "t_long": tl}, decide


out = {}
for label, args in {
    "B base 1위만 L=200": (200, "base", "base"),
    "B xsmall 1위만 L=200": (200, "xsmall", "xsmall"),
    "B 짧음 xsmall + 김 base, L=200": (200, "xsmall", "base"),
    "B 짧음 base + 김 xsmall, L=200": (200, "base", "xsmall"),
    "C 짧음 base, 김 완전일치만, L=200": (200, "base", None),
    "C 짧음 xsmall, 김 완전일치만, L=200": (200, "xsmall", None),
}.items():
    L, ms, ml = args
    params, decide = split_rule(L, ms, ml, long_off=ml is None)
    out[label] = {"params": params, "test": report(decide)}

# B base 1위만의 L 민감도
out["L_sensitivity_base_top1"] = []
for L in (100, 150, 200, 250, 300, 400):
    params, decide = split_rule(L, "base", "base")
    r = report(decide)
    out["L_sensitivity_base_top1"].append({**params, "chat": r["chat"], "d2": r["d2"]})

# 길이 구간별 test 개수(지연 추정용)
out["test_len_share"] = {ds: sum(1 for i in test if i["ds"] == ds and i["len"] <= 200) / sum(1 for i in test if i["ds"] == ds)
                         for ds in ("chat", "d2")}
json.dump(out, open("results/posthoc.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

f = lambda m: f"{m['para']:.3f}/{m['trap']:.3f}"
for k, r in out.items():
    if k in ("L_sensitivity_base_top1", "test_len_share"):
        continue
    t = r["test"]
    p = r["params"]
    print(f"{k:34s} ts={p['t_short']:.4g} tl={p['t_long'] if p['t_long'] is None else round(p['t_long'], 5)} | chat {f(t['chat'])} | d2 {f(t['d2'])} wrong {t['d2']['para_wrong']:.3f}")
    print("    d2 층: " + "  ".join(f"{s} {f(m)}" for s, m in t["d2_by_stratum"].items()))
print("\nL 민감도 (B base 1위만)")
for s in out["L_sensitivity_base_top1"]:
    print(f"  L={s['L']:4d} ts={s['t_short']:.4g} tl={s['t_long']:.5g} | chat {f(s['chat'])} | d2 {f(s['d2'])}")
print("≤200자 비율", out["test_len_share"])
