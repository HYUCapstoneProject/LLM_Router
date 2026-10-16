"""GPTCache 실행 결과 평가 (PLAN.md).

    python evaluate.py <빌드 ID>
"""
import base64
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cache-natural-v1"))
from analyze import cp  # noqa: E402

TRAP_LIMIT = 0.05
raw_path = HERE / "results/raw_gptcache.json"
if not raw_path.exists():
    log = subprocess.run(["gcloud", "builds", "log", sys.argv[1]], capture_output=True, text=True, shell=True,
                         encoding="utf-8").stdout
    body = log.split("===OUT-BEGIN===")[1].split("===OUT-END===")[0]
    blob = "".join(l.split()[-1] for l in body.strip().splitlines() if l.strip())
    json.dump(json.loads(gzip.decompress(base64.b64decode(blob))), open(raw_path, "w", encoding="utf-8"))
raw = json.load(open(raw_path, encoding="utf-8"))
meta = json.load(open(HERE / "data/meta.json", encoding="utf-8"))


def metrics(rows, decide):
    P = [r for r in rows if r["is_para"]]
    T = [r for r in rows if not r["is_para"]]
    pc = sum(1 for r in P if decide(r) == r["own"])
    pw = sum(1 for r in P if decide(r) not in (None, r["own"]))
    th = sum(1 for r in T if decide(r) is not None)
    return {"para": pc / len(P), "para_ci": cp(pc, len(P)), "para_wrong": pw / len(P),
            "trap": th / len(T), "trap_ci": cp(th, len(T)), "n_para": len(P), "n_trap": len(T)}


result = {"meta": raw["meta"], "put_errors": {ds: len(raw[ds]["put_errors"]) for ds in ("chat", "d2")}, "configs": {}}
for name in raw["chat"]["configs"]:
    rows = []
    for ds in ("chat", "d2"):
        c = raw[ds]["configs"][name]
        for qid, (hit, _) in c["0.8"]["answers"].items():
            hit0, score = c["0.0"]["answers"][qid]
            rows.append({**meta[f"{ds}|{qid}"], "ds": ds, "hit_default": hit, "top1": hit0, "score": score})
    # 점수로 낸 0.8 판정이 실제 0.8 실행과 같은지 (같아야 맞춘 기준을 보고한다)
    lo, hi = {"G1_distance": (0.0, 4.0)}.get(name, (0.0, 1.0))
    thr_rank = (hi - lo) * 0.8
    mismatch = sum(1 for r in rows if (r["top1"] if r["score"] is not None and r["score"] * (hi - lo) >= thr_rank - 1e-9
                                       else None) != r["hit_default"])
    test = [r for r in rows if r["split"] == "test"]
    calib = [r for r in rows if r["split"] == "calib"]
    out = {"errors": {ds: {t: len(raw[ds]["configs"][name][t]["errors"]) for t in ("0.8", "0.0")} for ds in ("chat", "d2")},
           "ms_per_query": {ds: raw[ds]["configs"][name]["0.8"]["ms_per_query"] for ds in ("chat", "d2")},
           "parity_mismatch": mismatch,
           "default_0.8": {ds: metrics([r for r in test if r["ds"] == ds], lambda r: r["hit_default"]) for ds in ("chat", "d2")}}
    if mismatch == 0:
        # 합친 calibration 에서 함정 ≤ 5% 인 가장 낮은 기준 (점수는 0~1 로 정규화된 값)
        traps = sorted((r["score"] for r in calib if not r["is_para"] and r["score"] is not None), reverse=True)
        k = int(TRAP_LIMIT * sum(1 for r in calib if not r["is_para"]))
        t = traps[k] + 1e-12 if k < len(traps) else float("-inf")
        dec = lambda r, t=t: r["top1"] if r["score"] is not None and r["score"] >= t else None
        out["tuned"] = {"threshold_normalized": t,
                        **{ds: metrics([r for r in test if r["ds"] == ds], dec) for ds in ("chat", "d2")}}
        # 데이터셋별로 따로 맞춘 기준(상한 참고)
        out["tuned_per_dataset"] = {}
        for ds in ("chat", "d2"):
            cc = [r for r in calib if r["ds"] == ds]
            tr = sorted((r["score"] for r in cc if not r["is_para"] and r["score"] is not None), reverse=True)
            kk = int(TRAP_LIMIT * sum(1 for r in cc if not r["is_para"]))
            tt = tr[kk] + 1e-12 if kk < len(tr) else float("-inf")
            out["tuned_per_dataset"][ds] = {"threshold_normalized": tt, **metrics(
                [r for r in test if r["ds"] == ds], lambda r, tt=tt: r["top1"] if r["score"] is not None and r["score"] >= tt else None)}
    result["configs"][name] = out

json.dump(result, open(HERE / "results/evaluation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
f = lambda m: f"{m['para']:.3f} [{m['para_ci'][0]:.2f},{m['para_ci'][1]:.2f}] / {m['trap']:.3f} [{m['trap_ci'][0]:.2f},{m['trap_ci'][1]:.2f}] (오재사용 {m['para_wrong']:.3f})"
print("meta", raw["meta"], "put_errors", result["put_errors"])
for name, o in result["configs"].items():
    print(f"\n{name}  errors={o['errors']}  ms/q={o['ms_per_query']}  parity_mismatch={o['parity_mismatch']}")
    print("  기본 0.8   대화형", f(o["default_0.8"]["chat"]), "| D2", f(o["default_0.8"]["d2"]))
    if "tuned" in o:
        print(f"  맞춘 기준({o['tuned']['threshold_normalized']:.4f}) 대화형", f(o["tuned"]["chat"]), "| D2", f(o["tuned"]["d2"]))
        for ds, m in o["tuned_per_dataset"].items():
            print(f"  {ds}만 맞춤({m['threshold_normalized']:.4f})", f(m))
