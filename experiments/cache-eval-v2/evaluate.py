"""cache-eval-v2 결과 집계. 빌드 로그에서 결과를 꺼내 results/ 에 두고 표를 만든다.

    python evaluate.py
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

(HERE / "results").mkdir(exist_ok=True)
R = {}
for line in open(HERE / "builds.txt", encoding="utf-8"):
    name, bid = line.split()
    path = HERE / f"results/raw_{name}.json"
    if not path.exists():
        status = subprocess.run(["gcloud", "builds", "describe", bid, "--format=value(status)"],
                                capture_output=True, text=True, shell=True).stdout.strip()
        if status != "SUCCESS":
            print(name, status)
            continue
        log = subprocess.run(["gcloud", "builds", "log", bid], capture_output=True, text=True, shell=True,
                             encoding="utf-8").stdout
        body = log.split("===OUT-BEGIN===")[1].split("===OUT-END===")[0]
        blob = "".join(l.split()[-1] for l in body.strip().splitlines() if l.strip())
        json.dump(json.loads(gzip.decompress(base64.b64decode(blob))), open(path, "w", encoding="utf-8"))
    R[name] = json.load(open(path, encoding="utf-8"))

P = ("C1", "C2", "C3", "C4")
pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "–"
ci = lambda a, b: "[{:.1f}, {:.1f}]".format(*(100 * x for x in cp(a, b))) if b else ""
summary = {}


def stream_row(m):
    return {"hit_rate": m["hits"] / m["requests"], "wrong_among_hits": m["wrong"] / m["hits"] if m["hits"] else None,
            "wrong_per_request": m["wrong"] / m["requests"],
            "recall": m["correct_when_available"] / m["available"] if m["available"] else None, **m}


print("\n## 흐름형 (히트율 = 호출 절감 / 히트 중 오재사용 [95%] / 요청당 오재사용 / 재사용 가능 요청 중 정답 히트)")
for name in ("lmarena", "search", "classification"):
    if name not in R:
        continue
    summary[name] = {}
    print(f"\n{name}  {R[name]['meta']}")
    for p in P:
        m = R[name]["policies"][p]
        summary[name][p] = stream_row(m)
        print(f"  {p}: {pct(m['hits'], m['requests'])} / {pct(m['wrong'], m['hits'])} {ci(m['wrong'], m['hits'])} / "
              f"{pct(m['wrong'], m['requests'])} / {pct(m['correct_when_available'], m['available'])}")
    if name == "classification":
        for ds, res in R[name]["by_dataset"].items():
            print(f"   [{ds}] " + "  ".join(f"{p} {pct(res[p]['hits'], res[p]['requests'])}/{pct(res[p]['wrong'], res[p]['hits'])}"
                                         for p in P))

if "dstream" in R:
    print(f"\n## D1 재생 (seed 3개 합산)  {R['dstream']['meta']}")
    summary["dstream"] = {}
    for r in (0, 10, 30, 50):
        row = {}
        for p in P:
            agg = {k: sum(R["dstream"]["traces"][f"r{r}_s{s}"][p][k] for s in range(3))
                   for k in ("requests", "hits", "wrong", "semantic_hits", "semantic_wrong", "correct")}
            row[p] = agg
        summary["dstream"][f"r{r}"] = row
        print(f"  r={r}%  " + "  ".join(f"{p} 절감 {pct(row[p]['hits'], row[p]['requests'])} "
                                       f"오재사용/히트 {pct(row[p]['wrong'], row[p]['hits'])}" for p in P))

print("\n## 쌍형 (같은 뜻 재사용 [95%] / 같은 뜻이 다른 항목에 / 함정·라벨0 통과 [95%])")
for name in ("service", "pairs"):
    if name not in R:
        continue
    for sub, res in R[name]["subsets"].items():
        summary[f"{name}:{sub}"] = {}
        print(f"\n{sub}  cache={res['meta']['cache']} queries={res['meta']['queries']}")
        for p in P:
            m = res[p]
            summary[f"{name}:{sub}"][p] = {k: v for k, v in m.items() if k != "decisions"}
            print(f"  {p}: {pct(m['pos_own'], m['n_pos'])} {ci(m['pos_own'], m['n_pos'])} / {pct(m['pos_other'], m['n_pos'])} / "
                  f"{pct(m['neg_hit'], m['n_neg'])} {ci(m['neg_hit'], m['n_neg'])}")
        if name == "service":
            kinds = res["meta"]["query_kinds"]
            for k in ("para_light", "para_heavy", "trap", "trap2"):
                idx = [i for i, x in enumerate(kinds) if x == k]
                print(f"   [{k} n={len(idx)}] " + "  ".join(
                    f"{p} {pct(sum(1 for i in idx if res[p]['decisions'][i] is not None), len(idx))}" for p in P))
            cats = res["meta"]["query_category"]
            summary[f"{name}:{sub}"]["by_category"] = {}
            for c in sorted(set(cats)):
                idx = [i for i, x in enumerate(cats) if x == c]
                pos = [i for i in idx if kinds[i].startswith("para")]
                neg = [i for i in idx if kinds[i].startswith("trap")]
                summary[f"{name}:{sub}"]["by_category"][c] = {
                    p: {"pos_own": sum(1 for i in pos if res[p]["decisions"][i] == "own"), "n_pos": len(pos),
                        "neg_hit": sum(1 for i in neg if res[p]["decisions"][i] is not None), "n_neg": len(neg)}
                    for p in P}

json.dump(summary, open(HERE / "results/summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
