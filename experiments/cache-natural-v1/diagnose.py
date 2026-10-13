"""성능이 낮은 원인 진단. 모두 자기 원질문과의 쌍 점수(own_sim, own_score)로 계산한다.

같은 데이터 안의 사후 분석이며 운영 임계값을 고르는 용도가 아니다.

    python diagnose.py
"""
import collections
import json
import math
import re
import statistics as st

from analyze import auc

groups = {json.loads(l)["id"]: json.loads(l) for l in open("data/dataset.jsonl", encoding="utf-8")}
V = [json.loads(l) for l in open("results/raw.jsonl", encoding="utf-8")]
V = [v for v in V if v["record"] == "variant"]
STOP = set("a an the of in on to for is are do does did i my me you your what how which why when who whom "
           "can should would could with and or it its that this be by at from as".split())


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        for k in range(i, j):
            r[order[k]] = (i + j - 1) / 2
        i = j
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


for v in V:
    o, t = groups[v["id"]]["original"], groups[v["id"]][v["kind"]]
    wo, wt = words(o), words(t)
    v["jac"] = jaccard(wo, wt)
    v["jac_content"] = jaccard([w for w in wo if w not in STOP], [w for w in wt if w not in STOP])
    v["same_bag"] = set(wo) == set(wt)
    v["len"] = len(wo)
    v["is_para"] = v["kind"] != "trap"

paras = [v for v in V if v["is_para"]]
traps = [v for v in V if not v["is_para"]]
out = {}


# 1. 리랭커와 코사인은 무엇을 따라가는가: 단어 겹침과의 상관
def sep(field, vs=V):
    return auc([v[field] for v in vs if v["is_para"]], [v[field] for v in vs if not v["is_para"]])


out["1_lexical"] = {
    "auc_para_vs_trap": {"cosine": sep("own_sim"), "rerank": sep("own_score"),
                         "word_jaccard": sep("jac"), "content_word_jaccard": sep("jac_content")},
    "spearman_with_word_jaccard": {
        "rerank_all": spearman([v["own_score"] for v in V], [v["jac"] for v in V]),
        "cosine_all": spearman([v["own_sim"] for v in V], [v["jac"] for v in V]),
        "rerank_paras_only": spearman([v["own_score"] for v in paras], [v["jac"] for v in paras]),
        "rerank_traps_only": spearman([v["own_score"] for v in traps], [v["jac"] for v in traps]),
    },
    "median_jaccard": {"para_light": st.median(v["jac"] for v in V if v["kind"] == "para_light"),
                       "para_heavy": st.median(v["jac"] for v in V if v["kind"] == "para_heavy"),
                       "trap": st.median(v["jac"] for v in traps)},
}

# 2. 단어 집합이 같고 순서만 다른 함정
bag = [v for v in traps if v["same_bag"]]
out["2_same_word_set_traps"] = {
    "n": len(bag), "c3_pass": sum(1 for v in bag if v["c3_hit"]),
    "median_rerank": st.median(v["own_score"] for v in bag) if bag else None,
    "examples": [(groups[v["id"]]["original"], groups[v["id"]]["trap"], round(v["own_score"], 3)) for v in bag][:6],
}

# 3. 리랭커 자체 경계(0.5)에서 본 판단
out["3_reranker_native_boundary"] = {
    "para_score_ge_0.5": sum(1 for v in paras if v["own_score"] >= 0.5) / len(paras),
    "trap_score_ge_0.5": sum(1 for v in traps if v["own_score"] >= 0.5) / len(traps),
    "para_score_lt_0.036": sum(1 for v in paras if v["own_score"] < 0.03581769) / len(paras),
    "trap_score_lt_0.036": sum(1 for v in traps if v["own_score"] < 0.03581769) / len(traps),
}


# 4. 같은 함정 통과율에서 같은 뜻을 얼마나 살리나: 운영 AND 규칙 vs 단일 점수
def own_pass(v, rule):
    return rule(v)


prod = lambda v: v["own_sim"] >= 0.8774 and v["own_score"] >= 0.03581769
trap_rate_prod = sum(1 for v in traps if prod(v)) / len(traps)
para_rate_prod = sum(1 for v in paras if prod(v)) / len(paras)


def para_at_trap_rate(score, target):
    """함정 통과율이 target 이하가 되는 가장 낮은 기준에서 같은 뜻 통과율."""
    ts = sorted({score(v) for v in V})
    best = None
    for th in ts:
        tr = sum(1 for v in traps if score(v) >= th) / len(traps)
        if tr <= target:
            pr = sum(1 for v in paras if score(v) >= th) / len(paras)
            best = (th, tr, pr)
            break
    return best


zc = (st.mean(v["own_sim"] for v in V), st.pstdev(v["own_sim"] for v in V))
zr = (st.mean(logit(v["own_score"]) for v in V), st.pstdev(logit(v["own_score"]) for v in V))
combo = lambda v: (v["own_sim"] - zc[0]) / zc[1] + (logit(v["own_score"]) - zr[0]) / zr[1]
out["4_matched_operating_point"] = {
    "production_AND": {"trap_pass": trap_rate_prod, "para_pass": para_rate_prod},
    "cosine_only": para_at_trap_rate(lambda v: v["own_sim"], trap_rate_prod),
    "rerank_only": para_at_trap_rate(lambda v: v["own_score"], trap_rate_prod),
    "z_sum_cos_plus_rerank_logit": para_at_trap_rate(combo, trap_rate_prod),
    "auc_z_sum": sep("_combo") if False else auc([combo(v) for v in paras], [combo(v) for v in traps]),
}

# 5. 유형별 점수: 무엇이 바뀐 함정이 새는가
by_type = collections.defaultdict(list)
for v in traps:
    by_type[v["trap_type"]].append(v)
out["5_by_trap_type"] = {
    t: {"n": len(vs), "median_cos": round(st.median(v["own_sim"] for v in vs), 3),
        "median_rerank": round(st.median(v["own_score"] for v in vs), 4),
        "median_jaccard": round(st.median(v["jac"] for v in vs), 3),
        "c3_pass": round(sum(1 for v in vs if v["c3_hit"]) / len(vs), 3)}
    for t, vs in sorted(by_type.items(), key=lambda x: -st.median(v["own_score"] for v in x[1]))}
out["5_paraphrase_reference"] = {
    k: {"median_cos": round(st.median(v["own_sim"] for v in V if v["kind"] == k), 3),
        "median_rerank": round(st.median(v["own_score"] for v in V if v["kind"] == k), 4),
        "median_jaccard": round(st.median(v["jac"] for v in V if v["kind"] == k), 3)}
    for k in ("para_light", "para_heavy")}

# 6. 질문 길이
short = [v for v in V if groups[v["id"]]["original"] and len(words(groups[v["id"]]["original"])) <= 7]
long_ = [v for v in V if len(words(groups[v["id"]]["original"])) > 7]
out["6_length"] = {"short_le7_words": {"n_groups": len({v["id"] for v in short}),
                                       "auc_cos": sep("own_sim", short), "auc_rerank": sep("own_score", short)},
                   "long_gt7_words": {"n_groups": len({v["id"] for v in long_}),
                                      "auc_cos": sep("own_sim", long_), "auc_rerank": sep("own_score", long_)}}

# 7. 분야별: 프로그래밍이 왜 나쁜가
dom = {}
for d in sorted({v["domain"] for v in V}):
    vs = [v for v in V if v["domain"] == d]
    dom[d] = {"auc_cos": round(sep("own_sim", vs), 3), "auc_rerank": round(sep("own_score", vs), 3),
              "median_rerank_para_heavy": round(st.median(v["own_score"] for v in vs if v["kind"] == "para_heavy"), 4),
              "median_rerank_trap": round(st.median(v["own_score"] for v in vs if v["kind"] == "trap"), 4),
              "median_jac_para_heavy": round(st.median(v["jac"] for v in vs if v["kind"] == "para_heavy"), 3),
              "median_jac_trap": round(st.median(v["jac"] for v in vs if v["kind"] == "trap"), 3),
              "trap_types": dict(collections.Counter(v["trap_type"] for v in vs if v["kind"] == "trap").most_common(4))}
out["7_by_domain"] = dom

json.dump(out, open("results/diagnosis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
