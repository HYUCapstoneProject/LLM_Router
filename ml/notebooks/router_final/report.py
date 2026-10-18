"""EV-10 10_report: tables and figures from the 08 test outputs (and 09_seeds, 08a latency when present).

Reads <run>/test (08), <run>/seeds (09, optional), <run>/latency (08a, optional) and the lock. Writes <output>/:

  results.md                 every number the report quotes, as tables (generated; the narrative REPORT.md cites it)
  tables/*.csv               the same tables, machine-readable (no question text)
  figures/*.png              mean gain by source, gain over kappa, strata, tau curves, task types
  manifest.json              SHA-256 of every input file and output

Nothing is recomputed from outcomes: the numbers come from the 08/09/08a files, whose SHA-256 are recorded.
Figure text is English (Colab has no Korean font).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import categories as K  # noqa: E402
import common_track as C  # noqa: E402

SOURCES = C.SOURCES
REFERENCE = ("oracle", "r0_operational")
NAMES = {"graphrouter": "GraphRouter", "r1_group": "R1-group", "r1_group_a": "R1-group-A", "r0_group": "R0-group",
         "knn": "kNN", "knn_a": "kNN-A", "svm": "SVM", "routerdc": "RouterDC", "irt_router": "IRT-Router",
         "r1_bge": "R1-BGE", "routellm_retrained": "RouteLLM (retrained)", "routellm_released": "RouteLLM (released)",
         "mf": "MF", "mlp": "MLP", "avengers_pro": "Avengers-Pro", "hybrid_llm": "Hybrid LLM", "oracle": "oracle",
         "r0_operational": "R0 operational"}
FOCUS = ("graphrouter", "r1_group", "r0_group", "knn", "routerdc", "routellm_released")
COLORS = ("#2a6fdb", "#e0582b", "#1d9a6c", "#9b4fd1", "#c9a227", "#5b6b7a")  # fixed order, one per focus router


def md_table(df: pd.DataFrame, floatfmt: str = "{:+.3f}") -> list:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---:" if df[c].dtype.kind in "fi" else "---"
                                                                   for c in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(("–" if pd.isna(v) else floatfmt.format(v)) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r) + " |")
    return out


def load(run_dir: Path) -> dict:
    test = run_dir / "test"
    manifest = json.loads((test / "manifest.json").read_text())
    if manifest["rehearsal_on_dev"]:
        raise SystemExit("08 output is a dev rehearsal")
    for name, digest in manifest["outputs"].items():
        if C.sha256_file(test / name) != digest:
            raise SystemExit(f"08 output changed: {name}")
    data = {n: pd.read_parquet(test / f"{n}.parquet") for n in
            ("track1", "track2", "mean-gain", "kappa-star", "by-stratum", "by-category", "rt08", "fixed-starts")}
    data["hyp"] = json.loads((test / "hypotheses.json").read_text())
    data["diag"] = json.loads((test / "diagnostics.json").read_text())
    data["manifest"] = manifest
    seeds = run_dir / "seeds"
    if (seeds / "manifest.json").exists() and not json.loads((seeds / "manifest.json").read_text())["rehearsal_on_dev"]:
        data["seeds"] = pd.read_parquet(seeds / "seeds-mean-gain.parquet")
        data["h5_seeds"] = json.loads((seeds / "h5-seeds.json").read_text())
    for device in ("cpu", "cuda"):
        path = run_dir / "latency" / f"latency-{device}.parquet"
        if path.exists():
            data[f"latency_{device}"] = pd.read_parquet(path)
    return data


def figures(data: dict, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    made = []
    mg = data["mean-gain"]
    main = mg[mg.basis.eq("main")]
    # 1. mean gain by source (dot + 95% interval), methods ordered by macro
    order = main[main.source.eq("macro")].sort_values("mean_gain").method.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(13, 5.2), sharey=True)
    for ax, source in zip(axes, list(SOURCES) + ["macro"]):
        part = main[main.source.eq(source)].set_index("method").reindex(order)
        y = np.arange(len(order))
        ax.errorbar(part.mean_gain, y, xerr=[part.mean_gain - part.lo, part.hi - part.mean_gain], fmt="o", ms=4,
                    color="#2a6fdb", ecolor="#9db8e8", elinewidth=2, capsize=0)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(source)
        ax.set_yticks(y, [NAMES.get(m, m) for m in order])
        ax.grid(axis="x", color="#eee")
    axes[2].set_xlabel("mean gain over the best fixed start at the same cost (kappa grid x cost levels)")
    fig.tight_layout()
    fig.savefig(out / "mean-gain.png", dpi=160)
    plt.close(fig)
    made.append("mean-gain.png")
    # 2. gain over kappa at the 50% cost level
    t1 = data["track1"]
    sel = t1[t1.basis.eq("main") & t1.rule.eq("escalation") & t1.level.eq(0.5)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
    for ax, source in zip(axes, SOURCES):
        for m, c in zip(FOCUS, COLORS):
            part = sel[sel.source.eq(source) & sel.method.eq(m)].sort_values("kappa")
            ax.plot(part.kappa, part.gain, color=c, lw=2, label=NAMES[m])
            ax.fill_between(part.kappa, part.gain_lo, part.gain_hi, color=c, alpha=0.12, lw=0)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axhline(0.01, color="#bbb", lw=0.8, ls="--")
        ax.set_title(source)
        ax.set_xlabel("kappa (weight of one failed attempt)")
    axes[0].set_ylabel("gain at the 50% cost level")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "gain-by-kappa.png", dpi=160)
    plt.close(fig)
    made.append("gain-by-kappa.png")
    # 3. strata (pooled over sources)
    st = data["by-stratum"]
    pooled = st[st.source.eq("all") & st.method.isin(FOCUS + ("oracle",))]
    strata = ("solvable", "beyond_partial", "beyond_zero")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.12
    for k, (m, c) in enumerate(zip(FOCUS + ("oracle",), COLORS + ("#bbbbbb",))):
        v = pooled[pooled.method.eq(m)].set_index("group").reindex(strata).mean_gain
        ax.bar(np.arange(3) + (k - 3) * width, v, width * 0.9, color=c, label=NAMES[m])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(np.arange(3), ["solvable (some group >= .8)", "beyond, partly solved", "beyond, never solved"])
    ax.set_ylabel("mean gain inside the stratum")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "strata.png", dpi=160)
    plt.close(fig)
    made.append("strata.png")
    # 4. tau curves: first-attempt success against cost
    t2 = data["track2"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, source in zip(axes, SOURCES):
        for m, c in zip(FOCUS, COLORS):
            part = t2[t2.source.eq(source) & t2.method.eq(m)].sort_values("tau")
            ax.plot(part.cost, part["first"], color=c, lw=2, marker="o", ms=3, label=NAMES[m])
        fixed = t2[t2.source.eq(source) & t2.method.str.startswith("fixed_")]
        ax.scatter(fixed.cost, fixed["first"], color="#333", marker="s", s=18, label="fixed starts", zorder=3)
        ax.set_title(source)
        ax.set_xlabel("expected cost (high only = 1)")
    axes[0].set_ylabel("first-attempt success")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "tau-curves.png", dpi=160)
    plt.close(fig)
    made.append("tau-curves.png")
    # 5. task types (pooled)
    bc = data["by-category"]
    tt = bc[bc.source.eq("all") & bc.method.isin(FOCUS)].pivot_table(index="method", columns="group", values="mean_gain")
    tt = tt[[t for t in K.TYPES if t in tt.columns]].reindex(FOCUS)
    english = {"수학": "math", "코드": "code", "지식 객관식": "knowledge MC", "상식·추론": "commonsense",
               "독해": "reading", "전문가 과학·난제": "expert/HLE", "사실 회상": "factual recall", "열린 대화·지시": "open chat"}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    lim = np.nanmax(np.abs(tt.to_numpy()))
    im = ax.imshow(tt.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(tt.shape[1]), [english[c] for c in tt.columns], rotation=20)
    ax.set_yticks(range(tt.shape[0]), [NAMES[m] for m in tt.index])
    for i in range(tt.shape[0]):
        for j in range(tt.shape[1]):
            ax.text(j, i, f"{tt.iat[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean gain")
    fig.tight_layout()
    fig.savefig(out / "task-types.png", dpi=160)
    plt.close(fig)
    made.append("task-types.png")
    return made


def tables(data: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t = {}
    mg = data["mean-gain"]
    wide = mg[mg.basis.eq("main")].pivot_table(index="method", columns="source", values="mean_gain")
    wide = wide[list(SOURCES) + ["macro"]]
    for basis in ("group_price", "length"):
        wide[f"macro_{basis}"] = mg[mg.basis.eq(basis) & mg.source.eq("macro")].set_index("method").mean_gain
    t["mean_gain"] = wide.sort_values("macro", ascending=False)
    ci = mg[mg.basis.eq("main") & mg.source.eq("macro")].set_index("method")[["mean_gain", "lo", "hi"]]
    t["mean_gain_macro_ci"] = ci.sort_values("mean_gain", ascending=False)
    ks = data["kappa-star"]
    t["kappa_star_50"] = ks[ks.basis.eq("main") & ks.rule.eq("escalation") & ks.level.eq(0.5)].pivot_table(
        index="method", columns="source", values="kappa_star", aggfunc="first")
    st = data["by-stratum"]
    t["strata_counts"] = st[st.source.ne("all")].drop_duplicates(["source", "group"]).pivot_table(
        index="source", columns="group", values="rows", aggfunc="first")
    t["strata_gain"] = st[st.source.eq("all") & ~st.method.astype(str).str.startswith("fixed_")].pivot_table(
        index="method", columns="group", values="mean_gain")
    t2 = data["track2"]
    t["tau80"] = t2[np.isclose(t2.tau.fillna(-1), 0.8)].pivot_table(
        index="method", columns="source", values="needed_tier_accuracy_solvable")
    t["tau80_first"] = t2[np.isclose(t2.tau.fillna(-1), 0.8)].pivot_table(index="method", columns="source", values="first")
    t["fixed"] = data["fixed-starts"][data["fixed-starts"].basis.eq("main")].set_index(["source", "start"])[
        ["resolved", "fails", "cost", "first"]]
    bc = data["by-category"]
    t["task_types"] = bc[bc.source.eq("all") & ~bc.method.astype(str).str.startswith("fixed_")].pivot_table(
        index="method", columns="group", values="mean_gain")[[c for c in K.TYPES if c in set(bc.group)]]
    if "seeds" in data:
        t["seeds"] = data["seeds"].pivot_table(index=["method", "seed"], columns="source", values="mean_gain")
    for device in ("cpu", "cuda"):
        if f"latency_{device}" in data:
            t[f"latency_{device}"] = data[f"latency_{device}"]
    for name, df in t.items():
        df.to_csv(out / f"{name}.csv")
    return t


def results_markdown(data: dict, t: dict, figs: list) -> str:
    h, d, m = data["hyp"], data["diag"], data["manifest"]
    L = ["# EV-10 결과 표 (10_report가 만든 파일)", "",
         f"08 test: 평가 행 {m['counts']['evaluated']}, bootstrap {m['bootstrap']['iterations']}회. "
         "이 파일의 숫자는 모두 08·09·08a 출력에서 그대로 옮겼다(`manifest.json`에 입력 SHA).", ""]
    L += ["## 1. 평균 이득 (갈래 ①, 상향 고려 규칙, κ 21개 × 비용 수준 3개 평균)", ""]
    w = t["mean_gain"].reset_index().rename(columns={"method": "방법"})
    w["방법"] = w["방법"].map(lambda x: NAMES.get(x, x))
    L += md_table(w) + ["", "원천 4개 평균의 95% 구간:", ""]
    c = t["mean_gain_macro_ci"].reset_index()
    c["method"] = c["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(c) + ["", "![mean gain](figures/mean-gain.png)", "", "![gain by kappa](figures/gain-by-kappa.png)", ""]
    L += ["## 2. κ\\* (비용 50% 수준, 이득 > .01이고 구간 하한 > 0인 가장 작은 κ)", ""]
    k = t["kappa_star_50"].reset_index()
    k["method"] = k["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(k, "{:.2f}") + [""]
    L += ["## 3. 고정 시작 단계 (test)", ""] + md_table(t["fixed"].reset_index()) + [""]
    L += ["## 4. 풀리는 정도별 층", ""] + md_table(t["strata_counts"].reset_index(), "{:,.0f}") + [""]
    s = t["strata_gain"].reset_index()
    s["method"] = s["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(s) + ["", "![strata](figures/strata.png)", ""]
    L += ["## 5. 갈래 ② (τ = .8)", "", "필요 단계 정확도(solvable 층):", ""]
    a = t["tau80"].reset_index()
    a["method"] = a["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(a, "{:.3f}") + ["", "첫 시도 성공:", ""]
    f = t["tau80_first"].reset_index()
    f["method"] = f["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(f, "{:.3f}") + ["", "![tau curves](figures/tau-curves.png)", ""]
    L += ["## 6. 과제 유형별 평균 이득 (원천 합계, 문항 수 가중, 기술 통계)", ""]
    tt = t["task_types"].reset_index()
    tt["method"] = tt["method"].map(lambda x: NAMES.get(x, x))
    L += md_table(tt) + ["", "![task types](figures/task-types.png)", ""]
    L += ["## 7. 가설 판정 (08 `hypotheses.json`)", ""]
    h1 = h["H1"]["result"]["macro"]
    L += [f"- **H1:** 원천 평균 1위 {NAMES[h1['top']]}, kNN {h1['knn_rank']}위, 1위와 구분되지 않는 방법 비율 "
          f"{h1['share_not_distinguishable_from_top']:.0%}. 원천별 1위: " +
          ", ".join(f"{s} {NAMES[h['H1']['result'][s]['top']]}" for s in SOURCES)]
    L += ["- **H2(solvable 층 정확도, 첫 시도 성공; R0-집단 − R1-집단):** " + "; ".join(
        f"{s} {v['needed_tier_accuracy_solvable']['diff_r0_minus_r1']:+.3f} / {v['first']['diff_r0_minus_r1']:+.3f}"
        for s, v in h["H2"]["result"].items())]
    L += ["- **H3(IRT 난이도 Spearman):** " + "; ".join(f"{s} {v['spearman']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
                                                    for s, v in h["H3"]["result"].items())]
    L += ["- **H4(재학습 − 공개본, 50% 수준 모든 κ에서 > .01):** " + ", ".join(
        f"{s} {'지지' if v['supported'] else '아님'}" for s, v in h["H4"]["result"].items())]
    for key, v in h["H5"]["result"].items():
        L += [f"- **H5 {key} (seed 42):** " + ", ".join(f"{s} {x['diff']:+.3f}" for s, x in v["per_source"].items())
              + f" → {'지지' if v['supported'] else '아님'}"]
    if "h5_seeds" in data:
        hs = data["h5_seeds"]
        L += ["- **H5 R1 (seed 42·17·73 평균, 09):** " + ", ".join(
            f"{s} {v['diff_seed_mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for s, v in hs["per_source"].items())
              + f" → {'지지' if hs['supported'] else '아님'}"]
    else:
        L += ["- **H5 seed 반복(09):** 아직 없음"]
    L += ["- **H6(v1·v2 순위 Kendall τ):** " + ", ".join(f"{s} {v['kendall_tau']:.2f}" for s, v in h["H6"]["result"].items())]
    L += [f"- **H7:** κ\\*가 없는(어떤 κ에서도 못 이기는) 라우터·원천·수준 비율 {h['H7']['share_without_kappa_star']:.1%}"]
    L += ["- **RT-08:** " + "; ".join(f"{v['basis']}/{v['source']} {v['verdict']}" for v in h["RT-08"]["result"])]
    L += ["", f"RouterArena: R0-집단 예측과 Bloom 라벨 일치 {d['routerarena_r0_group']['agreement_argmax_vs_bloom']:.1%}, "
          f"Spearman {d['routerarena_r0_group']['spearman_expected_class_vs_bloom']:.3f} "
          f"(Context가 붙은 {d['routerarena_rows']['with_context_dropped']}건 제외)."]
    L += ["", "## 8. 라우팅 지연 (08a)", ""]
    if "latency_cpu" in t:
        L += md_table(t["latency_cpu"].reset_index(drop=True).head(40), "{:.1f}")
    else:
        L += ["아직 없음(08a 실행 전)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="normally experiments/results/ev10-router-final-<date>/report in the repo")
    a = p.parse_args(argv)
    data = load(a.run_dir)
    figs = figures(data, a.output_dir / "figures")
    t = tables(data, a.output_dir / "tables")
    (a.output_dir / "results.md").write_text(results_markdown(data, t, figs))
    inputs = {f"test/{k}": v for k, v in data["manifest"]["outputs"].items()}
    for sub in ("seeds", "latency"):
        for f in sorted((a.run_dir / sub).glob("*.*")) if (a.run_dir / sub).exists() else []:
            if f.suffix in (".parquet", ".json"):
                inputs[f"{sub}/{f.name}"] = C.sha256_file(f)
    outputs = {str(f.relative_to(a.output_dir)): C.sha256_file(f) for f in sorted(a.output_dir.rglob("*"))
               if f.is_file() and f.name != "manifest.json"}
    (a.output_dir / "manifest.json").write_text(json.dumps({"experiment": "EV-10 10_report", "inputs": inputs,
                                                            "outputs": outputs}, indent=1))
    print(json.dumps({"figures": figs, "tables": sorted(t), "seeds": "seeds" in data,
                      "latency": [k for k in data if k.startswith("latency_")]}, indent=1))


if __name__ == "__main__":
    main()
