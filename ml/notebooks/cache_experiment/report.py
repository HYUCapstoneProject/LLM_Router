"""Tables, figures and the draft report from sealed stage outputs. No new inference.

End-to-end latency here is a simulation: measured local stage p50s, an assumed
cross-region round trip per FastAPI call, and — only for xRouteBench rows — the
recorded generation time. RouterBench has no generation time, so its E2E is
reported without generation and labelled as such.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cache_experiment as ce

MAIN_METRICS = ["hit_rate", "exact_hit_rate", "semantic_hit_rate", "false_reuse_among_hits", "false_reuse_per_request",
                "unknown_among_hits", "call_reduction", "llm_cost_reduction", "quality_change_conservative",
                "quality_change_optimistic"]


def d1_table(summary, arms=ce.ARMS):
    base = summary[summary.arm.isin(arms)]
    grouped = base.groupby(["repeat_fraction", "arm"])[MAIN_METRICS]
    table = grouped.mean().add_suffix("_mean").join(grouped.min().add_suffix("_min")).join(grouped.max().add_suffix("_max"))
    return table.reset_index()


def capacity_table(summary):
    cap = summary[summary.arm.str.contains("-cap") | summary.arm.isin(["C1", "C2", "C3"])].copy()
    cap["base_arm"] = cap.arm.str.split("-cap").str[0]
    cap["capacity"] = cap.capacity.fillna(np.inf)
    return cap.groupby(["repeat_fraction", "base_arm", "capacity"])[["hit_rate", "call_reduction", "false_reuse_among_hits",
                                                                      "evictions"]].mean().reset_index()


def stage_p50(models_latency, pg_latency, cache_size):
    p = {r.stage: r.p50_ms for r in models_latency.itertuples()}
    pg = pg_latency[pg_latency.cache_size.eq(cache_size)]
    p.update({r.stage: r.p50_ms for r in pg.itertuples()})
    return p


def e2e_simulation(log, p50, rtt_ms, rerank_stage="rerank_batch5", route_stage="route"):
    """Per-request latency estimate in ms from the ops each request performed."""
    ms = (log.exact_lookups * p50.get("pg_exact_lookup", np.nan)
          + log.embeds * (p50["embed"] + rtt_ms)
          + log.vector_searches * p50.get("pg_vector_search", np.nan)
          + (log.rerank_pairs > 0) * (p50.get(rerank_stage, np.nan) + rtt_ms)
          + log.routed * (p50.get(route_stage, 0.0) + rtt_ms))
    generation = (log.llm_latency_s * 1000.0).where(log.llm_called, 0.0)
    return pd.DataFrame({"position": log.position, "source_dataset": log.source_dataset, "decision": log.decision,
                         "local_ms": ms, "generation_ms": generation})


def threshold_exposure(logs, config, eps=1e-4):
    """Decisions whose score sits within eps of a threshold in any production arm.

    Inference ran on a GPU while the service runs on CPU; the parity gate bounds the
    numeric gap well below eps, so only these decisions could flip. Counted, not fixed.
    """
    c3 = config["policies"]["C3"]
    floor = min(config["policies"]["C2"]["thresholds"].values())
    rows = []
    for arm, log in logs.items():
        cos, score = log.cosine_best.to_numpy(float), log.rerank_score.to_numpy(float)
        rows.append({"arm": arm, "requests": int(len(log)),
                     "near_cosine_floor": int(np.sum(np.abs(cos - floor) < eps)),
                     "near_rerank_cosine": int(np.sum(np.abs(cos - c3["rerank_similarity"]) < eps)),
                     "near_rerank_score": int(np.sum(np.abs(score - c3["rerank_score"]) < eps))})
    return pd.DataFrame(rows)


def figures(run, d1_summary, d2_metrics, cumulative_trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = run.path("report", "figures")
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in [a for a in (*ce.ARMS[1:], "C_cal") if a in set(d1_summary.arm)]:
        s = d1_summary[d1_summary.arm.eq(arm)].groupby("repeat_fraction").call_reduction.agg(["mean", "min", "max"])
        ax.errorbar(s.index * 100, s["mean"] * 100, yerr=[(s["mean"] - s["min"]) * 100, (s["max"] - s["mean"]) * 100],
                    marker="o", capsize=3, label=arm)
    ax.set(xlabel="exact repeat share of requests (%)", ylabel="LLM calls avoided vs C0 (%)",
           title="D1: calls avoided by repeat share (mean, seed range)")
    ax.legend()
    fig.tight_layout()
    paths.append(out / "d1-call-reduction.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    if d2_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        at = d1_summary[d1_summary.repeat_fraction.eq(0.3)]
        for name, m in d2_metrics.items():
            if "overall" not in m:
                continue
            reduction = at[at.arm.eq("C_cal" if name == "calibrated" else name)].call_reduction.mean()
            o = m["overall"]
            f = o["false_reuse_among_hits"]
            if f is None or reduction != reduction:
                continue
            ax.errorbar([reduction * 100], [f * 100], yerr=[[max(f - o["false_reuse_ci95_low"], 0) * 100],
                                                          [(o["false_reuse_ci95_high"] - f) * 100]], marker="o", capsize=3,
                        label=name)
        ax.set(xlabel="D1 calls avoided at 30% repeats (%)", ylabel="D2 test false reuse among hits (%)",
               title="Safety vs savings (95% Clopper-Pearson)")
        ax.legend()
        fig.tight_layout()
        paths.append(out / "safety-vs-savings.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, log in cumulative_trace.items():
        axes[0].plot(log.position, log.llm_cost.cumsum(), label=arm)
        axes[1].plot(log.position, log.relation_error.astype(int).cumsum(), label=arm)
    axes[0].set(xlabel="request order", ylabel="cumulative recorded LLM cost (USD)")
    axes[1].set(xlabel="request order", ylabel="cumulative false reuse")
    axes[0].legend()
    fig.tight_layout()
    paths.append(out / "cumulative.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def build(run, representative_trace=None):
    cfg = run.config
    prepare = ce.read(run.require("prepare", "prepare-manifest.json"))
    d1 = ce.read_parquet(run.require("d1", "summary-production.parquet"))
    calibrated = run.path("d1", "summary-calibrated.parquet")
    if calibrated.exists():
        d1 = pd.concat([d1, ce.read_parquet(calibrated).query("arm == 'C_cal'")], ignore_index=True)
    d2 = ce.read(run.path("d2", "test-metrics.json")) if run.path("d2", "test-metrics.json").exists() else None
    tables = run.path("report", "tables")
    tables.mkdir(parents=True, exist_ok=True)
    main = d1_table(d1, [a for a in (*ce.ARMS, "C_cal") if a in set(d1.arm)])
    main.to_csv(tables / "d1-main.csv", index=False)
    capacity_table(d1).to_csv(tables / "d1-capacity.csv", index=False)
    if run.path("d1", "bootstrap-production.parquet").exists():
        ce.read_parquet(run.path("d1", "bootstrap-production.parquet")).to_csv(tables / "d1-bootstrap.csv", index=False)

    latency = e2e = None
    if run.path("latency", "models.parquet").exists() and run.path("latency", "pgvector.parquet").exists():
        models_latency = ce.read_parquet(run.path("latency", "models.parquet"))
        pg_latency = ce.read_parquet(run.path("latency", "pgvector.parquet"))
        latency = pd.concat([models_latency, pg_latency], ignore_index=True)
        latency.to_csv(tables / "latency-stages.csv", index=False)

    trace = representative_trace or ce.trace_id(0.3, cfg["d1"]["trace_seeds"][0])
    cumulative = {a: ce.read_parquet(run.require("d1", "traces", trace, a, "requests.parquet")) for a in ce.ARMS}
    if latency is not None:
        p50 = stage_p50(models_latency, pg_latency, max(cfg["latency"]["cache_sizes"]))
        rtt = cfg["latency"]["assumed_cross_region_rtt_ms"]
        e2e = pd.concat([e2e_simulation(log, p50, rtt).assign(arm=a) for a, log in cumulative.items()], ignore_index=True)
        xrb = e2e[e2e.source_dataset.eq("xroutebench")].assign(total_ms=lambda f: f.local_ms + f.generation_ms)
        pct = [.5, .95]
        summary = pd.concat([
            e2e.groupby("arm").local_ms.describe(percentiles=pct)[["50%", "95%", "mean"]].add_prefix("local_without_generation_"),
            xrb.groupby("arm").total_ms.describe(percentiles=pct)[["50%", "95%", "mean"]].add_prefix("xroutebench_with_recorded_generation_")],
            axis=1)
        summary.to_csv(tables / f"e2e-simulation-{trace}.csv")
    paths = figures(run, d1, d2, cumulative)
    exposure = pd.concat([threshold_exposure({a: ce.read_parquet(run.require("d1", "traces", t, a, "requests.parquet"))
                                              for a in ("C2", "C3")}, cfg).assign(trace_id=t)
                          for t in (ce.trace_id(r, seed) for seed in cfg["d1"]["trace_seeds"] for r in cfg["d1"]["repeat_fractions"])],
                         ignore_index=True)
    exposure.to_csv(tables / "threshold-exposure.csv", index=False)

    dataset = ce.read(run.path("d2", "dataset-manifest.json")) if run.path("d2", "dataset-manifest.json").exists() else {}
    mode = dataset.get("annotation_mode", "two_annotators" if dataset else None)
    pricing = cfg["pricing"]
    calibration = calibration_lines(run)
    lines = [f"# 캐시 실험 결과 초안 ({cfg['experiment_id']} / {run.run_id})", "",
             "`06_report.ipynb`가 봉인된 산출물로 만든 **초안**이다. 해석과 결론은 사람이 검토해 채운다.", "",
             "## 준비", "",
             f"- pool {prepare['pool']['rows']}행, D1 pool {prepare['d1_pool_rows']}행, D2 원질문 {prepare['d2_originals']['groups']}그룹",
             f"- 운영 패리티: {json.dumps(prepare['production_parity'], ensure_ascii=False)}",
             f"- 채점기 자기 일치: {json.dumps(prepare['scorer_selfcheck'], ensure_ascii=False)}",
             f"- 추론 장치: {json.dumps(prepare.get('devices'), ensure_ascii=False)}; 임계값 ±1e-4 안의 판정 수(D1 C2·C3 합): "
             f"{int(exposure[['near_cosine_floor', 'near_rerank_cosine', 'near_rerank_score']].to_numpy().sum())}", "",
             "## D1 (운영 설정, seed 평균)", "", _md(main.filter(regex="repeat_fraction|^arm$|_mean$")), "",
             "## D2 보정 (calibration 100그룹)", ""] + calibration + ["",
             "## D2 test", "",
             f"판정 방식: {mode}" + (" — **LLM 판정 1개 기준. 사람 검증된 안전성으로 주장하지 않는다.**" if mode == "single_llm" else ""),
             "", "```json", json.dumps(d2, indent=1, ensure_ascii=False) if d2 else "아직 실행하지 않음", "```", "",
             "## 지연 (로컬 실측)", "", _md(latency) if latency is not None else "아직 측정하지 않음", "",
             f"연산 단가: {pricing['compute_usd_per_hour']} USD/h ({pricing['compute_price_source']}, "
             f"{pricing['compute_price_snapshot_date']})", "", "## 그림", ""] + [f"- `{p.relative_to(run.root)}`" for p in paths]
    run.path("report", "REPORT-draft.md").write_text("\n".join(lines) + "\n")
    return main, d2, latency, e2e


def calibration_lines(run):
    """What 03 froze, and how close the grid came when nothing met the limit."""
    if not run.path("d2", "frozen-policy.json").exists():
        return ["아직 보정하지 않음"]
    policy = ce.read(run.path("d2", "frozen-policy.json"))
    lines = [f"- 상태: `{policy['status']}` (상한 false reuse among hits ≤ {policy['limit']})",
             f"- 봉인 정책: `{json.dumps(policy['spec'], ensure_ascii=False)}` → 운영 형태 "
             f"`{json.dumps(policy['operational_policy'], ensure_ascii=False)}`"]
    if policy["spec"] is None:
        lines.append("- 상한을 만족하는 의미 캐시 정책이 없어 C_cal은 exact-only다. D1의 C_cal은 C1과 같은 정책이다.")
    grid = run.path("d2", "calibration-grid.parquet")
    if grid.exists():
        g = ce.read_parquet(grid).query("hits > 0")
        best = g.sort_values(["false_reuse_among_hits", "reusable_hits"], ascending=[True, False]).iloc[0]
        cols = ["kind", "k", "cosine", "rerank_similarity", "rerank_score", "hits", "false_reuse",
                "false_reuse_among_hits", "reusable_hits", "paraphrase_hit_rate", "hard_negative_hit_rate"]
        lines += [f"- 격자 {len(g)}개 설정 중 false reuse among hits 최저 설정:", "", _md(best[cols].to_frame().T)]
    return lines


def _md(frame):
    if frame is None or frame.empty:
        return ""
    frame = frame.copy()
    for c in frame.columns:
        if frame[c].dtype.kind == "f":
            frame[c] = frame[c].map(lambda v: f"{v:.4f}")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    return "\n".join([header, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)])
