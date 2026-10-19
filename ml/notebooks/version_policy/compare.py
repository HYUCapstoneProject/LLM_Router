"""RF-06 03_compare (CPU): quality, compute and efficiency of P1 (version policy) vs P2 (full retraining).

Reads the sealed plan, the experiment tables and every finished round under arms/. Nothing is trained or chosen.

  quality (primary)  dev source-macro-F1 after every round; test source-macro-F1 of the last round (P0 = M0)
  prequential        each probe window k >= 2 predicted by the model in service when it arrived (round k-1)
  routing            predicted label = start tier, escalate on failure with the group pass probabilities and the
                     service cost .2/.6/1.0: first-attempt success, resolution, calls, cost, and first-attempt
                     success minus the fixed-start line at the same cost (descriptive)
  compute            cumulative optimizer steps, T4 fit seconds, samples and tokens processed
  judgement          plan 6: margin .01, question-group paired bootstrap 95% intervals (sources resampled
                     independently, source-macro mean), seed-averaged probabilities when several seeds exist

Writes <run>/compare/: results.json, tables/*.csv, figures/*.png, REPORT-draft.md and results-for-review.zip
(no question text). `--allow-partial` reports whatever rounds exist (for checking mid-run).
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import plan as P

TIERS, LABELS = P.TIERS, P.LABELS
ARM_NAMES = {"p0": "P0 동결(M0)", "p1": "P1 버전 정책", "p2": "P2 전체 재학습", "p2_eq": "P2-eq 같은 step"}
FIGURE_ARMS = {"p0": ("P0 frozen (M0)", "#7f7f7f"), "p1": ("P1 version policy", "#1f77b4"),
               "p2": ("P2 full retrain", "#d62728"), "p2_eq": ("P2-eq same steps", "#ff7f0e")}  # Colab has no Korean font


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def outcomes(p: np.ndarray, start: np.ndarray, cost: np.ndarray) -> dict:
    """Start at `start`, escalate on failure, independent draws (same as EV-10 lock.outcomes)."""
    n = len(p)
    reach, spent, fails = np.ones(n), np.zeros(n), np.zeros(n)
    for t in range(3):
        on = t >= start
        spent += np.where(on, reach * cost[t], 0.0)
        fails += np.where(on, reach * (1 - p[:, t]), 0.0)
        reach = np.where(on, reach * (1 - p[:, t]), reach)
    return {"first": p[np.arange(n), start], "resolved": 1 - reach, "cost": spent, "calls": (1 - reach) + fails}


def fixed_line(points: list, x: float) -> float:
    """Best first-attempt success over fixed starts and their mixtures at expected cost x.

    `points` are (cost, first) of the three fixed starts. Below the cheapest start the cheapest one is used."""
    pts = sorted(points)
    best = max(f for c, f in pts if c <= x) if any(c <= x for c, _ in pts) else pts[0][1]
    for (c1, f1), (c2, f2) in ((a, b) for a in pts for b in pts if a[0] < b[0]):
        if c1 <= x <= c2:
            best = max(best, f1 + (f2 - f1) * (x - c1) / (c2 - c1))
    return float(best)


def bootstrap_source_macro_f1(frame: pd.DataFrame, preds: dict, iterations: int, seed: int) -> dict:
    """Paired group bootstrap of source-macro-F1 for every prediction vector in `preds` (same draws)."""
    rng = np.random.default_rng(seed)
    y = P.label_ids(frame["difficulty_label"])
    draws = {name: np.zeros(iterations) for name in preds}
    sources = sorted(frame.source_dataset.unique())
    for source in sources:
        mask = frame.source_dataset.eq(source).to_numpy()
        codes, _ = pd.factorize(frame.loc[mask, "group_id"])
        g = codes.max() + 1
        counts = rng.multinomial(g, np.full(g, 1.0 / g), size=iterations)
        for name, pred in preds.items():
            cell = y[mask] * 3 + np.asarray(pred)[mask]
            per_group = np.zeros((g, 9))
            np.add.at(per_group, (codes, cell), 1)
            conf = (counts @ per_group).reshape(iterations, 3, 3)
            draws[name] += P.macro_f1_from_confusion(conf) / len(sources)
    return draws


def interval(draws: np.ndarray, confidence: float) -> list:
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
    return [float(lo), float(hi)]


def noninferiority(delta: float, ci: list, margin: float) -> str:
    if ci[0] > margin and delta > margin:
        return "P1 우위(+.01 초과)"
    if ci[0] > -margin:
        return "비열등(손실 .01 이내)"
    if ci[1] < -margin:
        return "열등(.01 넘게 낮음)"
    return "미확인(구간이 -.01에 걸침, seed 추가)"


def improvement(delta: float, ci: list, margin: float) -> str:
    return "효과 있음" if delta > margin and ci[0] > 0 else "효과 확인 안 됨"


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def logits_for(path: Path, ids: pd.Series) -> np.ndarray:
    s = pd.read_parquet(path).set_index("sample_id").loc[ids]
    return s[[f"score_{t}" for t in TIERS]].to_numpy(float)


def load_rounds(run_dir: Path, arm: str, rounds: int) -> dict:
    """{seed: [receipt per finished round]} for the leading run of finished rounds."""
    out = {}
    base = run_dir / "arms" / arm
    for folder in sorted(base.glob("seed*")) if base.exists() else []:
        done = []
        for k in range(1, rounds + 1):
            r = folder / f"round{k:02d}"
            if not (r / "_SUCCESS").exists():
                break
            receipt = json.loads((r / "receipt.json").read_text())
            for name, digest in receipt["outputs"].items():
                if P.sha256_file(r / name) != digest:
                    raise SystemExit(f"{r / name} changed after the round finished")
            done.append({**receipt, "dir": r})
        if done:
            out[int(folder.name[4:])] = done
    return out


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def run(run_dir: Path, data_dir: Path, output_dir: Path, allow_partial: bool) -> dict:
    cfg = json.loads((P.HERE / "config.json").read_text())
    judge = cfg["judgement"]
    margin, conf_level = judge["margin"], judge["confidence"]
    sealed = P.load_plan(run_dir)
    plan = sealed["plan"]
    rounds = plan["config"]["stream"]["rounds"]
    data = P.load_data(data_dir)
    dev, test, probe = data["dev"], data["test"], sealed["probe"]
    if plan["smoke"]:
        dev = dev[dev.sample_id.isin(pd.read_parquet(sealed["dir"] / "m0-dev-scores.parquet").sample_id)]
        test = test[test.sample_id.isin(sealed["m0-test-scores"].sample_id)]
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    cost = np.array([cfg["service_cost"][t] for t in TIERS])
    arms = {arm: load_rounds(run_dir, arm, rounds) for arm in ("p1", "p2", "p2_eq")}
    complete = {arm: {s: r for s, r in v.items() if len(r) == rounds} for arm, v in arms.items()}
    if not allow_partial and not (complete["p1"] and complete["p2"]):
        raise SystemExit("P1 and P2 must finish every round for at least one seed (use --allow-partial to peek)")

    m0_dev = softmax(logits_for(sealed["dir"] / "m0-dev-scores.parquet", dev.sample_id))
    m0_test = softmax(logits_for(sealed["dir"] / "m0-test-scores.parquet", test.sample_id))
    m0_probe = softmax(logits_for(sealed["dir"] / "m0-probe-scores.parquet", probe.sample_id))

    # per round: dev quality, prequential and compute
    rows = []
    for k in range(0, rounds + 1):
        rows.append({"arm": "p0", "seed": None, "round": k, "version": "1.0",
                     "dev": P.source_macro_f1(dev, m0_dev.argmax(1))["source_macro_f1"],
                     "steps": 0, "fit_seconds": 0.0, "samples": 0, "tokens": 0})
    for arm, by_seed in arms.items():
        for seed, receipts in by_seed.items():
            cum = {"steps": 0, "fit_seconds": 0.0, "samples": 0, "tokens": 0}
            for r in receipts:
                cum["steps"] += r["optimizer_steps"]
                cum["fit_seconds"] += r["fit_seconds"]
                cum["samples"] += r["planned_samples"]
                cum["tokens"] += r["tokens_per_epoch"] * r["epochs"]
                p_dev = softmax(logits_for(r["dir"] / "dev-scores.parquet", dev.sample_id))
                rows.append({"arm": arm, "seed": seed, "round": r["round"], "version": r["version"],
                             "dev": P.source_macro_f1(dev, p_dev.argmax(1))["source_macro_f1"],
                             "round_steps": r["optimizer_steps"], "round_fit_seconds": r["fit_seconds"],
                             "train_rows": r["train_rows"], **cum})
    per_round = pd.DataFrame(rows)

    preq = []
    for k in range(2, rounds + 1):
        window = probe[probe.window.eq(k)].reset_index(drop=True)
        mask = probe.window.eq(k).to_numpy()
        preq.append({"arm": "p0", "seed": None, "window": k,
                     "f1": P.source_macro_f1(window, m0_probe[mask].argmax(1))["source_macro_f1"]})
        for arm, by_seed in arms.items():
            for seed, receipts in by_seed.items():
                if len(receipts) >= k - 1:
                    r = receipts[k - 2]
                    p_win = softmax(logits_for(r["dir"] / "probe-scores.parquet", window.sample_id))
                    preq.append({"arm": arm, "seed": seed, "window": k,
                                 "f1": P.source_macro_f1(window, p_win.argmax(1))["source_macro_f1"]})
    preq = pd.DataFrame(preq)

    # final test, seed-averaged probabilities
    final_probs = {"p0": m0_test}
    for arm, by_seed in complete.items():
        if by_seed:
            final_probs[arm] = np.mean([softmax(logits_for(r[-1]["dir"] / "test-scores.parquet", test.sample_id))
                                        for r in by_seed.values()], axis=0)
    preds = {arm: p.argmax(1) for arm, p in final_probs.items()}
    draws = bootstrap_source_macro_f1(test, preds, judge["bootstrap_iterations"], judge["bootstrap_seed"])
    y = P.label_ids(test["difficulty_label"])
    p_true = test[[f"success_{t}" for t in TIERS]].to_numpy(float)
    final = []
    for arm, pred in preds.items():
        q = P.source_macro_f1(test, pred)
        o = outcomes(p_true, pred, cost)
        route = {}
        for source, g in test.groupby("source_dataset"):
            m = test.source_dataset.eq(source).to_numpy()
            fixed = [(float(outcomes(p_true[m], np.full(m.sum(), s), cost)["cost"].mean()),
                      float(p_true[m][:, s].mean())) for s in range(3)]
            route[source] = float(o["first"][m].mean()) - fixed_line(fixed, float(o["cost"][m].mean()))
        final.append({"arm": arm, "seeds": len(complete.get(arm, {})) if arm != "p0" else 0,
                      "test_source_macro_f1": q["source_macro_f1"], **{f"f1_{s}": v for s, v in q["by_source"].items()},
                      "accuracy": float((pred == y).mean()), "hard_recall": float((pred[y == 2] == 2).mean()),
                      "over": float((pred > y).mean()), "under": float((pred < y).mean()),
                      "first": float(o["first"].mean()), "resolved": float(o["resolved"].mean()),
                      "calls": float(o["calls"].mean()), "cost": float(o["cost"].mean()),
                      "first_gain_same_cost": float(np.mean(list(route.values())))})
    final = pd.DataFrame(final)

    obs = final.set_index("arm")["test_source_macro_f1"]
    pairs = [(a, b) for a, b in (("p1", "p2"), ("p1", "p0"), ("p2", "p0"), ("p2_eq", "p1"), ("p2_eq", "p2"))
             if a in draws and b in draws]
    verdicts = {}
    for a, b in pairs:
        d = float(obs[a] - obs[b])
        ci = interval(draws[a] - draws[b], conf_level)
        verdicts[f"{a}-{b}"] = {"delta": d, "ci": ci,
                                "verdict": noninferiority(d, ci, margin) if (a, b) == ("p1", "p2") else improvement(d, ci, margin)}

    # compute and efficiency (seed means)
    seed_mean = per_round.groupby(["arm", "round"], dropna=False).mean(numeric_only=True).reset_index()
    efficiency = {}
    if "p1" in seed_mean.arm.values and "p2" in seed_mean.arm.values:
        last = min(seed_mean[seed_mean.arm.eq(a)]["round"].max() for a in ("p1", "p2"))
        p1_last = seed_mean[(seed_mean.arm == "p1") & (seed_mean["round"] == last)].iloc[0]
        p2_last = seed_mean[(seed_mean.arm == "p2") & (seed_mean["round"] == last)].iloc[0]
        target = float(p2_last["dev"] - margin)
        reach = {}
        for arm in [a for a in ("p1", "p2", "p2_eq") if complete[a]]:
            s = seed_mean[(seed_mean.arm == arm) & (seed_mean.dev >= target)].sort_values("round")
            reach[arm] = None if s.empty else {"round": int(s.iloc[0]["round"]), "steps": float(s.iloc[0]["steps"]),
                                               "fit_hours": float(s.iloc[0]["fit_seconds"]) / 3600}
        efficiency = {"round": int(last), "steps_ratio_p2_p1": float(p2_last.steps / max(p1_last.steps, 1)),
                      "fit_ratio_p2_p1": float(p2_last.fit_seconds / max(p1_last.fit_seconds, 1e-9)),
                      "tokens_ratio_p2_p1": float(p2_last.tokens / max(p1_last.tokens, 1)),
                      "p1_fit_hours": float(p1_last.fit_seconds) / 3600, "p2_fit_hours": float(p2_last.fit_seconds) / 3600,
                      "dev_target": target, "first_round_reaching_target": reach}
    dev_gap = []
    for k in sorted(set(seed_mean["round"])):
        a = seed_mean[(seed_mean.arm == "p1") & (seed_mean["round"] == k)]
        b = seed_mean[(seed_mean.arm == "p2") & (seed_mean["round"] == k)]
        if len(a) and len(b) and k > 0:
            dev_gap.append({"round": int(k), "version": P.version_name(int(k), cfg["version_policy"]["cycle_size"]),
                            "p1_minus_p2_dev": float(a.dev.iloc[0] - b.dev.iloc[0])})

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    tables.mkdir(exist_ok=True)
    per_round.to_csv(tables / "per-round.csv", index=False)
    preq.to_csv(tables / "prequential.csv", index=False)
    final.to_csv(tables / "final-test.csv", index=False)
    pd.DataFrame(dev_gap).to_csv(tables / "dev-gap-by-round.csv", index=False)
    figures(per_round, preq, output_dir / "figures")
    results = {"experiment": plan["experiment"], "smoke": plan["smoke"], "partial": not (complete["p1"] and complete["p2"]),
               "seeds": {arm: sorted(v) for arm, v in complete.items()}, "selection": plan["selection"],
               "rows": plan["rows"], "window_rows": plan["window_rows"], "verdicts": verdicts, "efficiency": efficiency,
               "dev_gap_by_round": dev_gap, "final_test": final.to_dict("records"),
               "plan_sha256": P.sha256_file(sealed["dir"] / "plan.json")}
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    (output_dir / "REPORT-draft.md").write_text(report(results, plan, final, preq), encoding="utf-8")
    with zipfile.ZipFile(output_dir / "results-for-review.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".md", ".png"}:
                z.write(path, path.relative_to(output_dir))
        z.write(sealed["dir"] / "plan.json", "plan.json")
    return results


def figures(per_round: pd.DataFrame, preq: pd.DataFrame, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    mean = per_round.groupby(["arm", "round"], dropna=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, g in mean.groupby("arm"):
        g = g.sort_values("round")
        name, color = FIGURE_ARMS[arm]
        axes[0].plot(g["round"], g["dev"], marker="o", label=name, color=color)
        if arm != "p0":
            axes[1].plot(g["fit_seconds"] / 3600, g["dev"], marker="o", label=name, color=color)
    axes[0].set(xlabel="round", ylabel="dev source-macro-F1", title="Quality by round")
    axes[1].set(xlabel="cumulative T4 training hours", ylabel="dev source-macro-F1", title="Quality vs compute")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "quality-and-compute.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, g in mean[mean.arm != "p0"].groupby("arm"):
        name, color = FIGURE_ARMS[arm]
        ax.bar(g["round"] + {"p1": -.27, "p2": 0, "p2_eq": .27}[arm], g["round_fit_seconds"] / 60, width=.27,
               label=name, color=color)
    ax.set(xlabel="round", ylabel="T4 training minutes", title="Training time per round")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "time-per-round.png", dpi=150)
    plt.close(fig)


def report(results: dict, plan: dict, final: pd.DataFrame, preq: pd.DataFrame) -> str:
    sel = plan["selection"]
    lines = ["# RF-06 버전 정책 대 전체 재학습 — 결과 초안", "",
             f"실험 코드: RF-06. {'**smoke 실행이며 결과가 아니다.**' if results['smoke'] else ''}"
             f"{' **일부 round만 끝난 중간 집계다.**' if results['partial'] else ''}", "",
             "## 1. 설정", "",
             f"- M0: pilot 곡선에서 n0 = {sel['n0']}행(목표 {sel['target']:.4f}), stream {sel['stream_rows']}행, "
             f"pilot 이득 {sel['pilot_gain']:.4f}",
             f"- window {results['window_rows']}행 × {plan['config']['stream']['rounds']} round, "
             f"seed {results['rows']['seed']} / probe {results['rows']['probe']} / reserve {results['rows']['reserve']}",
             f"- 완료 seed: {results['seeds']}", "",
             "## 2. 최종 test (seed 평균 확률)", "",
             "| arm | source-macro-F1 | 정확도 | hard 재현율 | 높게 | 낮게 | 첫 시도 | 비용 | 같은 비용 첫 시도 이득 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in final.to_dict("records"):
        lines.append(f"| {ARM_NAMES.get(r['arm'], r['arm'])} | {r['test_source_macro_f1']:.4f} | {r['accuracy']:.3f} | "
                     f"{r['hard_recall']:.3f} | {r['over']:.3f} | {r['under']:.3f} | {r['first']:.3f} | {r['cost']:.3f} | "
                     f"{r['first_gain_same_cost']:+.3f} |")
    lines += ["", "## 3. 판정 (질문 그룹 paired bootstrap 95%)", "", "| 비교 | 차이 | 95% 구간 | 판정 |", "|---|---:|---|---|"]
    for name, v in results["verdicts"].items():
        a, b = name.split("-")
        lines.append(f"| {ARM_NAMES.get(a, a)} − {ARM_NAMES.get(b, b)} | {v['delta']:+.4f} | "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['verdict']} |")
    e = results["efficiency"]
    if e:
        lines += ["", "## 4. 계산량 (round " + str(e["round"]) + "까지 누적, seed 평균)", "",
                  f"- 학습 시간: P1 {e['p1_fit_hours']:.2f}h, P2 {e['p2_fit_hours']:.2f}h (P2/P1 {e['fit_ratio_p2_p1']:.2f}배)",
                  f"- optimizer step P2/P1 {e['steps_ratio_p2_p1']:.2f}배, 처리 토큰 P2/P1 {e['tokens_ratio_p2_p1']:.2f}배",
                  f"- dev 목표(P2 마지막 − .01 = {e['dev_target']:.4f})에 처음 닿은 round: "
                  + ", ".join(f"{ARM_NAMES.get(a, a)} {v['round']} ({v['fit_hours']:.2f}h)" if v else f"{ARM_NAMES.get(a, a)} 못 닿음"
                              for a, v in e["first_round_reaching_target"].items())]
    lines += ["", "## 5. round별 dev 차이 (P1 − P2)", "", "| round | 버전 | P1 − P2 |", "|---:|---|---:|"]
    lines += [f"| {g['round']} | {g['version']} | {g['p1_minus_p2_dev']:+.4f} |" for g in results["dev_gap_by_round"]]
    if len(preq):
        mean = preq.groupby("arm")["f1"].mean()
        lines += ["", "## 6. prequential (window 2~K 평균 source-macro-F1)", "",
                  ", ".join(f"{ARM_NAMES.get(a, a)} {v:.4f}" for a, v in mean.items())]
    lines += ["", "그림: `figures/quality-and-compute.png`, `figures/time-per-round.png`. 표: `tables/`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args(argv)
    if a.output_dir.exists() and not a.allow_partial:
        shutil.rmtree(a.output_dir)
    out = run(a.run_dir, a.data_dir, a.output_dir, a.allow_partial)
    print(json.dumps({k: out[k] for k in ("verdicts", "efficiency", "seeds", "partial")}, ensure_ascii=False, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
