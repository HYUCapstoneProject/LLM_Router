#!/usr/bin/env python3
"""Render a local R0/R1/R2 report from sealed validation evidence, without training.

Example:
  python ml/scripts/render-routing-objective-report.py --runs-dir /path/full \
    --validation-lock /path/validation-lock.json --output-dir /path/report

Dependencies: numpy, pandas, pyarrow, scikit-learn, matplotlib and the ML source.
Input model weights are not required; their size is reported from the manifest.
This command never publishes, approves, changes CURRENT, or calls a cloud API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_router_labeling.objective_comparison.evaluation import (
    evaluate_penalty_curve,
    route_actions,
    summarize_curve,
)

METHODS = ("ordinal", "pointwise", "pairwise")
SOURCES = ("routerbench", "xroutebench")
TIERS = ("low", "medium", "high")
LABELS = {"ordinal": "R0 CE", "pointwise": "R1 success BCE", "pairwise": "R2 pairwise"}
COLORS = {"ordinal": "#0072B2", "pointwise": "#D55E00", "pairwise": "#009E73"}
ARTIFACTS = ("metrics.json", "validation-scores.parquet", "validation-curve.parquet")
CONTRACT_FIELDS = (
    "experiment_version",
    "experiment_config_sha256",
    "outcome_dataset_version",
    "outcome_manifest_sha256",
    "base_training_sha256",
    "training_data_sha256",
    "validation_data_sha256",
    "common_training_config",
    "runtime_version",
    "runtime_contract_sha256",
    "seed",
)


class EvidenceError(ValueError):
    """The available evidence cannot support this completed comparison report."""


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object: {path}")
    return value


def number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise EvidenceError(f"Missing numeric evidence: {name}")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise EvidenceError(f"Invalid numeric evidence: {name}")
    return result


def equal_number(actual: Any, expected: Any, name: str) -> None:
    if actual is None and expected is None:
        return
    if not math.isclose(
        number(actual, name), number(expected, name), rel_tol=1e-7, abs_tol=1e-9
    ):
        raise EvidenceError(f"Metric mismatch: {name}: {actual!r} != {expected!r}")


def optional_number(value: Any) -> float | None:
    return None if value is None else number(value, "optional metric", nonnegative=True)


def load_method(root: Path, method: str) -> dict[str, Any]:
    folder = root / method
    manifest = read_json(folder / "manifest.json")
    if (
        manifest.get("schema_version") != "routing-objective-run-v1"
        or manifest.get("method") != method
    ):
        raise EvidenceError(f"Unsupported run manifest: {folder}")
    if any(
        int(manifest.get(name, 0))
        for name in (
            "maximum_train_rows",
            "maximum_validation_rows",
            "maximum_train_steps",
        )
    ):
        raise EvidenceError(f"A smoke/pilot cannot be reported as a full run: {method}")
    if (
        manifest.get("test_data_accessed") is not False
        or manifest.get("routerarena_accessed") is not False
    ):
        raise EvidenceError(
            f"The selection run must exclude sealed evaluations: {method}"
        )
    marker = folder / "_SUCCESS"
    if marker.exists() and marker.read_text().strip() != digest(
        folder / "manifest.json"
    ):
        raise EvidenceError(f"Completion marker mismatch: {method}")
    for name in ARTIFACTS:
        path = folder / name
        reference = manifest.get("artifacts", {}).get(name, {})
        if (
            reference.get("sha256") != digest(path)
            or reference.get("size_bytes") != path.stat().st_size
        ):
            raise EvidenceError(f"Artifact checksum or size mismatch: {method}/{name}")
    metrics = read_json(folder / "metrics.json")
    if metrics.get("method") != method or metrics.get("seed") != manifest.get("seed"):
        raise EvidenceError(f"Metrics identify another run: {method}")
    planned = (
        metrics.get("training_metrics", {}).get("optimization", {}).get("planned_steps")
    )
    if number(metrics.get("optimizer_steps"), "optimizer_steps") != number(
        planned, "planned_steps"
    ):
        raise EvidenceError(f"Incomplete training budget: {method}")
    frame = pd.read_parquet(folder / "validation-scores.parquet")
    required = [
        "sample_id",
        "source_dataset",
        *[
            f"{kind}_{tier}"
            for kind in ("score", "quality", "actual_cost", "decision_cost")
            for tier in TIERS
        ],
    ]
    if set(required) - set(frame.columns) or frame.empty:
        raise EvidenceError(f"Missing validation evidence columns: {method}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise EvidenceError(f"Validation IDs must be present and unique: {method}")
    if set(frame["source_dataset"]) != set(SOURCES) or len(frame) != metrics.get(
        "validation_rows"
    ):
        raise EvidenceError(f"This report requires both integrated-A sources: {method}")
    if "evaluation_split" in frame and set(frame["evaluation_split"]) != {"validation"}:
        raise EvidenceError(f"Only D_dev validation rows are allowed: {method}")
    if not np.isfinite(frame[required[2:]].to_numpy(float)).all():
        raise EvidenceError(f"Non-finite outcomes or scores: {method}")
    curve = pd.read_parquet(folder / "validation-curve.parquet")
    if set(curve["source_dataset"]) != set(SOURCES):
        raise EvidenceError(f"Missing source curve: {method}")
    return {
        "manifest": manifest,
        "metrics": metrics,
        "frame": frame,
        "curve": curve,
        "manifest_sha256": digest(folder / "manifest.json"),
        "folder": folder,
    }


def locked_budget_rows(
    run: dict[str, Any], method: str, lock_method: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    metrics, frame = run["metrics"], run["frame"]
    penalties = lock_method["locked_budget_penalties"]
    if penalties != metrics["validation_budget_penalties"] or set(penalties) != set(
        SOURCES
    ):
        raise EvidenceError(f"Locked policy differs from measured policy: {method}")
    for source in SOURCES:
        rows = frame.loc[frame["source_dataset"].eq(source)]
        scores = rows[[f"score_{tier}" for tier in TIERS]].to_numpy(float)
        quality = rows[[f"quality_{tier}" for tier in TIERS]].to_numpy(float)
        costs = rows[[f"actual_cost_{tier}" for tier in TIERS]].to_numpy(float)
        decision_costs = rows[[f"decision_cost_{tier}" for tier in TIERS]].to_numpy(
            float
        )
        reference = metrics["summary"]["per_source"][source]
        if reference["row_count"] != len(rows):
            raise EvidenceError(f"Source support mismatch: {method}/{source}")
        source_curve = run["curve"].loc[run["curve"]["source_dataset"].eq(source)]
        recreated = evaluate_penalty_curve(
            rows, scores, method=method, penalties=source_curve["penalty"].tolist()
        )
        curve_fields = [
            "penalty",
            "mean_quality",
            "mean_actual_cost",
            "relative_cost",
            "low_rate",
            "medium_rate",
            "high_rate",
        ]
        measured_curve = source_curve.sort_values("penalty")[curve_fields].to_numpy(
            float
        )
        recreated_curve = recreated.sort_values("penalty")[curve_fields].to_numpy(float)
        if measured_curve.shape != recreated_curve.shape or not np.allclose(
            measured_curve, recreated_curve, rtol=1e-7, atol=1e-9, equal_nan=True
        ):
            raise EvidenceError(
                f"Curve differs from saved logits/outcomes: {method}/{source}"
            )
        budgets = sorted(float(value) for value in penalties[source])
        if len(budgets) < 2:
            raise EvidenceError(f"Insufficient nAUC budget evidence: {method}/{source}")
        recomputed, _ = summarize_curve(
            rows, source_curve, budgets=budgets, retention_targets=[]
        )
        equal_number(
            recomputed["quality_cost_nauc"],
            reference["quality_cost_nauc"],
            f"{method}/{source} nAUC",
        )
        span = float(costs[:, 2].mean() - costs[:, 0].mean())
        for key, penalty in sorted(
            penalties[source].items(), key=lambda pair: float(pair[0])
        ):
            actions = route_actions(
                method,
                scores,
                decision_costs,
                number(penalty, "penalty", nonnegative=True),
            )
            selected_quality = quality[np.arange(len(rows)), actions]
            selected_cost = costs[np.arange(len(rows)), actions]
            actual_quality, actual_cost = (
                float(selected_quality.mean()),
                float(selected_cost.mean()),
            )
            equal_number(
                actual_quality,
                reference["quality_at_relative_cost_budget"][key],
                f"{method}/{source} locked quality",
            )
            relative_cost = (
                float((actual_cost - costs[:, 0].mean()) / span) if span > 0 else None
            )
            records.append(
                {
                    "method": method,
                    "source": source,
                    "rows": len(rows),
                    "budget": float(key),
                    "penalty": float(penalty),
                    "mean_quality": actual_quality,
                    "mean_actual_cost": actual_cost,
                    "relative_cost": relative_cost,
                    "budget_exceeded": None
                    if relative_cost is None
                    else relative_cost > float(key) + 1e-12,
                }
            )
    return records


def verify_comparison(
    runs: dict[str, dict[str, Any]], lock: dict[str, Any]
) -> list[dict[str, Any]]:
    if lock.get("schema_version") != "routing-objective-validation-lock-v1":
        raise EvidenceError("Unsupported validation lock schema")
    if lock.get("selection_metric") != "source_macro_quality_cost_nauc":
        raise EvidenceError("Integrated A must select source-macro nAUC")
    if lock.get("stage") != "final" or lock.get("seed_scope") != "canonical_only":
        raise EvidenceError("This report requires the final canonical-only A lock")
    if set(lock.get("required_methods", [])) != set(METHODS) or set(
        lock.get("methods", {})
    ) != set(METHODS):
        raise EvidenceError("The lock must contain exactly R0/R1/R2")
    if lock.get("selection_data") != "validation_only" or any(
        lock.get(field) is not False
        for field in ("test_data_accessed", "routerarena_accessed")
    ):
        raise EvidenceError("The lock is not sealed-test-free D_dev selection")
    reference = runs["ordinal"]
    reference_frame = reference["frame"].set_index("sample_id").sort_index()
    budget_rows = []
    for method, run in runs.items():
        manifest, metrics = run["manifest"], run["metrics"]
        for field in ("train_rows", "validation_rows", "optimizer_steps"):
            if metrics.get(field) != reference["metrics"].get(field):
                raise EvidenceError(
                    f"Completed row/step budget differs: {field}/{method}"
                )
        for field in CONTRACT_FIELDS:
            if manifest.get(field) is None or manifest[field] != reference[
                "manifest"
            ].get(field):
                raise EvidenceError(
                    f"Common comparison contract differs: {field}/{method}"
                )
        if manifest["seed"] != lock["canonical_seed"] or lock["methods"][method][
            "seeds"
        ] != [manifest["seed"]]:
            raise EvidenceError(f"Seed scope differs from lock: {method}")
        for field in ("experiment_version", "experiment_config_sha256"):
            if manifest[field] != lock.get(field):
                raise EvidenceError(f"Lock belongs to another experiment: {field}")
        comparable = run["frame"].set_index("sample_id").sort_index()
        columns = [name for name in reference_frame if not name.startswith("score_")]
        if not reference_frame[columns].equals(comparable[columns]):
            raise EvidenceError(f"Validation questions/outcomes differ: {method}")
        values = [
            number(
                metrics["summary"]["per_source"][source]["quality_cost_nauc"],
                "source nAUC",
            )
            for source in SOURCES
        ]
        macro = number(
            metrics["summary"]["source_macro_quality_cost_nauc"], "macro nAUC"
        )
        equal_number(macro, float(np.mean(values)), f"{method} source macro")
        for key in ("canonical_quality_cost_nauc", "quality_cost_nauc_mean"):
            equal_number(macro, lock["methods"][method][key], f"{method} lock {key}")
        budget_rows.extend(locked_budget_rows(run, method, lock["methods"][method]))
        default_rows = [
            row
            for row in budget_rows
            if row["method"] == method
            and math.isclose(row["budget"], lock["default_relative_cost_budget"])
        ]
        if len(default_rows) != len(SOURCES):
            raise EvidenceError(
                f"Default budget is not locked for both sources: {method}"
            )
        equal_number(
            metrics["validation_mean_actual_cost"],
            sum(row["mean_actual_cost"] * row["rows"] for row in default_rows)
            / sum(row["rows"] for row in default_rows),
            f"{method} default-budget mean cost",
        )

    def rank_key(method: str) -> tuple[Any, ...]:
        metrics = runs[method]["metrics"]
        latency = optional_number(metrics.get("batch_1_latency_ms", {}).get("p95"))
        return (
            -metrics["summary"]["source_macro_quality_cost_nauc"],
            metrics["validation_mean_actual_cost"],
            math.inf if latency is None else latency,
            method,
        )

    expected_winner = min(METHODS, key=rank_key)
    if lock.get("winner_method") != expected_winner:
        raise EvidenceError(
            f"Winner differs from measured ranking/tie-break: {expected_winner}"
        )
    return budget_rows


def fmt(value: Any, digits: int = 6) -> str:
    return "미측정 / N/A" if value is None else f"{float(value):,.{digits}f}"


def make_plots(runs: dict[str, dict[str, Any]], out: Path, synthetic: bool) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llm-router-report-mpl")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "figure.dpi": 120,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, source in zip(axes, SOURCES, strict=True):
        for method, marker in zip(METHODS, ("o", "s", "D"), strict=True):
            all_curves = runs[method]["curve"]
            curve = all_curves.loc[all_curves["source_dataset"].eq(source)].sort_values(
                "relative_cost"
            )
            ax.plot(
                curve["relative_cost"],
                curve["mean_quality"],
                marker=marker,
                markersize=4,
                linewidth=1.5,
                color=COLORS[method],
                label=LABELS[method],
            )
        support = runs["ordinal"]["metrics"]["summary"]["per_source"][source][
            "row_count"
        ]
        ax.set(
            title=f"{source} | D_dev n={support:,}",
            xlabel="Source-local relative actual cost",
            ylabel="Mean recorded outcome quality",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle(
        f"D_dev quality-cost policies | seed {runs['ordinal']['manifest']['seed']}"
        if not synthetic
        else "SYNTHETIC FIXTURE — not an experiment result",
        fontsize=13,
    )
    fig.savefig(out / "quality-cost-curves.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    groups = (*SOURCES, "source macro")
    x = np.arange(len(groups))
    for index, method in enumerate(METHODS):
        summary = runs[method]["metrics"]["summary"]
        values = [
            summary["per_source"][source]["quality_cost_nauc"] for source in SOURCES
        ] + [summary["source_macro_quality_cost_nauc"]]
        bars = ax.bar(
            x + (index - 1) * 0.24,
            values,
            width=0.23,
            label=LABELS[method],
            color=COLORS[method],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=groups,
        ylabel="Normalized quality-cost area (nAUC)",
        ylim=(0, 1.13),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    ax.set_title(
        "D_dev nAUC | one seed; uncertainty not measured"
        if not synthetic
        else "SYNTHETIC FIXTURE — not an experiment result"
    )
    fig.savefig(out / "nauc-comparison.png")
    plt.close(fig)


def render_report(
    runs: dict[str, dict[str, Any]],
    lock: dict[str, Any],
    budgets: list[dict[str, Any]],
    *,
    lock_sha: str,
    run_record: dict[str, Any] | None,
    synthetic: bool,
) -> str:
    winner = lock["winner_method"]
    title = (
        "SYNTHETIC FIXTURE — 실험 결과 아님"
        if synthetic
        else "R0/R1/R2 통합 A 비교 보고서"
    )
    lines = [
        f"# {title}",
        "",
        f"D_dev source-macro 비용–품질 nAUC의 validation lock 선정 방법은 **{LABELS[winner]}**다. 이 보고서는 로컬의 봉인된 검증 파일과 lock을 대조한 결과이며 운영 승인·활성화 증거가 아니다.",
        "",
        f"- 범위: `{lock['experiment_version']}`, seed `{lock['canonical_seed']}`, canonical 세 방법. B·RouterDC·추가 seed 반복은 포함하지 않는다.",
        f"- validation lock SHA-256: `{lock_sha}`",
        "- 파일 checksum, 공통 데이터·학습 계약, 동일 validation 행·outcome, source-macro 계산, locked policy 및 승자·동률 규칙 일치 확인.",
        "- model weight는 이 renderer에서 다시 로드하거나 검증하지 않는다. 모델 크기는 봉인 manifest의 기록값이다.",
        "",
        "## 1. 비교와 선정",
        "",
        "| 방법 | seed | train / D_dev 행 | 실제 step | RB nAUC | xRB nAUC | source macro nAUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, run in runs.items():
        metric = run["metrics"]
        summary = metric["summary"]
        lines.append(
            f"| {LABELS[method]} | {metric['seed']} | {metric['train_rows']:,} / {metric['validation_rows']:,} | {metric['optimizer_steps']:,} | {fmt(summary['per_source']['routerbench']['quality_cost_nauc'])} | {fmt(summary['per_source']['xroutebench']['quality_cost_nauc'])} | **{fmt(summary['source_macro_quality_cost_nauc'])}** |"
        )
    manifest = runs["ordinal"]["manifest"]
    training = manifest["common_training_config"]
    lines += [
        "",
        "| 공통 실행 조건 | 봉인된 값 |",
        "|---|---|",
        f"| backbone / revision | `{training.get('model_id', '미기록')}` / `{training.get('model_revision', '미기록')}` |",
        f"| max length / epochs | {training.get('max_length', '미기록')} / {training.get('epochs', '미기록')} |",
        f"| batch / accumulation / learning rate | {training.get('train_batch_size', '미기록')} / {training.get('gradient_accumulation_steps', '미기록')} / {training.get('learning_rate', '미기록')} |",
        f"| sampler / precision | `{training.get('train_sampling_strategy', '미기록')}` / `{training.get('precision', '미기록')}` |",
        f"| runtime | `{manifest['runtime_version']}` |",
        f"| outcome dataset | `{manifest['outcome_dataset_version']}` |",
        f"| outcome manifest SHA-256 | `{manifest['outcome_manifest_sha256']}` |",
        f"| validation data SHA-256 | `{manifest['validation_data_sha256']}` |",
    ]
    lines += [
        "",
        "![source별 quality-cost 곡선](quality-cost-curves.png)",
        "",
        "![세 방법 nAUC](nauc-comparison.png)",
        "",
        "각 source의 실제 후보 pool에서 always-low/high를 기준으로 비용·품질을 정규화하고 고정 budget grid의 clipped 회복 품질을 적분한 nAUC다. 두 source의 nAUC를 같은 비중으로 평균하며 전체 행 가중 평균과 구분한다. 곡선은 D_dev에서 관측한 정책 점을 연결한 것이며 test frontier가 아니다.",
        "",
        "| source | D_dev 지원 행 | 실제 pool 해석 |",
        "|---|---:|---|",
    ]
    for source in SOURCES:
        lines.append(
            f"| {source} | {runs['ordinal']['metrics']['summary']['per_source'][source]['row_count']:,} | 해당 source의 고정 low/medium/high 후보; 다른 source·현재 Claude와 동일 모델로 취급하지 않음 |"
        )
    lines += [
        "",
        "## 2. 동결한 budget별 품질·비용",
        "",
        "| 방법 | source | budget | λ | 관측 품질 | 평균 기록 비용 | 상대 실제 비용 | budget 초과 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in budgets:
        lines.append(
            f"| {LABELS[row['method']]} | {row['source']} | {fmt(row['budget'], 2)} | {fmt(row['penalty'])} | {fmt(row['mean_quality'])} | {fmt(row['mean_actual_cost'], 8)} | {fmt(row['relative_cost'])} | {'N/A' if row['budget_exceeded'] is None else '예' if row['budget_exceeded'] else '아니오'} |"
        )
    lines += [
        "",
        "표의 품질·비용은 lock의 source별 λ를 저장된 질문별 logits·outcome에 다시 적용한 값이다. 단위는 원천 actual_cost 단위이며 이 보고서는 달러·현재 Claude 절감률로 변환하지 않는다. budget에 가능한 정책이 없으면 가장 낮은 비용의 정책도 목표를 초과할 수 있으며 그 사실을 그대로 표시한다.",
        "",
        "## 3. 시간·크기와 출력 의미",
        "",
        "| 방법 | 학습 device | 기록된 train 시간(s) | batch-1 forward p95(ms) | latency 표본 수 | 모델 artifact(MiB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, run in runs.items():
        metric, manifest = run["metrics"], run["manifest"]
        latency = metric.get("batch_1_latency_ms", {})
        size = optional_number(manifest.get("model_artifact_size_bytes"))
        device = (
            metric.get("training_metrics", {}).get("model", {}).get("device", "미기록")
        )
        lines.append(
            f"| {LABELS[method]} | `{device}` | {fmt(optional_number(metric.get('fit_seconds')), 2)} | {fmt(optional_number(latency.get('p95')), 3)} | {fmt(optional_number(latency.get('sample_count')), 0)} | {fmt(None if size is None else size / 1024**2, 2)} |"
        )
    lines += [
        "",
        "학습 시간은 trainer가 기록한 wall time이다. 체크포인트에 저장된 학습 시간은 이어받을 수 있지만 유실·재실행으로 소모한 계산, 실패·복구·전송 및 전체 GPU 과금 시간을 모두 합산한 값은 아니다. p95는 warmup 뒤 batch-1 model forward 측정이며 tokenizer·서버·네트워크 지연을 포함하지 않는다. 미기록 시간·사용량은 추정해 채우지 않는다.",
        "",
        "| 방법 | 학습 목표·출력 의미 |",
        "|---|---|",
        "| R0 CE | T1 easy/medium/hard 분류; score는 class softmax. 난이도 정확도와 LLM 성공률은 다름 |",
        "| R1 success BCE | 실제 후보별 success를 독립 sigmoid로 예측. 세 값의 합이 1일 필요 없음 |",
        "| R2 pairwise | 관측 quality 차이에 따른 후보 상대 score. class probability/confidence가 아님 |",
        "",
        "## 4. 관측·해석·한계",
        "",
    ]
    winner_value = runs[winner]["metrics"]["summary"]["source_macro_quality_cost_nauc"]
    for method in METHODS:
        if method != winner:
            delta = (
                winner_value
                - runs[method]["metrics"]["summary"]["source_macro_quality_cost_nauc"]
            )
            lines.append(
                f"- 관측: {LABELS[winner]}의 source-macro nAUC와 {LABELS[method]}의 차이는 `{delta:+.6f}`다."
            )
            source_deltas = [
                f"{source} `{runs[winner]['metrics']['summary']['per_source'][source]['quality_cost_nauc'] - runs[method]['metrics']['summary']['per_source'][source]['quality_cost_nauc']:+.6f}`"
                for source in SOURCES
            ]
            lines.append(
                f"  같은 비교의 source별 차이: {', '.join(source_deltas)}. source macro 개선이 모든 source의 개선을 뜻하지는 않는다."
            )
    lines += [
        "- 선정은 nAUC 내림차순, 정확한 동률이면 default-budget validation 평균 비용·batch-1 p95·고정 method ID 순서와 일치한다. 단일 seed의 측정상 순위이며 통계적 우월성 또는 미래 서비스 성능 보장이 아니다.",
        "- 결과만으로 label 압축·학습 신호·분포가 차이를 일으켰다고 인과적으로 확정하지 않는다. 원인 설명은 별도 통제 실험이 필요한 가설이다.",
        "- seed 반복·paired group bootstrap CI는 이 입력에서 제공되지 않았으므로 미측정이다. 단일 seed의 표준편차 0을 안정성 근거로 쓰지 않는다.",
        "- sealed test·D_gate·D_stream·D_retention_test·RouterArena 결과는 이 보고서 입력에 없으며 미측정이다. 0점이나 통과로 표시하지 않는다.",
        "- 공개 outcome replay와 현재 Claude 서비스의 실제 응답 품질·사용량·가격을 구분한다. R1/R2 채택은 score·mapping·관측 가능한 feedback target/loss·gate 인수가 필요하다.",
        "",
        "## 5. 보관·게시·활성화",
        "",
        "| 상태 | 이 보고서의 근거 |",
        "|---|---|",
    ]
    statuses = (
        ("model_package_published", "GCS 모델 package 게시"),
        ("current_pointer_updated", "CURRENT 갱신"),
        ("serving_activated", "서버 활성화"),
        ("personal_drive_copy_verified", "개인 Drive 보존 검증"),
    )
    for key, label in statuses:
        value = (run_record or {}).get("status", {}).get(key)
        state = (
            "미확인: 해당 원격 증거 입력 없음"
            if value is None
            else "외부 run 기록은 완료로 표시; 이 명령은 원격 상태를 독립 확인하지 않음"
            if value is True
            else "외부 run 기록은 미완료로 표시"
        )
        lines.append(f"| {label} | {state} |")
    lines += [
        "",
        "후보 학습 파일의 GCS 백업, 운영 package 게시, CURRENT 변경, 실제 serving 활성화는 서로 다른 단계다. validation lock의 승자가 정해져도 이 명령은 승인·게시·활성화하지 않는다.",
        "",
        "## 6. 재현 근거",
        "",
        "| 방법 | run manifest SHA-256 |",
        "|---|---|",
    ]
    lines.extend(
        f"| {LABELS[method]} | `{run['manifest_sha256']}` |"
        for method, run in runs.items()
    )
    lines += [
        "",
        "입력 파일별 hash와 생성기 hash, machine-readable budget 집계는 `report.json`에 보존한다. 실제 model weight의 원격 보존과 추론 호환성은 별도 인수 절차로 확인한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing ordinal/pointwise/pairwise",
    )
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-record",
        type=Path,
        help="Optional existing standard run.json; remote statuses are attributed, not verified",
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Prominently mark synthetic formatting tests",
    )
    args = parser.parse_args()
    try:
        root, out = args.runs_dir.resolve(), args.output_dir.resolve()
        if any(
            out == root / method or (root / method) in out.parents for method in METHODS
        ):
            raise EvidenceError("Write the report outside each immutable run directory")
        if out.exists() and any(out.iterdir()):
            raise EvidenceError(
                "Output directory must be absent or empty; existing reports are immutable"
            )
        runs = {method: load_method(root, method) for method in METHODS}
        if (
            any(
                "SYNTHETIC" in run["manifest"]["experiment_version"].upper()
                for run in runs.values()
            )
            and not args.synthetic_fixture
        ):
            raise EvidenceError("Synthetic evidence must be marked --synthetic-fixture")
        lock = read_json(args.validation_lock)
        budget_rows = verify_comparison(runs, lock)
        run_record = read_json(args.run_record) if args.run_record else None
        lock_sha = digest(args.validation_lock)
        out.mkdir(parents=True, exist_ok=True)
        make_plots(runs, out, args.synthetic_fixture)
        report = render_report(
            runs,
            lock,
            budget_rows,
            lock_sha=lock_sha,
            run_record=run_record,
            synthetic=args.synthetic_fixture,
        )
        (out / "REPORT.md").write_text(report, encoding="utf-8")
        record = {
            "schema_version": "routing-objective-comparison-report-v1",
            "synthetic_fixture": args.synthetic_fixture,
            "comparison_scope": "A_integrated_source_local_symbolic_tiers",
            "selection_status": "validation_lock_verified",
            "selected_method": lock["winner_method"],
            "seed": lock["canonical_seed"],
            "validation_lock_sha256": lock_sha,
            "common_training_config": runs["ordinal"]["manifest"][
                "common_training_config"
            ],
            "data_contract": {
                field: runs["ordinal"]["manifest"][field]
                for field in CONTRACT_FIELDS
                if field != "common_training_config"
            },
            "generator_sha256": digest(Path(__file__)),
            "evaluator_source_sha256": digest(
                Path(__file__).resolve().parents[1]
                / "src/llm_router_labeling/objective_comparison/evaluation.py"
            ),
            "operation": "local_report_only",
            "production_activation_performed": False,
            "model_weights_verified_by_renderer": False,
            "remote_status_independently_verified": False,
            "reported_remote_status": (run_record or {}).get("status", {}),
            "source_run_record_sha256": digest(args.run_record)
            if args.run_record
            else None,
            "locked_budget_results": budget_rows,
            "methods": {
                method: {
                    "manifest_sha256": run["manifest_sha256"],
                    "summary": run["metrics"]["summary"],
                    "input_artifacts": run["manifest"]["artifacts"],
                }
                for method, run in runs.items()
            },
            "limitations": [
                "single_training_seed",
                "no_group_bootstrap_ci",
                "no_sealed_or_external_evaluation_inputs",
                "no_online_Claude_outcomes",
                "model_size_from_manifest",
                "training_time_excludes_unrecorded_attempts",
            ],
        }
        (out / "report.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "report": str(out / "REPORT.md"),
                    "winner": lock["winner_method"],
                    "synthetic_fixture": args.synthetic_fixture,
                    "cloud_actions": False,
                },
                ensure_ascii=False,
            )
        )
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        parser.exit(2, f"Report evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
