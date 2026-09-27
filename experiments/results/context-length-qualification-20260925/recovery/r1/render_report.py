"""Render verified R1 analysis artifacts; never reads or fits inference policies."""
import argparse
import hashlib
import json
from pathlib import Path


def number(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.3f}" if percent else f"{value:.6f}"


def selected_slices(report):
    if report["arena_is_difficulty_slot_diagnostic_not_success_or_cost"]:
        for split in ("full", "robustness"):
            # Use actual frozen split names, without combining original/perturbed rows.
            matching = [key for key in report["slices"] if key == f"evaluation_split={split}"]
            for key in matching:
                yield key, report["slices"][key]
                long_key = key + "/over512"
                if long_key in report["slices"]:
                    yield long_key, report["slices"][long_key]
        if "over8192" in report["slices"]:
            yield "over8192", report["slices"]["over8192"]
    else:
        for key in ("all", "over512"):
            if key in report["slices"]:
                yield key, report["slices"][key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [(path, json.loads(path.read_text())) for path in args.analysis]
    lines = ["# R1 동일 checkpoint의 입력 길이 검증", "",
             "512 토큰으로 학습한 기존 가중치를 그대로 두고, 동일한 실행 환경에서 추론 최대 길이만 512 → 8,192로 바꾼 진단이다. 임계값·비용 정책을 다시 맞추거나 추가 학습하지 않았다.", "",
             "A R1 pointwise는 초기 비교 모델이고, G1 R1-natural은 이후 실제 채택을 추천한 별개의 모델이다. 서로 대체해서 해석하지 않는다.", "",
             "- quality: 기존 후보 모델 결과에 기록된 연속 품질 점수의 출처별 평균을 다시 평균한 값.",
             "- success: 실제 관측 후보의 기존 이진 성공 target 평균. quality와 서로 다른 지표다.",
             "- RouterArena 정확도: 역사적 easy/medium/hard 라벨과 라우팅 슬롯의 일치율이다. 실제 서비스 모델의 성공률·비용으로 해석할 수 없다.",
             "- 변화량은 8,192 − 512이며, 신뢰구간은 같은 샘플의 paired cluster bootstrap 95% 구간이다. 학습 seed 불확실성·다중 비교·새 독립 평가를 포함하지 않는다.", ""]
    panel_rows = [[], [], [], []]
    for path, analysis in reports:
        experiment = analysis["experiment"]
        label = "A R1 pointwise" if experiment == "A" else "G1 R1-natural"
        lines += [f"## {label}", "", f"모델 SHA256: `{analysis['model_sha256']}`", "",
                  f"검증 원본: [{path.name}]({path})", ""]
        for evidence in analysis["inference_evidence"].values():
            runtime = evidence["metadata"]
            lines += [f"실행환경: torch {runtime['torch']}, transformers {runtime['transformers']}, {runtime['device']}, {runtime['attention']}, {runtime['dtype']}.",
                      "가중치·config·tokenizer 관련 3파일·training_config의 6파일 SHA가 기존 모델과 모두 일치한다.",
                      f"추론 영향 파일과 분리한 metadata-only 차이: {', '.join(evidence['metadata_only_file_differences']) or '없음'}.", "",
                      "| 평가 데이터 | 전체 추론 시간 512→8192 (s) | CUDA peak allocated 512→8192 (GiB) |", "|---|---:|---:|"]
            for dataset_name, recorded in runtime["datasets"].items():
                before, after = recorded["512"], recorded["8192"]
                mem_before, mem_after = before.get("peak_allocated_bytes"), after.get("peak_allocated_bytes")
                memory = f"{mem_before / 2**30:.3f}→{mem_after / 2**30:.3f}" if mem_before is not None and mem_after is not None else "미기록"
                lines.append(f"| {dataset_name} | {before['seconds']:.2f}→{after['seconds']:.2f} | {memory} |")
            lines += ["", "시간은 전체 데이터셋의 batch 평가 실행 시간이며 API 요청 지연이 아니다. 장문은 batch=1로 실행했다. CUDA 메모리는 이 평가 환경 수치로, 서비스 CPU/RAM 측정이나 배포 적합성 판정을 대신하지 않는다.", ""]
        for name, report in analysis["datasets"].items():
            lines += [f"### {name}", "",
                      "| 데이터·범위 | N | 긴 입력 N | 정책 품질/정확도 512→8192 (%) | Δ (pp), 95% CI | 라우팅 변경 | hard→low 수 512→8192 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            arena = report["arena_is_difficulty_slot_diagnostic_not_success_or_cost"]
            for scope, result in selected_slices(report):
                before, after = result["lengths"]["512"], result["lengths"]["8192"]
                field = "difficulty_slot_accuracy" if arena else "source_macro_quality"
                ci = result["uncertainty"]["paired_delta_95_ci"]
                long_n = result["rows"] if scope.endswith("over512") or scope == "over8192" else report["long_rows"] if scope == "all" else report["slices"].get(scope + "/over512", {}).get("rows", 0)
                caption = f"{name}/{scope}"
                lines.append(f"| {caption} | {result['rows']} | {long_n} | {number(before[field], True)}→{number(after[field], True)} | {number(result['primary_quality_delta'], True)} [{number(ci[0], True)}, {number(ci[1], True)}] | {result['action_changed_count']} | {before['severe_underroute_hard_to_low_count']}→{after['severe_underroute_hard_to_low_count']} |")
                axis_label = f"{label}: {caption} (n={result['rows']})"
                target_panel = 3 if arena else 0
                panel_rows[target_panel].append((axis_label, experiment, result["primary_quality_delta"], ci))
                if not arena:
                    panel_rows[1].append((axis_label, experiment, result["observed_success_target_delta"], result["observed_success_target_uncertainty"]["paired_delta_95_ci"]))
                    old_cost, new_cost = before["source_macro_relative_cost"], after["source_macro_relative_cost"]
                    if old_cost is not None and new_cost is not None:
                        panel_rows[2].append((axis_label, experiment, new_cost - old_cost, None))
            lines += ["", f"**{name} 재현 확인:** 과거 512와 이번 512의 라우팅 변경 {report['previous512_reproduction']['action_change_count']}건, 최대 logit 차이 {report['previous512_reproduction']['max_abs_logit_delta']:.8g}. 이 차이는 길이 변경 효과와 분리했다.",
                      f"512 이하 입력의 양쪽 최대 logit 차이 {report['short_max_abs_logit_delta']:.8g}; 전체 {report['rows']}건 중 >512 {report['long_rows']}건, >8192 {report['beyond8192_rows']}건.", ""]
            if not arena:
                lines += ["| 출처 (전체) | quality 512→8192 | success 512→8192 | relative cost 512→8192 |", "|---|---:|---:|---:|"]
                all_metric = report["slices"]["all"]["lengths"]
                for source, old in all_metric["512"]["per_source"].items():
                    new = all_metric["8192"]["per_source"][source]
                    lines.append(f"| {source} | {number(old['mean_quality'])}→{number(new['mean_quality'])} | {number(old['mean_observed_success_target'])}→{number(new['mean_observed_success_target'])} | {number(old['relative_cost'])}→{number(new['relative_cost'])} |")
                lines += ["", "비용은 각 출처 내 high/low 기준에 대해 정규화했다. 서로 다른 출처의 절대 비용을 합쳐 달러 비용으로 계산하지 않았다.", ""]
            blockers = [(scope, result["predeclared_automatic_quality_pass_blockers"])
                        for scope, result in selected_slices(report) if result["predeclared_automatic_quality_pass_blockers"]]
            if blockers:
                lines += ["사전 선언한 자동 통과 보류 조건:", ""]
                lines += [f"- {scope}: {'; '.join(flags)}" for scope, flags in blockers]
                lines.append("")
            adverse_subgroups = [(scope, result) for scope, result in report["slices"].items()
                                 if scope.startswith(("category=", "arena_source_dataset="))
                                 and result["predeclared_automatic_quality_pass_blockers"]]
            if adverse_subgroups:
                lines += ["세부 범위에서 관측된 악화 신호(탐색적 다중 비교이며 작은 범위는 결론 보류):", "",
                          "| 세부 범위 | N | primary Δ (pp), 95% CI | 보류 조건 |", "|---|---:|---:|---|"]
                for scope, result in adverse_subgroups:
                    ci = result["uncertainty"]["paired_delta_95_ci"]
                    flags = "; ".join(result["predeclared_automatic_quality_pass_blockers"])
                    lines.append(f"| {scope} | {result['rows']} | {number(result['primary_quality_delta'], True)} [{number(ci[0], True)}, {number(ci[1], True)}] | {flags} |")
                lines.append("")
    lines += ["## 해석 한계", "", "- G1 task holdout에는 >512 입력이 0개다. 결과가 같아도 긴 입력 품질을 검증한 근거가 아니다.",
              "- 긴 후보 결과 평가의 표본이 작거나 CI가 넓으면 결론은 보류다. 개선된 point estimate만으로 일반화 성공을 선언하지 않는다.",
              "- RouterArena full과 robustness는 원본·변형 관계가 있다. 둘을 독립 평가 건수로 더하지 않는다.",
              "- 기존 v1/학습 설정과 과거 결과는 바꾸지 않았다. 이 보고서는 배포 승인 또는 실제 서비스 후보 pool 검증을 대체하지 않는다.",
              "- 실제 서버와 맞는 장치·attention backend·메모리·지연 검증은 별도로 필요하다.", ""]
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = ["Candidate quality delta (source macro)", "Observed binary success-target delta (source macro)",
              "Relative recorded cost delta (source macro; lower costs less)", "RouterArena historical difficulty-slot accuracy delta"]
    figure, axes = plt.subplots(4, 1, figsize=(14, max(12, sum(len(rows) for rows in panel_rows) * .21)), constrained_layout=True)
    for ax, title, rows in zip(axes, titles, panel_rows):
        for index, (label, experiment, value, ci) in enumerate(rows):
            color = "#0f766e" if experiment == "G1" else "#64748b"
            if ci is not None:
                ax.plot([100 * ci[0], 100 * ci[1]], [index, index], color=color, linewidth=1.6)
            ax.scatter([100 * value], [index], color=color, s=22, zorder=3)
        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("8,192 minus 512 (percentage points)", fontsize=8)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.6)
    figure.suptitle("Frozen R1 checkpoints: inference length only; no retraining or policy tuning", fontsize=12)
    figure.savefig(args.output_dir / "r1-length-deltas.png", dpi=180)
    figure.savefig(args.output_dir / "r1-length-deltas.svg")
    plt.close(figure)
    lines += [f"![R1 length deltas](r1-length-deltas.png)", ""]
    (args.output_dir / "R1_LENGTH_REPORT.md").write_text("\n".join(lines))
    receipt = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in reports}
    receipt["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output_dir / "R1_LENGTH_REPORT.md")


if __name__ == "__main__":
    main()
