"""Frozen R1 length ablation; consumes actual forward scores, never fits a policy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIERS = ("low", "medium", "high")
LABELS = ("easy", "medium", "hard")
SCORE_COLS = [f"score_{x}" for x in TIERS]

# Frozen before new inference: data SHA -> historical same-checkpoint 512 score SHA.
HISTORICAL_SCORE_SHA = {'A': {}, 'G1': {}}


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def aligned_scores(rows, path):
    frame = pd.read_parquet(path)
    if frame.sample_id.duplicated().any() or rows.sample_id.duplicated().any():
        raise ValueError("duplicate sample_id")
    if not set(frame.sample_id) == set(rows.sample_id):
        raise ValueError("scores differ from original evaluation population")
    frame = frame.set_index("sample_id").loc[rows.sample_id].reset_index()
    logits = frame[SCORE_COLS].to_numpy(float)
    if logits.shape != (len(rows), 3) or not np.isfinite(logits).all():
        raise ValueError("invalid raw logits")
    if "input_token_count" not in frame:
        raise ValueError("full untruncated token counts must be recorded")
    counts = frame.input_token_count.to_numpy(int)
    if (counts <= 0).any():
        raise ValueError("invalid token counts")
    return logits, counts


def probabilities(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def actions(rows, logits, policies, key="0.5000"):
    costs = rows[[f"decision_cost_{t}" for t in TIERS]].to_numpy(float)
    penalty = np.array([policies[str(s)][key] for s in rows.source_dataset])
    return (probabilities(logits) - costs * penalty[:, None]).argmax(1)


def macro_f1(y, action):
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, action), 1)
    den = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * np.diag(confusion), den, out=np.zeros(3), where=den != 0)
    return float(f1.mean()), confusion.tolist()


def cluster_bootstrap_delta(rows, delta, iterations=10000, seed=42, *, is_arena=False):
    # The bootstrap statistic must match the reported point estimate.
    # Arena pools full/perturbed rows only descriptively; original+variant share one cluster.
    rng = np.random.default_rng(seed)
    draws = []
    counts = {}
    if is_arena:
        if "source_sample_id" not in rows or rows.source_sample_id.isna().any():
            raise ValueError("Arena uncertainty requires original source_sample_id clusters")
        partitions = [("all_rows", rows.index)]
        unit, strata, estimand = "source_sample_id", "none", "row_weighted_accuracy_delta"
    else:
        partitions = list(rows.groupby("source_dataset", sort=True).groups.items())
        unit, strata, estimand = "duplicate_group_id", "source_dataset", "source_macro_quality_delta"
    for source, index in partitions:
        sub = rows.loc[index]
        column = "source_sample_id" if is_arena else "duplicate_group_id"
        if column not in sub or sub[column].isna().any():
            raise ValueError("Missing declared bootstrap cluster IDs")
        groups = sub[column].astype(str)
        aggregated = pd.DataFrame({"group": groups.to_numpy(), "value": delta[index]}).groupby("group").value.agg(["sum", "count"])
        sums, size = aggregated["sum"].to_numpy(), aggregated["count"].to_numpy()
        counts[str(source)] = len(sums)
        values = np.empty(iterations)
        for start in range(0, iterations, 100):
            ids = rng.integers(0, len(sums), size=(min(100, iterations-start), len(sums)))
            values[start:start+len(ids)] = sums[ids].sum(1) / size[ids].sum(1)
        draws.append(values)
    array = np.mean(draws, axis=0)
    interval = np.quantile(array, [0.025, 0.975]).tolist()
    ci_name = "paired_row_weighted_delta_95_ci" if is_arena else "paired_source_macro_delta_95_ci"
    return {"estimand": estimand, "paired_delta_95_ci": interval, ci_name: interval,
            "iterations": iterations, "seed": seed, "unit": unit, "strata": strata,
            "groups_per_source": counts,
            "warning": "sampling-only uncertainty; not a training-seed CI or new independent test",
            "arena_full_and_perturbed_rows_clustered_together": bool(is_arena)}


def measure(rows, a512, a8192, is_arena):
    rows = rows.reset_index(drop=True)
    y = rows.difficulty_label.map(dict(zip(LABELS, range(3)))).to_numpy()
    values = {}
    for length, act in [(512, a512), (8192, a8192)]:
        f1, confusion = macro_f1(y, act)
        item = {"difficulty_slot_accuracy": float((act == y).mean()), "difficulty_slot_macro_f1": f1,
                "severe_underroute_hard_to_low_count": int(((y == 2) & (act == 0)).sum()),
                "severe_underroute_hard_to_low_rate": float(((y == 2) & (act == 0)).mean()),
                "hard_label_rows": int((y == 2).sum()),
                "hard_recall": float((act[y == 2] == 2).mean()) if (y == 2).any() else None,
                "hard_to_low_conditional_rate": float((act[y == 2] == 0).mean()) if (y == 2).any() else None,
                "confusion_matrix": confusion, "actions": {t: int((act == i).sum()) for i, t in enumerate(TIERS)}}
        if not is_arena:
            q = rows[[f"quality_{t}" for t in TIERS]].to_numpy(float)
            cost = rows[[f"actual_cost_{t}" for t in TIERS]].to_numpy(float)
            success = rows[[f"success_{t}" for t in TIERS]].to_numpy(float)
            if not np.isfinite(q).all() or not np.isfinite(cost).all() or not np.isin(success, [0, 1]).all():
                raise ValueError("frozen quality/cost/observed success targets are invalid")
            chosen_q, chosen_c = q[np.arange(len(rows)), act], cost[np.arange(len(rows)), act]
            chosen_success = success[np.arange(len(rows)), act]
            by_source = {}
            for source, index in rows.groupby("source_dataset", sort=True).groups.items():
                selected_q, selected_c = chosen_q[index], chosen_c[index]
                span = float(cost[index, 2].mean() - cost[index, 0].mean())
                relative = float((selected_c.mean() - cost[index, 0].mean()) / span) if span > 0 else None
                by_source[str(source)] = {"rows": len(index), "mean_quality": float(selected_q.mean()),
                    "mean_actual_cost": float(selected_c.mean()), "relative_cost": relative,
                    "mean_observed_success_target": float(chosen_success[index].mean()),
                    "actual_cost_semantics": "recorded source cost units, not current service prices",
                    "default_budget_exceeded": relative > 0.5 + 1e-12 if relative is not None else None}
            item.update(per_source=by_source, source_macro_quality=float(np.mean([r["mean_quality"] for r in by_source.values()])),
                        source_macro_observed_success_target=float(np.mean([
                            r["mean_observed_success_target"] for r in by_source.values()])),
                        row_weighted_quality=float(chosen_q.mean()),
                        quality_semantics="historical per-source candidate quality; source contracts may differ",
                        success_semantics="frozen binary success targets of these candidate pools; not service-model success",
                        aggregate_cost_semantics="source-macro relative cost only; absolute costs are never pooled")
            relative = [r["relative_cost"] for r in by_source.values()]
            item["source_macro_relative_cost"] = float(np.mean(relative)) if all(r is not None for r in relative) else None
        values[str(length)] = item
    delta = ((a8192 == y).astype(float) - (a512 == y).astype(float)) if is_arena else (
        q[np.arange(len(rows)), a8192] - q[np.arange(len(rows)), a512])
    output = {"rows": len(rows), "action_changed_count": int((a512 != a8192).sum()),
              "action_changed_rate": float((a512 != a8192).mean()), "lengths": values,
              "uncertainty": cluster_bootstrap_delta(rows, delta, is_arena=is_arena)}
    output["primary_quality_delta"] = (values["8192"]["difficulty_slot_accuracy"] - values["512"]["difficulty_slot_accuracy"]
        if is_arena else values["8192"]["source_macro_quality"] - values["512"]["source_macro_quality"])
    if not is_arena:
        success_delta = success[np.arange(len(rows)), a8192] - success[np.arange(len(rows)), a512]
        output["observed_success_target_delta"] = (
            values["8192"]["source_macro_observed_success_target"]
            - values["512"]["source_macro_observed_success_target"])
        success_uncertainty = cluster_bootstrap_delta(rows, success_delta)
        success_uncertainty["estimand"] = "source_macro_observed_success_target_delta"
        output["observed_success_target_uncertainty"] = success_uncertainty
    output["small_or_noisy_slice"] = len(rows) < 30 or any(
        count < 30 for count in output["uncertainty"]["groups_per_source"].values())
    flags = []
    if output["primary_quality_delta"] < 0:
        flags.append("observed primary quality deterioration")
    if values["8192"]["severe_underroute_hard_to_low_rate"] > values["512"]["severe_underroute_hard_to_low_rate"]:
        flags.append("increased severe difficulty-slot under-routing")
    if not is_arena:
        if output["observed_success_target_delta"] < 0:
            flags.append("observed binary success target deterioration")
        for source, newer in values["8192"]["per_source"].items():
            older = values["512"]["per_source"][source]
            if newer["mean_observed_success_target"] < older["mean_observed_success_target"]:
                flags.append(f"source binary success target deterioration: {source}")
            if newer["mean_quality"] < older["mean_quality"]:
                flags.append(f"source quality deterioration: {source}")
            if (newer["default_budget_exceeded"] and older["relative_cost"] is not None
                    and newer["relative_cost"] > older["relative_cost"]):
                flags.append(f"worsened default budget violation: {source}")
    output["predeclared_automatic_quality_pass_blockers"] = flags
    return output


def analyze_dataset(name, dataset_path, score512, score8192, policies, baseline=None):
    rows = pd.read_parquet(dataset_path).reset_index(drop=True)
    raw512, counts = aligned_scores(rows, score512)
    raw8192, counts8192 = aligned_scores(rows, score8192)
    if not np.array_equal(counts, counts8192):
        raise ValueError("different token counts between arms")
    is_arena = "evaluation_split" in rows
    a512 = raw512.argmax(1) if is_arena else actions(rows, raw512, policies)
    a8192 = raw8192.argmax(1) if is_arena else actions(rows, raw8192, policies)
    parts = {"all": np.ones(len(rows), dtype=bool), "over512": counts > 512, "at_most512": counts <= 512,
             "over8192": counts > 8192}
    # Arena full and robustness are separate; combined all is explicitly descriptive.
    for column in ["source_dataset", "category", "evaluation_split", "arena_source_dataset"]:
        if column not in rows: continue
        for value in sorted(rows[column].astype(str).unique()):
            mask = rows[column].astype(str).eq(value).to_numpy()
            parts[f"{column}={value}"] = mask
            parts[f"{column}={value}/over512"] = mask & (counts > 512)
    reports = {key: measure(rows.loc[mask], a512[mask], a8192[mask], is_arena) for key, mask in parts.items() if mask.any()}
    output = {"dataset": name, "rows": len(rows), "long_rows": int((counts > 512).sum()),
        "beyond8192_rows": int((counts > 8192).sum()), "max_tokens": int(counts.max()),
        "original_data_sha256": sha(dataset_path), "scores512_sha256": sha(score512), "scores8192_sha256": sha(score8192),
        "arena_is_difficulty_slot_diagnostic_not_success_or_cost": is_arena,
        "arena_combined_all_rows_interpretation": "descriptive only; full and robustness reported separately, original+variant kept in source_sample_id clusters" if is_arena else None,
        "primary_policy": "lambda_zero_action_slot_argmax" if is_arena else "original_frozen_source_policy_budget_0.5",
        "short_max_abs_logit_delta": float(np.abs(raw8192[counts <= 512] - raw512[counts <= 512]).max()) if (counts <= 512).any() else None,
        "comparison_kind": "same_checkpoint_same_environment_512_vs_8192",
        "metric_semantics": {"difficulty_slot_accuracy": "agreement with historical easy/medium/hard labels",
                             "candidate_quality": "mean frozen quality of the actually selected candidate, absent on Arena",
                             "candidate_success": "mean frozen binary success target of the actually selected candidate, absent on Arena",
                             "candidate_cost": "per-source recorded costs and source-macro relative cost, absent on Arena"},
        "slices": reports}
    if baseline:
        old = pd.read_parquet(baseline).set_index("sample_id").loc[rows.sample_id].reset_index()
        previous = old[SCORE_COLS].to_numpy(float)
        old_actions = previous.argmax(1) if is_arena else actions(rows, previous, policies)
        reproduction = measure(rows, old_actions, a512, is_arena)
        labeled = reproduction.pop("lengths")
        reproduction["historical_512"] = labeled["512"]
        reproduction["recomputed_same_environment_512"] = labeled["8192"]
        reproduction["comparison_kind"] = "historical512_to_new512_reproduction_check_not_a_length_change"
        output["previous512_reproduction"] = {"path": str(baseline), "sha256": sha(baseline),
            "max_abs_logit_delta": float(np.abs(previous-raw512).max()),
            "action_change_count": int((a512 != old_actions).sum()),
            "reproduction_metrics": reproduction,
            "note": "This compares two 512 evaluations. Runtime/numerical/batching differences are separate from the same-environment 512-to-8192 length effect."}
    zero_a512, zero_a8192 = raw512.argmax(1), raw8192.argmax(1)
    if not is_arena:
        output["zero_penalty_secondary"] = {key: measure(rows.loc[mask], zero_a512[mask], zero_a8192[mask], False)
            for key, mask in parts.items() if mask.any() and key in ("all", "over512", "at_most512")}
    comparison = rows[[c for c in ["sample_id", "source_dataset", "category", "evaluation_split", "difficulty_label"] if c in rows]].copy()
    comparison["input_token_count"] = counts
    comparison["action512"] = np.array(TIERS)[a512]
    comparison["action8192"] = np.array(TIERS)[a8192]
    comparison["action_changed"] = a512 != a8192
    return output, comparison


def pairs(items):
    return dict(item.split("=", 1) for item in items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "G1"], required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=parquet")
    parser.add_argument("--scores512", action="append", required=True, help="name=parquet")
    parser.add_argument("--scores8192", action="append", required=True, help="name=parquet")
    parser.add_argument("--previous512", action="append", default=[], help="name=parquet")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads((HERE/"frozen-policy.json").read_text())
    model_key = "a_pointwise" if args.experiment == "A" else "g1_r1_natural"
    if args.model_sha256 != policy["models"][model_key]["model_sha256"]:
        raise ValueError("checkpoint identity differs from frozen experiment")
    policies = policy["a_policies" if args.experiment == "A" else "g1_natural_policies"]
    data, old, new, previous = map(pairs, [args.dataset, args.scores512, args.scores8192, args.previous512])
    if set(data) != set(old) or set(data) != set(new) or set(data) != set(previous):
        raise ValueError("Every dataset requires paired new scores and a pinned historical512 baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    inference_evidence = {}
    audit = json.loads((HERE / "dataset-audit.json").read_text())
    roles = {"arena", *(key for key in audit if key.startswith("a_" if args.experiment == "A" else "g1_"))}
    allowed_hashes = {audit[key]["sha256"] for key in roles}
    identities = json.loads((HERE / "model-identities.json").read_text())
    candidate_pools = identities[model_key]["candidate_pools"]["sources"]
    identity = identities[model_key]
    expected_model_files = identity.get("files") or {
        key.removeprefix("model/"): record["sha256"]
        for key, record in identity["expected_artifacts"].items() if key.startswith("model/")}
    identity_files = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "training_config.json"}
    for name, path in data.items():
        if sha(path) not in allowed_hashes:
            raise ValueError("dataset is not a frozen role of this model's experiment")
        if sha(previous[name]) != HISTORICAL_SCORE_SHA[args.experiment][sha(path)]:
            raise ValueError("historical512 scores do not match the frozen checkpoint/dataset reference")
        original = pd.read_parquet(path)
        if "evaluation_split" not in original:
            for source, source_rows in original.groupby("source_dataset"):
                selected = candidate_pools[str(source)]["selected_models"]
                for tier in TIERS:
                    if not source_rows[f"model_{tier}"].eq(selected[tier]).all():
                        raise ValueError("outcome candidate pool differs from checkpoint experiment")
        parent = Path(old[name]).parent
        if parent != Path(new[name]).parent:
            raise ValueError("paired scores must originate from one recorded inference run")
        manifest_path = parent / "manifest.json"
        marker = parent / "_SUCCESS"
        if not marker.is_file() or marker.read_text().strip() != sha(manifest_path):
            raise ValueError("inference manifest is incomplete or lacks its verified success marker")
        evidence = json.loads(manifest_path.read_text())
        if evidence["model_files"]["model.safetensors"] != args.model_sha256:
            raise ValueError("actual inference checkpoint does not match the model identity pin")
        for filename in identity_files:
            if evidence["model_files"].get(filename) != expected_model_files[filename]:
                raise ValueError(f"inference-affecting model file or training history changed: {filename}")
        metadata_differences = {
            filename: {"historical": expected, "actual": evidence["model_files"].get(filename)}
            for filename, expected in expected_model_files.items()
            if filename not in identity_files and evidence["model_files"].get(filename) != expected}
        if evidence.get("max_lengths") != [512, 8192] or evidence.get("dtype") != "float32":
            raise ValueError("inference length pair or frozen numerical precision changed")
        if evidence.get("attention") not in {"eager", "sdpa"}:
            raise ValueError("actual attention backend was not recorded")
        spec = evidence["datasets"][name]
        if spec.get("rows") != len(original):
            raise ValueError("inference manifest population differs from original dataset")
        for length, score_path in [(512, old[name]), (8192, new[name])]:
            scored = pd.read_parquet(score_path)
            if spec[str(length)].get("computed_rows") != len(original):
                raise ValueError("this full-forward runner must recompute every row; reuse is not evidenced")
            token_counts = scored.input_token_count
            if not np.isfinite(token_counts).all() or not token_counts.eq(token_counts.astype(int)).all():
                raise ValueError("token counts must be finite integers")
            if not scored.was_truncated.eq(token_counts > length).all():
                raise ValueError("score truncation flag disagrees with token counts")
            if (spec.get("long_rows") != int((token_counts > 512).sum())
                    or spec.get("short_rows") != int((token_counts <= 512).sum())
                    or spec.get("above8192") != int((token_counts > 8192).sum())):
                raise ValueError("manifest length populations differ from scored rows")
        if spec["sha256"] != sha(path):
            raise ValueError("actual inference dataset bytes differ from evaluation outcomes")
        if not spec.get("short_token_ids_equal"):
            raise ValueError("short input token identity was not verified")
        if spec["512"]["score_sha256"] != sha(old[name]) or spec["8192"]["score_sha256"] != sha(new[name]):
            raise ValueError("raw score bytes differ from the inference manifest")
        inference_evidence[str(manifest_path)] = {
            "sha256": sha(manifest_path), "metadata": evidence,
            "inference_and_history_files_verified": sorted(identity_files),
            "metadata_only_file_differences": metadata_differences,
            "identity_note": "Metrics/receipt-only changes are disclosed separately; weights, tokenizer, config and training history remain strict."}
        report, predictions = analyze_dataset(name, path, old[name], new[name], policies, previous.get(name))
        reports[name] = report
        predictions.to_parquet(args.output_dir/f"{name}-paired-actions.parquet", index=False)
    result = {"schema_version": "r1-fixed-policy-length-diagnostic-v1", "experiment": args.experiment,
              "model_sha256": args.model_sha256, "policies_retuned": False, "model_retrained": False,
              "production_approved": False, "inference_protocol_sha256": sha(HERE.parent/"PROTOCOL.md"),
              "analysis_script_sha256": sha(__file__), "inference_evidence": inference_evidence, "datasets": reports,
              "limitations": ["G1 task holdout has no >512 examples; unchanged results cannot qualify long-input quality.",
                              "Arena is historical difficulty-slot alignment, not service-pool success/cost.",
                              "Actual runtime qualification and prior service-pool promotion gates remain separate."]}
    (args.output_dir/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: {"rows": r["rows"], "long_rows": r["long_rows"], "all": r["slices"]["all"]} for name,r in reports.items()}, indent=2))


if __name__ == "__main__": main()
