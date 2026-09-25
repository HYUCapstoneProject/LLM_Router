"""Project sealed outcomes using already selected train-only source-local arms.

This prepares targets only. It performs no model inference, candidate selection,
policy fitting, metric computation, or inspection of model predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..objective_comparison.data import (
    ObjectiveDataError,
    _load_anchor_rows,
    _project_source,
    _routerbench_matrices,
    _xroute_matrices,
)
from ..shared.dataset_io import load_config, sha256_file, verify_file, write_json

BASE_MANIFEST_SHA256 = (
    "f6a4cff1684bc2626a8901dbfda9f81e7d5d7e8753e5f48320cb1ada44c8cca4"
)
OUTCOME_MANIFEST_SHA256 = (
    "28cbc342c510fee7589151f99cdfe440281bfbe6288abd749eac9fd52f4c0b19"
)
CONFIG_SHA256 = "45e31296a82cded0fe82a8eaa003326e40e851f40745c793b2eb167f5c226d69"
PARTITIONS = {
    "train": "train/training.parquet",
    "validation": "validation/validation.parquet",
    "gate": "sealed/gate.parquet",
    "retention_test": "sealed/retention_test.parquet",
    "test_routerbench": "test/routerbench.parquet",
    "test_xroutebench": "test/xroutebench.parquet",
    "stream": "sealed/stream.parquet",
}
OUTPUT_ROLES = {
    "gate": "D_gate",
    "retention_test": "D_retention_test",
    "test_routerbench": "routerbench_fixed_test",
    "test_xroutebench": "xroutebench_official_test",
    "stream": "D_stream_feedback_supply",
}
SELECTED_MODELS = {
    "routerbench": {
        "low": "mistralai/mistral-7b-chat",
        "medium": "zero-one-ai/Yi-34B-Chat",
        "high": "gpt-4-1106-preview",
    },
    "xroutebench": {
        "low": "gpt-oss-20b",
        "medium": "gpt-oss-120b",
        "high": "cogito-v2-1-671b",
    },
}


def _read_pinned_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ObjectiveDataError(f"Frozen JSON checksum mismatch: {path.name}")
    return load_config(path)


def _verify_artifacts(directory: Path, records: dict[str, Any]) -> None:
    for relative, record in sorted(records.items()):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ObjectiveDataError(f"Invalid frozen artifact path: {relative}")
        verify_file(path, record)


def _verify_sources(
    config: dict[str, Any], base_manifest: dict[str, Any], project_root: Path
) -> None:
    required = {
        config["routerbench"]["labeled_path"],
        config["routerbench"]["model_tiers_path"],
        *(
            str(Path(config["xroutebench"]["data_dir"]) / config["xroutebench"][key])
            for key in ("train_file", "test_file", "candidates_file")
        ),
    }
    records = base_manifest["source_artifacts"]
    if not required.issubset(records):
        raise ObjectiveDataError(
            "Outcome source paths are not all pinned in the base manifest"
        )
    # Verify every original source, including records not needed by this projection.
    _verify_artifacts(project_root, records)


def _frozen_statistics(statistics: pd.DataFrame, arms: dict[str, Any]) -> None:
    if statistics.duplicated(["source_dataset", "model_name"]).any():
        raise ObjectiveDataError("Frozen candidate statistics contain duplicate models")
    for source, expected in SELECTED_MODELS.items():
        record = arms["sources"][source]
        if record["selected_models"] != expected:
            raise ObjectiveDataError(
                f"Selected {source} arms differ from the frozen comparison"
            )
        selected_stats = statistics.loc[
            statistics["source_dataset"].eq(source)
        ].set_index("model_name")
        costs = []
        for tier, model in expected.items():
            if model not in selected_stats.index:
                raise ObjectiveDataError(
                    f"Frozen candidate statistics missing {source}/{tier}"
                )
            stat = selected_stats.loc[model]
            if stat["selected_tier"] != tier or int(stat["valid_rows"]) != int(
                record["training_row_count"]
            ):
                raise ObjectiveDataError(
                    f"Frozen train-only arm statistics disagree: {source}/{tier}"
                )
            costs.append(float(stat["mean_cost"]))
        if not np.isfinite(costs).all() or not 0 <= costs[0] < costs[1] < costs[2]:
            raise ObjectiveDataError(
                f"Frozen decision costs are not increasing: {source}"
            )


def _coverage(
    anchor: pd.DataFrame,
    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "complete",
        "rows": len(anchor),
        "sources": {},
        "invalid_target_count": 0,
    }
    for source, (quality, success, cost) in matrices.items():
        source_rows = anchor.loc[anchor["source_dataset"].eq(source)]
        wanted = source_rows["sample_id"].tolist()
        source_report: dict[str, Any] = {"row_count": len(wanted), "fields": {}}
        for tier, model in SELECTED_MODELS[source].items():
            for prefix, matrix in (
                ("quality", quality),
                ("success", success),
                ("actual_cost", cost),
            ):
                if matrix.index.has_duplicates:
                    raise ObjectiveDataError(
                        f"Duplicate outcome indices: {source}/{prefix}"
                    )
                values = pd.to_numeric(
                    matrix.reindex(index=wanted, columns=[model])[model],
                    errors="coerce",
                ).to_numpy(dtype=float, na_value=np.nan)
                invalid = ~np.isfinite(values) | (values < 0)
                if prefix == "quality":
                    invalid |= values > 1
                elif prefix == "success":
                    invalid |= ~np.isin(values, [0.0, 1.0])
                count = int(invalid.sum())
                source_report["fields"][f"{prefix}_{tier}"] = {
                    "requested_rows": len(wanted),
                    "valid_rows": len(wanted) - count,
                    "invalid_rows": count,
                    "invalid_sample_ids": source_rows.loc[invalid, "sample_id"]
                    .head(10)
                    .tolist(),
                }
                report["invalid_target_count"] += count
        report["sources"][source] = source_report
    if report["invalid_target_count"]:
        report["status"] = "blocked_incomplete_outcomes"
    return report


def build_frozen_evaluation_outcomes(
    *,
    project_root: Path,
    config_path: Path,
    outcome_dir: Path,
    output_dir: Path,
    expected_base_manifest_sha256: str = BASE_MANIFEST_SHA256,
    expected_outcome_manifest_sha256: str = OUTCOME_MANIFEST_SHA256,
    expected_config_sha256: str = CONFIG_SHA256,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    outcome_dir = outcome_dir.resolve()
    output_dir = output_dir.resolve()
    config = _read_pinned_json(config_path, expected_config_sha256)
    dataset_dir = (project_root / config["base_dataset_dir"]).resolve()
    if (
        output_dir.exists()
        or output_dir.is_relative_to(dataset_dir)
        or output_dir.is_relative_to(outcome_dir)
    ):
        raise ObjectiveDataError("Output must be new and outside frozen datasets")
    if config.get("input_contract") != "structure-preserving-v2":
        raise ObjectiveDataError(
            "Outcome projection requires structure-preserving-v2 inputs"
        )
    if config["base_dataset_manifest_sha256"] != expected_base_manifest_sha256:
        raise ObjectiveDataError("Config and frozen base manifest pin disagree")
    base_manifest = _read_pinned_json(
        dataset_dir / "manifest.json", expected_base_manifest_sha256
    )
    outcome_manifest = _read_pinned_json(
        outcome_dir / "manifest.json", expected_outcome_manifest_sha256
    )
    if (
        outcome_manifest["base_dataset_manifest_sha256"]
        != expected_base_manifest_sha256
    ):
        raise ObjectiveDataError("Frozen arm and base dataset contracts disagree")
    if (
        outcome_dir / "_SUCCESS"
    ).read_text().strip() != expected_outcome_manifest_sha256:
        raise ObjectiveDataError("Frozen outcome dataset completion marker disagrees")
    _verify_artifacts(dataset_dir, base_manifest["artifacts"])
    _verify_artifacts(outcome_dir, outcome_manifest["artifacts"])
    _verify_sources(config, base_manifest, project_root)
    arms = load_config(outcome_dir / "arms.json")
    if (
        arms != outcome_manifest["arms"]
        or arms["selection_split"] != "train_only"
        or outcome_manifest["routerarena_used_for_selection"]
    ):
        raise ObjectiveDataError("Frozen arms must have train-only provenance")
    statistics = pd.read_parquet(outcome_dir / "candidate-statistics.parquet")
    _frozen_statistics(statistics, arms)
    # Include train/dev in the separation guard without including them in outputs.
    all_rows = _load_anchor_rows(
        dataset_dir,
        split_files=PARTITIONS,
        artifact_contracts=base_manifest["artifacts"],
    )
    if set(all_rows["source_dataset"]) != set(SELECTED_MODELS):
        raise ObjectiveDataError("Frozen rows contain unexpected dataset sources")
    for logical_split, rows in all_rows.groupby("objective_split", sort=False):
        original_split = {
            "test_routerbench": "test",
            "test_xroutebench": "external_test",
        }.get(logical_split, logical_split)
        if len(rows) != base_manifest["row_counts"][original_split] or set(
            rows["split"]
        ) != {original_split}:
            raise ObjectiveDataError(
                f"Frozen partition contract mismatch: {logical_split}"
            )
    anchor = all_rows.loc[all_rows["objective_split"].isin(OUTPUT_ROLES)].reset_index(
        drop=True
    )
    try:
        matrices = {
            "routerbench": _routerbench_matrices(config, project_root, anchor),
            "xroutebench": _xroute_matrices(config, project_root, anchor),
        }
        coverage = _coverage(anchor, matrices)
    except ObjectiveDataError as error:
        output_dir.mkdir(parents=True)
        write_json(
            output_dir / "coverage-report.json",
            {
                "status": "blocked_source_alignment",
                "rows": len(anchor),
                "error": str(error),
            },
        )
        raise
    output_dir.mkdir(parents=True)
    write_json(output_dir / "coverage-report.json", coverage)
    if coverage["status"] != "complete":
        raise ObjectiveDataError(
            f"Frozen selected-arm outcomes are incomplete; see {output_dir / 'coverage-report.json'}"
        )
    parts = [
        _project_source(
            anchor,
            source,
            *values,
            arms["sources"][source]["selected_models"],
            statistics.loc[statistics["source_dataset"].eq(source)].drop(
                columns="source_dataset"
            ),
        )
        for source, values in matrices.items()
    ]
    projected = (
        pd.concat(parts, ignore_index=True)
        .set_index("sample_id", drop=False)
        .loc[anchor["sample_id"]]
        .reset_index(drop=True)
    )
    projected = projected.drop(columns="oracle_quality_tier")
    if not projected.loc[:, anchor.columns].equals(anchor):
        raise ObjectiveDataError(
            "Projection changed frozen IDs, text, labels, or metadata"
        )
    artifact_paths = [output_dir / "coverage-report.json"]
    for split in OUTPUT_ROLES:
        path = output_dir / f"{split}.parquet"
        projected.loc[projected["objective_split"].eq(split)].to_parquet(
            path, index=False
        )
        artifact_paths.append(path)
    manifest = {
        "schema_version": "frozen-evaluation-outcomes-v1",
        "stage": "target_projection_only_no_model_evaluation",
        "base_dataset_manifest_sha256": expected_base_manifest_sha256,
        "train_outcome_manifest_sha256": expected_outcome_manifest_sha256,
        "objective_config_sha256": expected_config_sha256,
        "arms_sha256": outcome_manifest["artifacts"]["arms.json"]["sha256"],
        "candidate_statistics_sha256": outcome_manifest["artifacts"][
            "candidate-statistics.parquet"
        ]["sha256"],
        "arms": arms,
        "input_contract": config["input_contract"],
        "partition_roles": OUTPUT_ROLES,
        "row_counts": {
            str(split): int(count)
            for split, count in projected["objective_split"]
            .value_counts()
            .sort_index()
            .items()
        },
        "source_partition_row_counts": {
            f"{source}/{split}": int(count)
            for (source, split), count in projected.groupby(
                ["source_dataset", "objective_split"]
            )
            .size()
            .items()
        },
        "decision_cost_policy": "source-local normalized train-only frozen candidate mean cost",
        "coverage_status": coverage["status"],
        "dropped_row_count": 0,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/routing_objective_continual_v1.json"),
    )
    parser.add_argument(
        "--outcome-dir",
        type=Path,
        default=Path("ml/artifacts/routing-objective-continual-v1/dataset"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_frozen_evaluation_outcomes(
        project_root=args.project_root,
        config_path=args.config,
        outcome_dir=args.outcome_dir,
        output_dir=args.output_dir,
    )
    print(
        f"Projected {sum(result['row_counts'].values())} frozen target rows with no model evaluation: {result['row_counts']}"
    )


if __name__ == "__main__":
    main()
