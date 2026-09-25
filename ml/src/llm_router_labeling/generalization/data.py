"""Frozen generalization roles and loss-weighting views; local files only.

Splits depend on input identities and a predeclared task taxonomy, never on
quality, success, inherited difficulty labels or model predictions. Candidate
selection and decision costs use only the newly assigned train rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..objective_comparison.data import (
    TIERS,
    _project_source,
    _routerbench_matrices,
    _xroute_matrices,
    select_three_arms,
)
from ..shared.dataset_io import sha256_file, write_json
from ..shared.structured_text import clean_text

ROLES = ("train", "calibration", "selection", "task_holdout")
SOURCES = ("routerbench", "xroutebench")
SCHEMA_VERSION = "routing-generalization-dataset-v1"


class GeneralizationDataError(ValueError):
    """A source, split or frozen artifact violates the dataset contract."""


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GeneralizationDataError(f"Expected a JSON object: {path}")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _check_pin(path: Path, pin: Mapping[str, Any]) -> None:
    if not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise GeneralizationDataError(f"SHA-256 mismatch or missing file: {path}")
    if "bytes" in pin and path.stat().st_size != pin["bytes"]:
        raise GeneralizationDataError(f"Byte length mismatch: {path}")


def _load_anchor(config: Mapping[str, Any], project_root: Path) -> pd.DataFrame:
    base = config["base"]
    for key in ("manifest", "training"):
        _check_pin(_path(project_root, base[key]["path"]), base[key])
    if (
        _json(_path(project_root, base["manifest"]["path"])).get(
            "coverage_resolution_status"
        )
        != "complete"
    ):
        raise GeneralizationDataError("Parent dataset coverage must be complete")
    frame = pd.read_parquet(_path(project_root, base["training"]["path"]))
    required = {
        "sample_id",
        "source_dataset",
        "source_split",
        "source_task",
        "category",
        "model_input_text",
        "difficulty_label",
        "duplicate_group_id",
        "split",
        "dataset_version",
    }
    if not required.issubset(frame):
        raise GeneralizationDataError(
            f"Missing anchor columns: {sorted(required - set(frame))}"
        )
    if (
        frame.empty
        or frame["sample_id"].isna().any()
        or frame["sample_id"].duplicated().any()
    ):
        raise GeneralizationDataError("Anchor sample IDs must be nonempty and unique")
    if set(frame["source_dataset"]) != set(SOURCES):
        raise GeneralizationDataError(
            "Exactly RouterBench and xRouteBench sources are required"
        )
    if not set(frame["split"]).issubset({"train", "validation"}):
        raise GeneralizationDataError(
            "Only previous train/development rows are permitted"
        )
    if set(frame.loc[frame.source_dataset.eq("xroutebench"), "source_split"]) != {
        "train"
    }:
        raise GeneralizationDataError("Original xRouteBench test data is forbidden")
    if not set(
        frame.loc[frame.source_dataset.eq("routerbench"), "source_split"]
    ).issubset({"train", "validation"}):
        raise GeneralizationDataError("Original RouterBench test data is forbidden")
    if frame[["model_input_text", "duplicate_group_id"]].isna().any().any():
        raise GeneralizationDataError(
            "Input text and duplicate group IDs must be present"
        )
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    frame["model_input_text"] = frame.model_input_text.map(clean_text)
    if frame.model_input_text.str.strip().eq("").any():
        raise GeneralizationDataError("Empty normalized inputs are not allowed")
    return frame


def _map_tasks(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    tasks, domains = [], []
    for row in result.itertuples(index=False):
        section = config["task_taxonomy"][row.source_dataset]
        value = str(getattr(row, section["field"]))
        exact = section["exact"].get(value)
        matches = [
            v
            for prefix, v in section.get("prefix", {}).items()
            if value.startswith(prefix)
        ]
        if exact is not None:
            matches = [exact]
        if len(matches) != 1:
            raise GeneralizationDataError(
                f"Unmapped or ambiguous task: {row.source_dataset}/{value}"
            )
        tasks.append(matches[0]["task"])
        domains.append(matches[0]["domain"])
    result["task_group_id"] = tasks
    result["domain_group_id"] = domains
    if result.groupby("task_group_id").domain_group_id.nunique().gt(1).any():
        raise GeneralizationDataError(
            "A canonical task must belong to exactly one domain"
        )
    return result


def _assign_roles(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = _map_tasks(frame, config)
    ids = result.sample_id.astype(str).tolist()
    parent = list(range(len(result)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen_group: dict[str, int] = {}
    seen_text: dict[str, int] = {}
    for i, row in enumerate(result.itertuples(index=False)):
        for seen, key in (
            (seen_group, str(row.duplicate_group_id)),
            (seen_text, row.model_input_text),
        ):
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    members: dict[int, list[str]] = {}
    for i, sample_id in enumerate(ids):
        members.setdefault(find(i), []).append(sample_id)
    group_ids = {
        key: "g:" + hashlib.sha256("\n".join(sorted(value)).encode()).hexdigest()
        for key, value in members.items()
    }
    result["parent_duplicate_group_id"] = result.duplicate_group_id
    result["duplicate_group_id"] = [group_ids[find(i)] for i in range(len(result))]
    spec = config["split"]
    requested_tasks = set(spec["holdout_task_groups"])
    requested_domains = set(spec["holdout_domain_groups"])
    if not requested_tasks.issubset(
        set(result.task_group_id)
    ) or not requested_domains.issubset(set(result.domain_group_id)):
        raise GeneralizationDataError(
            "Requested holdout groups must exist in the input"
        )
    heldout_tasks = requested_tasks | set(
        result.loc[result.domain_group_id.isin(requested_domains), "task_group_id"]
    )
    # Whole-task transitive closure ensures duplicate-linked families cannot leak.
    linked_tasks = result.groupby("duplicate_group_id").task_group_id.agg(set).tolist()
    while True:
        expanded = heldout_tasks | set().union(
            *(tasks for tasks in linked_tasks if tasks & heldout_tasks)
        )
        if expanded == heldout_tasks:
            break
        heldout_tasks = expanded
    fractions = spec["in_domain_fractions"]
    if set(fractions) != {"train", "calibration", "selection"} or not np.isclose(
        sum(fractions.values()), 1.0
    ):
        raise GeneralizationDataError("In-domain fractions must sum to one")
    if any(not np.isfinite(v) or v <= 0 for v in fractions.values()):
        raise GeneralizationDataError(
            "All in-domain fractions must be positive and finite"
        )
    seed = int(config["seed"])
    assignments = {}
    for group in sorted(set(result.duplicate_group_id)):
        key = f"generalization-role-v1:{seed}:{group}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        assignments[group] = (
            "train"
            if value < fractions["train"]
            else "calibration"
            if value < fractions["train"] + fractions["calibration"]
            else "selection"
        )
    result["parent_split"] = result["split"]
    result["parent_dataset_version"] = result.dataset_version
    result["generalization_role"] = result.duplicate_group_id.map(assignments)
    result.loc[result.task_group_id.isin(heldout_tasks), "generalization_role"] = (
        "task_holdout"
    )
    result["split"] = result.generalization_role
    result["objective_split"] = result.generalization_role
    result["dataset_version"] = config["dataset_version"]
    result["split_version"] = spec["version"]
    if result.groupby("duplicate_group_id").generalization_role.nunique().gt(1).any():
        raise GeneralizationDataError("A duplicate group crosses roles")
    ratio = float(result.generalization_role.eq("task_holdout").mean())
    lower, upper = spec["holdout_fraction_bounds"]
    if not lower <= ratio <= upper:
        raise GeneralizationDataError(
            f"Holdout fraction {ratio:.4f} violates declared bounds"
        )
    counts = {}
    for role in ROLES:
        source_counts = (
            result.loc[result.generalization_role.eq(role)]
            .groupby("source_dataset")
            .size()
            .to_dict()
        )
        if any(
            int(source_counts.get(source, 0)) < spec["minimum_rows_per_source"][role]
            for source in SOURCES
        ):
            raise GeneralizationDataError(
                f"Insufficient per-source support for {role}: {source_counts}"
            )
        counts[role] = {source: int(source_counts[source]) for source in SOURCES}
    audit = {
        "role_source_counts": counts,
        "requested_holdout_tasks": sorted(requested_tasks),
        "requested_holdout_domains": sorted(requested_domains),
        "effective_holdout_tasks": sorted(heldout_tasks),
        "duplicate_closure_added_tasks": sorted(
            heldout_tasks
            - requested_tasks
            - set(
                result.loc[
                    result.domain_group_id.isin(requested_domains), "task_group_id"
                ]
            )
        ),
        "holdout_fraction": ratio,
        "duplicate_groups": int(result.duplicate_group_id.nunique()),
        "cross_role_duplicate_groups": 0,
        "holdout_task_overlap_with_other_roles": 0,
        "grouping_contract": "inherited-frozen-near-duplicate-groups-plus-structure-preserving-exact-union",
        "new_near_duplicate_search_performed": False,
        "task_taxonomy_uses_outcomes_or_difficulty": False,
        "holdout_interpretation": "Dataset-defined task-family and code-generation domain shift; not proof of semantic or pretraining novelty.",
    }
    return result, audit


def _balanced_weights(train: pd.DataFrame, cap: float) -> np.ndarray:
    if not np.isfinite(cap) or cap < 1.0:
        raise GeneralizationDataError("Balanced weight cap must be finite and >= 1")
    counts = train.groupby(["source_dataset", "task_group_id"]).size()
    task_counts = train.groupby("source_dataset").task_group_id.nunique()
    n, sources = len(train), train.source_dataset.nunique()
    raw = np.array(
        [
            n
            / (
                sources
                * task_counts.loc[row.source_dataset]
                * counts.loc[(row.source_dataset, row.task_group_id)]
            )
            for row in train.itertuples(index=False)
        ],
        dtype=float,
    )
    # Find scale after capping; unlike clip-then-normalize this preserves max cap.
    if cap == 1.0:
        return np.ones(n)
    low, high = 0.0, 1.0
    while np.minimum(raw * high, cap).mean() < 1.0:
        high *= 2
    for _ in range(100):
        mid = (low + high) / 2
        if np.minimum(raw * mid, cap).mean() < 1.0:
            low = mid
        else:
            high = mid
    weights = np.minimum(raw * ((low + high) / 2), cap)
    if not np.isclose(weights.mean(), 1.0, atol=1e-12) or np.max(weights) > cap:
        raise GeneralizationDataError("Bounded balancing failed its mean/cap invariant")
    return weights


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if rows is not None:
        result["rows"] = rows
    return result


def build_generalization_dataset(
    config: Mapping[str, Any], *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Create a new frozen dataset. Refuse to overwrite any existing output."""
    project_root, output_dir = Path(project_root), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GeneralizationDataError(f"Output directory must be empty: {output_dir}")
    if (
        config.get("input_contract") != "structure-preserving-v2"
        or config["backbone"]["max_length"] != 512
    ):
        raise GeneralizationDataError(
            "This version requires structure-preserving input with max_length=512"
        )
    anchor = _load_anchor(config, project_root)
    anchor, audit = _assign_roles(anchor, config)
    required_sources = {
        config["routerbench"]["labeled_path"],
        config["routerbench"]["model_tiers_path"],
        str(
            Path(config["xroutebench"]["data_dir"])
            / config["xroutebench"]["train_file"]
        ),
        str(
            Path(config["xroutebench"]["data_dir"])
            / config["xroutebench"]["candidates_file"]
        ),
    }
    if set(config["source_artifacts"]) != required_sources:
        raise GeneralizationDataError(
            "Pin exactly the four permitted source artifacts; no test/Arena/gate/stream inputs"
        )
    for path, pin in config["source_artifacts"].items():
        _check_pin(_path(project_root, path), pin)
    frames, stats_frames, arms = [], [], {}
    for source, matrix_loader in (
        ("routerbench", _routerbench_matrices),
        ("xroutebench", _xroute_matrices),
    ):
        quality, success, cost = matrix_loader(config, project_root, anchor)
        train_ids = anchor.loc[
            anchor.source_dataset.eq(source) & anchor.generalization_role.eq("train"),
            "sample_id",
        ].tolist()
        # Eligibility, pool selection and fixed cost normalization are train-only.
        complete = (
            quality.loc[train_ids].notna().all()
            & success.loc[train_ids].notna().all()
            & cost.loc[train_ids].notna().all()
        )
        models = complete.index[complete].tolist()
        if len(models) < 3:
            raise GeneralizationDataError(
                f"Fewer than three complete train candidates for {source}"
            )
        selected, statistics = select_three_arms(
            quality.loc[:, models],
            cost.loc[:, models],
            train_ids,
            tolerance=float(config["arm_selection"]["quality_tolerance"]),
            minimum_frontier_size=int(config["arm_selection"]["minimum_frontier_size"]),
        )
        projected = _project_source(
            anchor, source, quality, success, cost, selected, statistics
        )
        frames.append(projected)
        statistics["source_dataset"] = source
        stats_frames.append(statistics)
        arms[source] = {
            "selected_models": selected,
            "train_sample_ids_sha256": canonical_config_sha256(
                {"sample_ids": sorted(train_ids)}
            ),
            "train_rows": len(train_ids),
            "excluded_incomplete_train_candidates": sorted(
                complete.index[~complete].tolist()
            ),
            "decision_costs": {
                tier: float(projected[f"decision_cost_{tier}"].iloc[0])
                for tier in TIERS
            },
        }
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    frame["natural_sample_weight"] = 1.0
    frame["balanced_sample_weight"] = 1.0
    train_mask = frame.generalization_role.eq("train")
    cap = float(config["balancing"]["max_sample_weight"])
    frame.loc[train_mask, "balanced_sample_weight"] = _balanced_weights(
        frame.loc[train_mask], cap
    )
    frame["sample_weight"] = 1.0
    audit["balanced_weight_statistics"] = {
        "mean": float(frame.loc[train_mask, "balanced_sample_weight"].mean()),
        "min": float(frame.loc[train_mask, "balanced_sample_weight"].min()),
        "max": float(frame.loc[train_mask, "balanced_sample_weight"].max()),
        "max_cap": cap,
        "capped_rows": int(
            frame.loc[train_mask, "balanced_sample_weight"].ge(cap - 1e-12).sum()
        ),
        "target": "equal-source then equal-canonical-task within source, subject to final mean1 cap",
        "labels_and_outcomes_used": False,
    }
    audit["role_task_counts"] = {
        role: {str(k): int(v) for k, v in part.groupby("task_group_id").size().items()}
        for role, part in frame.groupby("generalization_role")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, roles = {}, {}
    for role in ROLES:
        rows = frame.loc[frame.generalization_role.eq(role)].reset_index(drop=True)
        filename = f"{role}.parquet"
        rows.to_parquet(output_dir / filename, index=False)
        artifacts[filename] = _artifact(output_dir / filename, rows=len(rows))
        roles[role] = {"file": filename, "rows": len(rows)}
    pd.concat(stats_frames, ignore_index=True).to_parquet(
        output_dir / "candidate-statistics.parquet", index=False
    )
    write_json(
        output_dir / "arms.json",
        {
            "selection_role": "train",
            "scope": "source-local-public-pools",
            "sources": arms,
        },
    )
    write_json(output_dir / "split-audit.json", audit)
    write_json(output_dir / "config.json", dict(config))
    for filename in (
        "candidate-statistics.parquet",
        "arms.json",
        "split-audit.json",
        "config.json",
    ):
        artifacts[filename] = _artifact(output_dir / filename)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": config["dataset_version"],
        "experiment_version": config["experiment_version"],
        "config_sha256": canonical_config_sha256(config),
        "input_contract": config["input_contract"],
        "max_length": 512,
        "roles": roles,
        "artifacts": artifacts,
        "base_artifacts": dict(config["base"]),
        "source_artifacts": dict(config["source_artifacts"]),
        "code_sha256": {
            "generalization/data.py": sha256_file(Path(__file__)),
            "objective_comparison/data.py": sha256_file(
                Path(__file__).parent.parent / "objective_comparison/data.py"
            ),
            "shared/structured_text.py": sha256_file(
                Path(__file__).parent.parent / "shared/structured_text.py"
            ),
        },
        "candidate_selection_role": "train",
        "decision_cost_fit_role": "train",
        "views": {
            "natural": "unit weights",
            "balanced": "source/task loss weights, same train rows and shuffle",
        },
        "role_usage": {
            "train": "gradient updates and source-local candidate selection",
            "calibration": "optional calibration fitting; natural weights",
            "selection": "predeclared objective/policy selection; natural weights",
            "task_holdout": "one-time diagnostic after selection lock; never trainer input",
        },
        "requires_fresh_pretrained_initialization": True,
        "previous_models_exposed_to_parent_rows": True,
        "inherited_difficulty_label_usage": "diagnostic only; historically fitted T1 labels are not refitted or used for split/weight/loss",
        "protected_prior_evaluations_used": False,
        "production_activation": False,
        "limitations": [
            "A new frozen dataset version and two training views, not a new data source.",
            "Task taxonomy and protocol were designed after earlier project diagnostics; this is prospective for fresh runs, not an untouched historical benchmark.",
            "Source-local candidate identities differ; shared output slots do not establish a deployable service pool.",
            "Inherited near-duplicate groups plus exact-input union; undetected paraphrase overlap remains possible.",
            "Task/domain holdout does not establish absence from backbone pretraining or semantic independence of all domains.",
            "Single seed comparisons are exploratory until independently repeated.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "_SUCCESS",
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        },
    )
    verify_generalization_dataset(output_dir)
    return manifest


def verify_generalization_dataset(
    root: Path, *, roles: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate seal and requested roles; never open unrequested role files."""
    root = Path(root)
    wanted = list(ROLES if roles is None else roles)
    if not wanted or len(set(wanted)) != len(wanted) or not set(wanted).issubset(ROLES):
        raise GeneralizationDataError(f"Unknown, empty or duplicate roles: {wanted}")
    seal, manifest = _json(root / "_SUCCESS"), _json(root / "manifest.json")
    if seal.get("schema_version") != SCHEMA_VERSION or seal.get(
        "manifest_sha256"
    ) != sha256_file(root / "manifest.json"):
        raise GeneralizationDataError("Manifest seal mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or set(
        manifest["roles"]
    ) != set(ROLES):
        raise GeneralizationDataError("Invalid generalization manifest schema")
    for filename in (
        "config.json",
        "arms.json",
        "split-audit.json",
        "candidate-statistics.parquet",
    ):
        _check_pin(root / filename, manifest["artifacts"][filename])
    config = _json(root / "config.json")
    if canonical_config_sha256(config) != manifest["config_sha256"]:
        raise GeneralizationDataError("Canonical config hash mismatch")
    frames = []
    for role in wanted:
        filename = f"{role}.parquet"
        if manifest["roles"][role]["file"] != filename:
            raise GeneralizationDataError("Role path must be a fixed local filename")
        _check_pin(root / filename, manifest["artifacts"][filename])
        frame = pd.read_parquet(root / filename)
        if (
            len(frame) != manifest["roles"][role]["rows"]
            or not frame.generalization_role.eq(role).all()
        ):
            raise GeneralizationDataError(
                f"Role row count or assignment mismatch: {role}"
            )
        if frame.sample_id.duplicated().any() or frame.empty:
            raise GeneralizationDataError(f"Empty or duplicate IDs: {role}")
        if (
            not frame.natural_sample_weight.eq(1.0).all()
            or not frame.sample_weight.eq(1.0).all()
        ):
            raise GeneralizationDataError("Stored natural weights must all be one")
        if role == "train":
            weights = frame.balanced_sample_weight.to_numpy(float)
            if (
                not np.isfinite(weights).all()
                or (weights <= 0).any()
                or weights.max() > config["balancing"]["max_sample_weight"]
                or not np.isclose(weights.mean(), 1.0, atol=1e-12)
            ):
                raise GeneralizationDataError("Invalid balanced train weights")
        elif not frame.balanced_sample_weight.eq(1.0).all():
            raise GeneralizationDataError("Evaluation roles must have unit weights")
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True)
    if (
        joined.sample_id.duplicated().any()
        or joined.groupby("duplicate_group_id")
        .generalization_role.nunique()
        .gt(1)
        .any()
        or joined.groupby("model_input_text").generalization_role.nunique().gt(1).any()
    ):
        raise GeneralizationDataError("A sample or duplicate crosses requested roles")
    holdout_tasks = set(_json(root / "split-audit.json")["effective_holdout_tasks"])
    if (
        joined.loc[~joined.generalization_role.eq("task_holdout"), "task_group_id"]
        .isin(holdout_tasks)
        .any()
    ):
        raise GeneralizationDataError(
            "A declared holdout task leaks into an in-domain role"
        )
    return manifest


def load_role(root: Path, role: str) -> pd.DataFrame:
    """Load one verified role with natural unit weights; holdout is explicit."""
    verify_generalization_dataset(root, roles=[role])
    frame = pd.read_parquet(Path(root) / f"{role}.parquet")
    frame["sample_weight"] = 1.0
    return frame


def load_training_view(
    root: Path, *, view: Literal["natural", "balanced"]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if view not in {"natural", "balanced"}:
        raise GeneralizationDataError(f"Unknown training view: {view}")
    manifest = verify_generalization_dataset(root, roles=["train"])
    frame = pd.read_parquet(Path(root) / "train.parquet")
    frame["sample_weight"] = frame[f"{view}_sample_weight"]
    return frame, manifest
