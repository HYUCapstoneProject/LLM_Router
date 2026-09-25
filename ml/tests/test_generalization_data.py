"""CPU-only synthetic checks; no real models, cloud, or protected evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from llm_router_labeling.generalization import data
from llm_router_labeling.shared.dataset_io import sha256_file, write_json


def _pin(path: Path) -> dict:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture()
def fixture(tmp_path, monkeypatch):
    rows = []
    for source in data.SOURCES:
        for i in range(200):
            task = (
                "general_a"
                if i < 140
                else "general_b"
                if i < 180
                else "held_task"
                if i < 190
                else "code"
            )
            rows.append(
                {
                    "sample_id": f"{source}:synthetic-{i:03}",
                    "source_dataset": source,
                    "source_split": "train",
                    "source_task": task
                    if source == "routerbench"
                    else f"question-id-{i}",
                    "category": task,
                    "model_input_text": f"Synthetic {source} question {i}; x + {i}",
                    "difficulty_label": i % 3,
                    "duplicate_group_id": f"old:{source}:{i}",
                    "split": "train" if i % 3 else "validation",
                    "dataset_version": "synthetic-parent",
                    "sample_weight": 0.5,
                }
            )
    anchor = pd.DataFrame(rows)
    anchor.to_parquet(tmp_path / "training.parquet", index=False)
    write_json(
        tmp_path / "base-manifest.json", {"coverage_resolution_status": "complete"}
    )
    inputs = {}
    for name in ("rb.parquet", "tiers.json", "xr-train.parquet", "candidates.parquet"):
        (tmp_path / name).write_text(
            "SYNTHETIC: unused because outcome readers are mocked\n"
        )
        inputs[name] = {k: v for k, v in _pin(tmp_path / name).items() if k != "path"}
    tasks = {
        name: {
            "task": name,
            "domain": "code_generation" if name == "code" else "general",
        }
        for name in ("general_a", "general_b", "held_task", "code")
    }
    config = {
        "dataset_version": "SYNTHETIC-generalization-test",
        "experiment_version": "SYNTHETIC-generalization-test",
        "seed": 42,
        "input_contract": "structure-preserving-v2",
        "base": {
            "manifest": _pin(tmp_path / "base-manifest.json"),
            "training": _pin(tmp_path / "training.parquet"),
        },
        "source_artifacts": inputs,
        "routerbench": {"labeled_path": "rb.parquet", "model_tiers_path": "tiers.json"},
        "xroutebench": {
            "data_dir": ".",
            "train_file": "xr-train.parquet",
            "test_file": "FORBIDDEN.parquet",
            "candidates_file": "candidates.parquet",
        },
        "task_taxonomy": {
            "routerbench": {"field": "source_task", "exact": tasks},
            "xroutebench": {"field": "category", "exact": tasks},
        },
        "split": {
            "version": "synthetic-v1",
            "holdout_task_groups": ["held_task"],
            "holdout_domain_groups": ["code_generation"],
            "in_domain_fractions": {"train": 0.8, "calibration": 0.1, "selection": 0.1},
            "holdout_fraction_bounds": [0.02, 0.5],
            "minimum_rows_per_source": {role: 3 for role in data.ROLES},
        },
        "backbone": {"max_length": 512},
        "balancing": {"max_sample_weight": 3.0},
        "arm_selection": {"quality_tolerance": 1e-12, "minimum_frontier_size": 3},
    }

    def matrices(source, anchor_frame, poison=False):
        part = anchor_frame.loc[anchor_frame.source_dataset.eq(source)]
        quality = pd.DataFrame(
            np.tile([0.1, 0.6, 0.7, 0.9], (len(part), 1)),
            index=part.sample_id,
            columns=[f"{source}:synthetic-model-{i}" for i in range(4)],
        )
        success = quality.ge(0.5).astype(float)
        cost = pd.DataFrame(
            np.tile([1.0, 2.0, 3.0, 6.0], (len(part), 1)),
            index=quality.index,
            columns=quality.columns,
        )
        if poison:
            held = part.loc[part.generalization_role.eq("task_holdout"), "sample_id"]
            quality.loc[held] = [0.99, 0.9, 0.8, 0.0]
            cost.loc[held] = [99, 50, 3, 0.1]
        return quality, success, cost

    monkeypatch.setattr(
        data,
        "_routerbench_matrices",
        lambda config, root, frame: matrices("routerbench", frame),
    )
    monkeypatch.setattr(
        data,
        "_xroute_matrices",
        lambda config, root, frame: matrices("xroutebench", frame),
    )
    return tmp_path, config, anchor, matrices


def test_roles_weights_candidate_provenance_and_exposure(fixture):
    root, config, _, _ = fixture
    output = root / "out"
    manifest = data.build_generalization_dataset(
        config, project_root=root, output_dir=output
    )
    frames = {role: data.load_role(output, role) for role in data.ROLES}
    joined = pd.concat(frames.values())
    assert len(joined) == 400
    assert joined.groupby("duplicate_group_id").generalization_role.nunique().max() == 1
    assert set(frames["task_holdout"].task_group_id) == {"held_task", "code"}
    assert all(
        not set(frame.task_group_id) & {"held_task", "code"}
        for role, frame in frames.items()
        if role != "task_holdout"
    )
    natural, _ = data.load_training_view(output, view="natural")
    balanced, _ = data.load_training_view(output, view="balanced")
    pd.testing.assert_frame_equal(
        natural.drop(columns="sample_weight"), balanced.drop(columns="sample_weight")
    )
    assert natural.sample_weight.eq(1).all()
    assert balanced.sample_weight.max() <= 3
    assert balanced.sample_weight.mean() == pytest.approx(1, abs=1e-12)
    assert not balanced.sample_weight.eq(1).all()
    assert all(frame.sample_weight.eq(1).all() for frame in frames.values())
    assert manifest["requires_fresh_pretrained_initialization"]
    assert manifest["previous_models_exposed_to_parent_rows"]
    assert manifest["protected_prior_evaluations_used"] is False
    assert manifest["config_sha256"] == data.canonical_config_sha256(config)
    arms = json.loads((output / "arms.json").read_text())
    assert arms["selection_role"] == "train"
    for source in data.SOURCES:
        ids = natural.loc[natural.source_dataset.eq(source), "sample_id"].tolist()
        assert arms["sources"][source][
            "train_sample_ids_sha256"
        ] == data.canonical_config_sha256({"sample_ids": sorted(ids)})


def test_deterministic_build_and_input_order(fixture, monkeypatch):
    root, config, anchor, _ = fixture
    data.build_generalization_dataset(
        config, project_root=root, output_dir=root / "one"
    )
    data.build_generalization_dataset(
        config, project_root=root, output_dir=root / "two"
    )
    for path in (root / "one").iterdir():
        assert path.read_bytes() == (root / "two" / path.name).read_bytes()
    first, _ = data._assign_roles(anchor, config)
    shuffled, _ = data._assign_roles(
        anchor.sample(frac=1, random_state=99).reset_index(drop=True), config
    )
    pd.testing.assert_frame_equal(
        first.set_index("sample_id").sort_index(),
        shuffled.set_index("sample_id").sort_index(),
    )


def test_heldout_outcomes_cannot_change_candidate_selection(fixture, monkeypatch):
    root, config, _, matrices = fixture
    data.build_generalization_dataset(
        config, project_root=root, output_dir=root / "one"
    )
    monkeypatch.setattr(
        data,
        "_routerbench_matrices",
        lambda config, root, frame: matrices("routerbench", frame, poison=True),
    )
    monkeypatch.setattr(
        data,
        "_xroute_matrices",
        lambda config, root, frame: matrices("xroutebench", frame, poison=True),
    )
    data.build_generalization_dataset(
        config, project_root=root, output_dir=root / "two"
    )
    for name in (
        "arms.json",
        "candidate-statistics.parquet",
        "train.parquet",
        "calibration.parquet",
        "selection.parquet",
    ):
        assert (root / "one" / name).read_bytes() == (root / "two" / name).read_bytes()
    assert (root / "one/task_holdout.parquet").read_bytes() != (
        root / "two/task_holdout.parquet"
    ).read_bytes()


def test_duplicate_task_closure_and_cross_source_exact_union(fixture):
    _, config, anchor, _ = fixture
    # An exact copy links held_task to general_b across sources despite distinct old IDs.
    anchor.loc[anchor.sample_id.eq("xroutebench:synthetic-140"), "model_input_text"] = (
        anchor.loc[
            anchor.sample_id.eq("routerbench:synthetic-180"), "model_input_text"
        ].iloc[0]
    )
    assigned, audit = data._assign_roles(anchor, config)
    assert audit["duplicate_closure_added_tasks"] == ["general_b"]
    assert (
        assigned.loc[assigned.task_group_id.eq("general_b"), "generalization_role"]
        .eq("task_holdout")
        .all()
    )
    assert assigned.groupby("model_input_text").generalization_role.nunique().max() == 1


def test_labels_never_affect_roles_or_weights(fixture):
    _, config, anchor, _ = fixture
    before, _ = data._assign_roles(anchor, config)
    anchor["difficulty_label"] = 99
    anchor["sample_weight"] = 1000
    after, _ = data._assign_roles(anchor, config)
    assert before.generalization_role.equals(after.generalization_role)
    assert np.array_equal(
        data._balanced_weights(before.loc[before.generalization_role.eq("train")], 3),
        data._balanced_weights(after.loc[after.generalization_role.eq("train")], 3),
    )


def test_final_weight_cap_after_normalization():
    frame = pd.DataFrame(
        {
            "source_dataset": ["large"] * 1000 + ["tiny"],
            "task_group_id": ["same"] * 1001,
        }
    )
    weights = data._balanced_weights(frame, 3)
    assert weights[-1] == 3
    assert weights.max() <= 3
    assert weights.mean() == pytest.approx(1, abs=1e-12)


def test_partial_verify_and_loading_do_not_open_holdout(fixture, monkeypatch):
    root, config, _, _ = fixture
    output = root / "out"
    data.build_generalization_dataset(config, project_root=root, output_dir=output)
    (output / "task_holdout.parquet").unlink()
    data.verify_generalization_dataset(
        output, roles=["train", "calibration", "selection"]
    )
    data.load_training_view(output, view="balanced")
    data.load_role(output, "calibration")
    with pytest.raises(data.GeneralizationDataError, match="missing file"):
        data.verify_generalization_dataset(output)


@pytest.mark.parametrize(
    "filename",
    ["train.parquet", "selection.parquet", "manifest.json", "config.json", "arms.json"],
)
def test_hash_tamper_is_rejected(fixture, filename):
    root, config, _, _ = fixture
    output = root / "out"
    data.build_generalization_dataset(config, project_root=root, output_dir=output)
    with (output / filename).open("ab") as f:
        f.write(b" ")
    with pytest.raises(data.GeneralizationDataError):
        data.verify_generalization_dataset(output)


def test_forbidden_parent_role_unknown_task_and_overwrite_rejected(fixture):
    root, config, anchor, _ = fixture
    output = root / "out"
    data.build_generalization_dataset(config, project_root=root, output_dir=output)
    with pytest.raises(data.GeneralizationDataError, match="must be empty"):
        data.build_generalization_dataset(config, project_root=root, output_dir=output)
    with pytest.raises(data.GeneralizationDataError, match="Unknown"):
        data.load_role(output, "test_routerbench")
    anchor.loc[0, "source_task"] = "unknown"
    with pytest.raises(data.GeneralizationDataError, match="Unmapped"):
        data._assign_roles(anchor, config)
    anchor.loc[0, "split"] = "test"
    anchor.to_parquet(root / "training.parquet", index=False)
    modified = copy.deepcopy(config)
    modified["base"]["training"] = _pin(root / "training.parquet")
    with pytest.raises(data.GeneralizationDataError, match="Only previous"):
        data.build_generalization_dataset(
            modified, project_root=root, output_dir=root / "forbidden"
        )


def test_config_hash_is_order_independent_and_no_nonfinite_values():
    assert data.canonical_config_sha256(
        {"a": 1, "b": 2}
    ) == data.canonical_config_sha256({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        data.canonical_config_sha256({"a": float("nan")})
