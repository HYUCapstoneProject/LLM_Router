from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from llm_router_labeling.objective_comparison.objectives import (
    masked_pointwise_row_losses,
    primary_row_losses,
)
from llm_router_labeling.training.r1_observations import (
    merge_observed_training,
    project_observations,
    question_sha256,
)
from llm_router_labeling.training.routing_policy import RoutingPolicy
from test_routing_policy import fixture_policy_payload


def policy():
    return RoutingPolicy.from_mapping(fixture_policy_payload())


def event(**updates):
    value = {
        "schema_version": "r1-observation-v1",
        "event_id": "e1",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "question": "A question?",
        "router_model_version": "r1-model-v1",
        "policy_sha256": policy().policy_sha256,
        "candidate_pool_version": policy().candidate_pool_version,
        "candidate_model_id": policy().candidates[0].model_id,
        "action": "low",
        "success": False,
        "assessment_source": "reference_evaluator",
        "assessment_id": "assessor-result-1",
        "assessed_at": "2026-09-26T00:00:00Z",
        "pii_redacted": True,
    }
    value.update(updates)
    return value


def project(*events, protected=()):
    return project_observations(
        {"schema_version": "r1-observation-batch-v1", "observations": list(events)},
        policy(),
        protected_question_hashes=protected,
        allowed_router_model_versions=["r1-model-v1"],
    )


def test_masked_bce_has_zero_gradient_for_unknown_candidates_and_same_full_loss():
    logits = torch.tensor([[1.0, -1.0, 2.0], [0.5, 0.2, -0.4]], requires_grad=True)
    targets = torch.tensor([[1.0, float("nan"), float("nan")], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    loss = masked_pointwise_row_losses(logits, targets, mask)
    assert torch.isfinite(loss).all()
    full = primary_row_losses(
        "pointwise", logits[1:], difficulty_ids=None, success_targets=targets[1:]
    )
    torch.testing.assert_close(loss[1:], full)
    loss.sum().backward()
    assert logits.grad[0, 1:].tolist() == [0.0, 0.0]
    assert logits.grad[0, 0] != 0


@pytest.mark.parametrize(
    "targets,mask",
    [([0, 0, 0], [0, 0, 0]), ([0.5, 0, 0], [1, 0, 0]), ([0, 0, 0], [1, 2, 0])],
)
def test_invalid_observations_cannot_reach_bce(targets, mask):
    with pytest.raises(ValueError):
        masked_pointwise_row_losses(
            torch.zeros((1, 3)), torch.tensor([targets]), torch.tensor([mask])
        )


def test_projection_preserves_unknowns_without_fake_difficulty_or_final_success():
    rows, manifest = project(event())
    row = rows.iloc[0]
    assert row.success_low == 0 and row.observed_low == 1
    assert pd.isna(row.success_medium) and row.observed_medium == 0
    assert pd.isna(row.success_high) and row.observed_high == 0
    assert "difficulty_label" not in rows
    assert manifest["observed_counts"] == {"low": 1, "medium": 0, "high": 0}
    assert (
        project(event())[1]["canonical_rows_sha256"]
        == manifest["canonical_rows_sha256"]
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"success": 1},
        {"assessment_source": "chain_finished"},
        {"pii_redacted": False},
        {"assessed_at": "2026-09-26T00:00:00"},
        {"candidate_model_id": "unknown"},
        {"policy_sha256": "0" * 64},
        {"router_model_version": "unapproved"},
        {"unexpected": 1},
        {"event_id": "path/event"},
        {"request_id": "q1"},
        {"assessed_at": "2026-09-26 00:00:00+00:00"},
        {"assessment_id": "with space"},
    ],
)
def test_projection_rejects_implicit_or_wrong_provenance(updates):
    with pytest.raises(ValueError):
        project(event(**updates))


def test_projection_rejects_conflicting_candidate_observations():
    with pytest.raises(ValueError, match="Conflicting"):
        project(event(), event(event_id="e2", success=True))
    with pytest.raises(ValueError, match="Duplicate event"):
        project(event(), event())


def test_protected_questions_cannot_become_feedback_training_rows():
    good = event(
        event_id="e2",
        request_id="00000000-0000-4000-8000-000000000002",
        question="Another question?",
    )
    rows, manifest = project(event(), good, protected=[question_sha256("A question?")])
    assert rows.request_id.tolist() == ["00000000-0000-4000-8000-000000000002"]
    assert manifest["protected_observations_excluded"] == 1


def test_merge_preserves_validation_masks_and_rejects_cross_pool_or_leakage():
    base, _ = project(event())
    base["split"] = "validation"
    feedback, _ = project(
        event(
            event_id="e2",
            request_id="00000000-0000-4000-8000-000000000002",
            question="New training question",
        )
    )
    merged = merge_observed_training(base, feedback)
    assert merged.split.tolist() == ["validation", "train"]
    assert merged.observed_high.tolist() == [0, 0]
    assert merged.success_high.isna().all()
    wrong = feedback.copy()
    wrong["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy_sha256"):
        merge_observed_training(base, wrong)
    leak = feedback.copy()
    leak["duplicate_group_id"] = base.iloc[0].duplicate_group_id
    with pytest.raises(ValueError, match="protected"):
        merge_observed_training(base, leak)


def test_protected_text_is_checked_across_different_group_namespaces():
    base, _ = project(event())
    base["split"] = "validation"
    base["duplicate_group_id"] = "base-evaluation-group-1"
    feedback, _ = project(
        event(event_id="e2", request_id="00000000-0000-4000-8000-000000000002")
    )
    with pytest.raises(ValueError, match="protected.*questions"):
        merge_observed_training(base, feedback)


def test_actual_spring_export_and_policy_prepare_immutable_parquet(tmp_path):
    from llm_router_labeling.shared.dataset_io import sha256_file
    from llm_router_labeling.training.routing_policy import load_routing_policy

    fixtures = Path(__file__).parent / "fixtures/r1-observations"
    policy_file = fixtures / "routing_policy.json"
    exported = fixtures / "observation_batch.json"
    hashes = tmp_path / "protected.json"
    hashes.write_text("[]")
    output = tmp_path / "prepared"
    cli = Path(__file__).parents[1] / "scripts/prepare-r1-observations.py"
    command = [
        sys.executable,
        str(cli),
        "--observations",
        str(exported),
        "--policy",
        str(policy_file),
        "--policy-file-sha256",
        sha256_file(policy_file),
        "--protected-question-hashes",
        str(hashes),
        "--router-model-version",
        "router-r1-fixture",
        "--output-dir",
        str(output),
    ]
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    rows = pd.read_parquet(output / "feedback-training.parquet")
    assert manifest["output_sha256"] == sha256_file(
        output / "feedback-training.parquet"
    )
    assert manifest["policy_sha256"] == load_routing_policy(policy_file).policy_sha256
    assert rows.success_low.tolist() == [0]
    assert rows.observed_medium.tolist() == [0]
    assert rows.success_medium.isna().all()
    assert "difficulty_label" not in rows
    repeat = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeat.returncode != 0
    assert "immutable" in repeat.stderr
