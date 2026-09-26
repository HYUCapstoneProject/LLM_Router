from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from ..shared.dataset_io import sha256_file, write_json
from ..shared.object_store import LocalObjectStore, ObjectStore, build_object_store
from ..shared.observability import write_observability_json
from .model_publish import (
    approve_candidate,
    candidate_uri,
    publish_candidate,
    read_current_pointer,
    rollback_current_pointer,
)
from .promotion_evidence import (
    PromotionEvidenceError,
    build_promotion_evidence,
    validate_gate_package_binding,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _object_store(args: argparse.Namespace) -> ObjectStore:
    if args.local_store_root is not None:
        return LocalObjectStore(args.local_store_root)
    return build_object_store(args.root_uri, project=args.project)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish, approve, and roll back router model packages on GCS"
    )
    parser.add_argument(
        "--local-store-root",
        type=Path,
        help="Use a filesystem-backed store instead of GCS (dry runs and tests)",
    )
    parser.add_argument("--project", help="GCP project for the GCS client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish-candidate", help="Upload a gate-passed package to candidates"
    )
    publish.add_argument("--package-dir", type=Path, required=True)
    publish.add_argument("--root-uri", required=True)
    publish.add_argument("--model-version", required=True)
    publish.add_argument("--gate-result", type=Path)
    publish.add_argument("--output", type=Path)

    approve = subparsers.add_parser(
        "approve", help="Copy a candidate to approved and move the CURRENT pointer"
    )
    approve.add_argument("--root-uri", required=True)
    approve.add_argument("--model-version", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--pipeline-run-id")
    approve.add_argument("--logical-model-version")
    approve.add_argument(
        "--training-mode",
        choices=(
            "initial_training",
            "minor_fine_tune",
            "major_consolidation",
            "pristine_rebase",
        ),
    )
    approve.add_argument("--retention-anchor-model-version")
    approve.add_argument(
        "--publish-result",
        type=Path,
        help="Pipeline publish_result.json used to prove the candidate upload completed",
    )
    approve.add_argument("--output", type=Path)
    approve.add_argument("--observability-run-uri")
    approve.add_argument("--batch-id")
    approve.add_argument(
        "--version-policy-mode",
        choices=("successfulHttpRequest", "validatedUniqueBatch"),
        default="successfulHttpRequest",
    )
    approve.add_argument("--retraining-control-root-uri")
    approve.add_argument(
        "--allow-pointer-rollback",
        action="store_true",
        help="Permit approving a model older than the current one",
    )

    rollback = subparsers.add_parser(
        "rollback", help="Point CURRENT back at the previously approved model"
    )
    rollback.add_argument("--root-uri", required=True)
    rollback.add_argument("--approved-by", required=True)
    rollback.add_argument("--output", type=Path)

    show = subparsers.add_parser("show-current", help="Print the CURRENT pointer")
    show.add_argument("--root-uri", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = _object_store(args)

    if (
        args.command == "approve"
        and args.observability_run_uri
        and (not args.batch_id or not args.pipeline_run_id or not args.publish_result)
    ):
        raise SystemExit("Pipeline approval requires batch, job and publish evidence")
    if args.command == "publish-candidate":
        gate = _read_json(args.gate_result) if args.gate_result else None
        if gate is not None:
            try:
                validate_gate_package_binding(
                    _read_json(args.package_dir / "model_manifest.json"), gate
                )
            except PromotionEvidenceError as exc:
                raise SystemExit(str(exc)) from exc
            if (
                gate.get("passed") is not True
                or gate.get("candidate_model_version") != args.model_version
            ):
                raise SystemExit("Gate result does not authorize this candidate")
            if gate.get("candidate_manifest_sha256") != sha256_file(
                args.package_dir / "model_manifest.json"
            ):
                raise SystemExit("Gate result package checksum mismatch")
        result = publish_candidate(
            args.package_dir,
            root_uri=args.root_uri,
            model_version=args.model_version,
            object_store=store,
            gate_result=gate,
        )
        result["candidate_manifest_sha256"] = sha256_file(
            args.package_dir / "model_manifest.json"
        )
        result["gate_result"] = gate
    elif args.command == "approve":
        resumed_pointer = None
        gate = {}
        if args.publish_result:
            published = _read_json(args.publish_result)
            expected_uri = candidate_uri(args.root_uri, args.model_version)
            if published.get("model_version") != args.model_version:
                raise SystemExit(
                    "publish result model_version does not match approval request"
                )
            if published.get("package_uri") != expected_uri:
                raise SystemExit(
                    "publish result package_uri does not match candidate location"
                )
            current = read_current_pointer(args.root_uri, object_store=store)
            if (
                current
                and current.get("model_version") == args.model_version
                and args.observability_run_uri
            ):
                if current.get(
                    "pipeline_run_id"
                ) != args.pipeline_run_id or current.get(
                    "candidate_manifest_sha256"
                ) != published.get("candidate_manifest_sha256"):
                    raise SystemExit("Existing CURRENT belongs to a different approval")
                resumed_pointer = current
            else:
                candidate_text = store.read_text(
                    expected_uri.rstrip("/") + "/model_manifest.json"
                )
                if candidate_text is None or _text_sha(candidate_text) != published.get(
                    "candidate_manifest_sha256"
                ):
                    raise SystemExit("Published candidate manifest checksum mismatch")
            gate = published.get("gate_result") or {}
            if not isinstance(gate, dict):
                raise SystemExit("Publish gate evidence must be an object")
            # Recheck the actual package contract before approving or resuming.
            manifest_uri = (
                args.root_uri.rstrip("/")
                + f"/{args.model_version}/approved/model_manifest.json"
                if resumed_pointer is not None
                else expected_uri.rstrip("/") + "/model_manifest.json"
            )
            manifest_text = store.read_text(manifest_uri)
            if manifest_text is None:
                raise SystemExit("Published manifest is missing")
            try:
                validate_gate_package_binding(json.loads(manifest_text), gate)
            except PromotionEvidenceError as exc:
                raise SystemExit(str(exc)) from exc
            if (
                gate.get("passed") is not True
                or gate.get("candidate_model_version") != args.model_version
                or gate.get("candidate_manifest_sha256")
                != published.get("candidate_manifest_sha256")
            ):
                raise SystemExit("Publish evidence is not bound to a passing gate")
            if gate.get("qualification_mode") == "initial_baseline":
                if args.training_mode != "initial_training":
                    raise SystemExit("Initial qualification cannot authorize feedback retraining")
                if gate.get("model_publish_root_uri") != args.root_uri.rstrip("/"):
                    raise SystemExit("Initial qualification belongs to another model registry")
            if args.observability_run_uri:
                if (
                    gate.get("batch_id") != args.batch_id
                    or gate.get("pipeline_job_resource_name") != args.pipeline_run_id
                ):
                    raise SystemExit("Publish gate belongs to another batch or job")
                gate_text = store.read_text(
                    args.observability_run_uri.rstrip("/") + "/gate/gate-result.json"
                )
                if gate_text is None:
                    raise SystemExit("Gate observability evidence is missing")
                if (
                    args.training_mode not in (None, "initial_training")
                    and resumed_pointer is None
                    and (
                        not current
                        or current.get("model_version")
                        != gate.get("baseline_model_version")
                    )
                ):
                    raise SystemExit("Approved baseline changed since qualification")
                observed_gate = json.loads(gate_text)
                if any(
                    observed_gate.get(key) != value
                    for key, value in gate.items()
                    if key != "schema_version"
                ):
                    raise SystemExit(
                        "Gate observability evidence does not match publication"
                    )
        if args.version_policy_mode == "validatedUniqueBatch":
            _validate_batch_reservation(args, store, resumed_pointer is not None)
        if resumed_pointer is not None:
            result = resumed_pointer
        else:
            with tempfile.TemporaryDirectory(prefix="router-approve-") as work_dir:
                result = approve_candidate(
                    root_uri=args.root_uri,
                    model_version=args.model_version,
                    object_store=store,
                    approved_by=args.approved_by,
                    pipeline_run_id=args.pipeline_run_id,
                    work_dir=Path(work_dir),
                    allow_pointer_rollback=args.allow_pointer_rollback,
                    expected_baseline_model_version=(
                        gate["baseline_model_version"]
                        if args.observability_run_uri
                        and args.training_mode not in (None, "initial_training")
                        else None
                    ),
                    expected_baseline_manifest_sha256=(
                        gate.get("baseline_manifest_sha256")
                        if args.observability_run_uri
                        and args.training_mode not in (None, "initial_training")
                        else None
                    ),
                    logical_model_version=args.logical_model_version,
                    training_mode=args.training_mode,
                    retention_anchor_model_version=args.retention_anchor_model_version,
                    initial_qualification=(gate if gate.get("qualification_mode") == "initial_baseline" else None),
                )
    elif args.command == "rollback":
        result = rollback_current_pointer(
            root_uri=args.root_uri,
            object_store=store,
            approved_by=args.approved_by,
        )
    else:
        pointer = read_current_pointer(args.root_uri, object_store=store)
        if pointer is None:
            raise SystemExit("No CURRENT.json exists under the given root URI")
        result = pointer

    evidence = None
    if args.command == "approve" and args.observability_run_uri:
        evidence = _promotion_evidence(args, result, store, gate_text)
    if getattr(args, "output", None):
        write_json(args.output, result)
    if args.command == "approve" and args.version_policy_mode == "validatedUniqueBatch":
        _commit_validated_batch(args, result, store)
    if args.command == "approve" and args.observability_run_uri:
        if not args.batch_id:
            raise SystemExit("observability requires --batch-id")
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/result/publish-result.json",
            evidence,
            object_store=store,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _promotion_evidence(args, result, store, gate_text):
    try:
        return build_promotion_evidence(
            root_uri=args.root_uri,
            model_version=args.model_version,
            batch_id=args.batch_id,
            pipeline_run_id=args.pipeline_run_id,
            result=result,
            store=store,
            gate_text=gate_text,
        )
    except PromotionEvidenceError as exc:
        raise SystemExit(str(exc)) from exc


def _validate_batch_reservation(args, store, allow_committed):
    if not args.retraining_control_root_uri or not args.batch_id:
        raise SystemExit("validatedUniqueBatch requires control root and batch ID")
    root = args.retraining_control_root_uri.rstrip("/") + "/validated-batches/"
    states = ("pending", "committed") if allow_committed else ("pending",)
    matches = []
    for state in states:
        for uri in store.list_uris(root + state):
            value = json.loads(store.read_text(uri))
            if (
                value.get("batch_id") == args.batch_id
                and value.get("candidate_model_version") == args.model_version
            ):
                matches.append((state, value))
    # A crash may leave both the durable committed record and its pending
    # predecessor. Permit only that same-digest transition on an approved retry.
    recovering_transition = (
        allow_committed
        and len(matches) == 2
        and {state for state, _ in matches} == {"pending", "committed"}
        and len({value.get("canonical_batch_digest") for _, value in matches}) == 1
    )
    if len(matches) != 1 and not recovering_transition:
        raise SystemExit("Validated batch reservation was not uniquely found")


def _commit_validated_batch(
    args: argparse.Namespace, result: dict[str, Any], store: ObjectStore
) -> None:
    if not args.retraining_control_root_uri or not args.batch_id:
        raise SystemExit(
            "validatedUniqueBatch approval requires control root and batch ID"
        )
    root = args.retraining_control_root_uri.rstrip("/")
    pending_root = f"{root}/validated-batches/pending"
    matches: list[tuple[str, dict[str, Any]]] = []
    for uri in store.list_uris(pending_root):
        text = store.read_text(uri)
        if not text:
            continue
        marker = json.loads(text)
        if (
            marker.get("batch_id") == args.batch_id
            and marker.get("candidate_model_version") == args.model_version
        ):
            matches.append((uri, marker))
    if not matches:
        for uri in store.list_uris(f"{root}/validated-batches/committed"):
            marker = json.loads(store.read_text(uri))
            if (
                marker.get("batch_id") == args.batch_id
                and marker.get("candidate_model_version") == args.model_version
                and marker.get("approved_at") == result.get("approved_at")
                and marker.get("model_version") == result.get("model_version")
            ):
                return
    if len(matches) != 1:
        raise SystemExit("validated feedback pending marker was not uniquely found")
    pending_uri, marker = matches[0]
    digest = str(marker["canonical_batch_digest"])
    committed_uri = f"{root}/validated-batches/committed/{digest}.json"
    committed = {
        **marker,
        "approved_at": result.get("approved_at"),
        "model_version": result.get("model_version"),
    }
    text = json.dumps(committed, sort_keys=True, separators=(",", ":"))
    existing = store.read_text(committed_uri)
    if existing is None:
        store.write_text(committed_uri, text, if_generation_match=None)
    elif json.loads(existing) != committed:
        raise SystemExit("validated feedback digest is committed to another run")
    store.delete_uri(pending_uri)


if __name__ == "__main__":
    main()
