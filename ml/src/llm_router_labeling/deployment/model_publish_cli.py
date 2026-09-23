from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from ..shared.dataset_io import write_json
from ..shared.object_store import LocalObjectStore, ObjectStore, build_object_store
from ..shared.observability import write_observability_json
from .model_publish import (
    approve_candidate,
    candidate_uri,
    publish_candidate,
    read_current_pointer,
    rollback_current_pointer,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
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

    if args.command == "publish-candidate":
        result = publish_candidate(
            args.package_dir,
            root_uri=args.root_uri,
            model_version=args.model_version,
            object_store=store,
            gate_result=_read_json(args.gate_result) if args.gate_result else None,
        )
    elif args.command == "approve":
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
        with tempfile.TemporaryDirectory(prefix="router-approve-") as work_dir:
            result = approve_candidate(
                root_uri=args.root_uri,
                model_version=args.model_version,
                object_store=store,
                approved_by=args.approved_by,
                pipeline_run_id=args.pipeline_run_id,
                work_dir=Path(work_dir),
                allow_pointer_rollback=args.allow_pointer_rollback,
                logical_model_version=args.logical_model_version,
                training_mode=args.training_mode,
                retention_anchor_model_version=args.retention_anchor_model_version,
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

    if getattr(args, "output", None):
        write_json(args.output, result)
    if args.command == "approve" and args.version_policy_mode == "validatedUniqueBatch":
        _commit_validated_batch(args, result, store)
    if args.command == "approve" and args.observability_run_uri:
        if not args.batch_id:
            raise SystemExit("observability requires --batch-id")
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/result/publish-result.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.model_version,
                "outcome": "promoted",
                "approved_at": result.get("approved_at"),
                "model_version": result.get("model_version"),
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


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
