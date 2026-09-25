"""Validate shared cloud layout and collect a bounded, read-only GCP inventory.

An inventory is not an effective IAM test, a release approval, or a deployment.
The module has no cloud mutation, credential export, or job submission path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "router-cloud-foundation-v1"
IMAGE_ROLES = ("data_processing", "training", "evaluation", "promotion")
ROOT_ROLES = (
    "config",
    "pipelines",
    "feedback",
    "control",
    "observability",
    "training_output",
    "registry",
    "pipeline_artifacts",
)
POST_DEPLOY_CHECKS = [
    "Verify mounted model/SQLite volumes, UID 10001 writes, and backup restoration.",
    "Verify running identities, secret injection and Spring-to-FastAPI connectivity.",
    "Measure 8192-token FastAPI latency, peak memory and concurrency on the target VM.",
    "Verify accepted batch -> Vertex -> gate -> CURRENT -> serving activation.",
    "Verify retry, load failure, concurrent promotion and previous-package rollback.",
]
IAM_CHECKS = [
    "Review inherited project and bucket IAM, conditions, and object-list permissions.",
    "Verify Serving -> Pipeline SA and Pipeline -> Training SA actAs on those accounts.",
    "Verify per-identity reads/writes and Vertex-managed image-pull identity permissions.",
]


def _gcs(value: object, bucket: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme == "gs"
        and parsed.netloc == bucket
        and bool(parsed.path.strip("/"))
        and not parsed.query
        and not parsed.fragment
        and not any(
            part in (".", "..", "") for part in parsed.path.lstrip("/").split("/")
        )
        and "%" not in parsed.path
    )


def _overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def namespace_sha256(profile: dict[str, Any], accounts: dict[str, Any]) -> str:
    contract = {
        key: profile.get(key)
        for key in (
            *(f"{role}_root_uri" for role in ROOT_ROLES),
            "current_pointer_uri",
            "sqlite_path",
        )
    }
    contract["service_accounts"] = accounts
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_provisioning_evidence(
    name: str, profile: dict[str, Any], accounts: dict[str, Any], root: Path
) -> None:
    reference = profile.get("namespace_evidence")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise TypeError("provisioned namespace requires a local evidence path and SHA")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("namespace evidence must stay inside the evidence root")
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size > 1_000_000
    ):
        raise ValueError(
            "namespace evidence is missing, outside its root, or too large"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.get("sha256"):
        raise ValueError("namespace evidence SHA mismatch")
    evidence = json.loads(raw)
    if not isinstance(evidence, dict) or any(
        evidence.get(key) != value
        for key, value in {
            "schema_version": "router-namespace-provisioning-v1",
            "profile": name,
            "namespace_sha256": namespace_sha256(profile, accounts),
            "iam_configuration_verified": True,
            "prefix_access_probe_verified": True,
            "production_pointer_unchanged": True,
        }.items()
    ):
        raise ValueError(
            "namespace evidence does not bind the current layout, identities and required checks"
        )
    if not isinstance(evidence.get("observed_at"), str):
        raise TypeError("namespace evidence requires an observation timestamp")
    datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00"))


def validate_config(
    config: dict[str, Any], *, evidence_root: Path | None = None
) -> tuple[list[str], list[str]]:
    """Return structural errors and outstanding release inputs, without side effects."""
    errors: list[str] = []
    pending: list[str] = []
    if config.get("schema_version") != SCHEMA:
        errors.append("schema_version must be router-cloud-foundation-v1")
    project = config.get("project", "")
    if not isinstance(project, str) or not re.fullmatch(
        r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project
    ):
        errors.append("project must be a Google Cloud project ID")
    bucket = config.get("bucket", "")
    if not isinstance(bucket, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", bucket
    ):
        errors.append("bucket must be an explicit bucket name")
    for key in ("region", "artifact_repository", "serving_instance", "serving_zone"):
        if not isinstance(config.get(key), str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*", config[key]
        ):
            errors.append(f"{key} must be explicit")
    deployment = config.get("deployment_policy", {})
    required_deployment = {
        "service_mode": "single-active-objective",
        "http_contract": "existing-v1-unchanged",
        "gcs_model_groups": "active-objective-only",
        "inactive_model_storage": "google-drive",
        "switch_requires_remote_archive_readback": True,
        "switch_requires_no_inflight_retraining": True,
        "cross_objective_gcs_overlap_allowed": False,
    }
    if not isinstance(deployment, dict):
        errors.append("deployment_policy must be an object")
        deployment = {}
    for key, expected in required_deployment.items():
        if (
            type(deployment.get(key)) is not type(expected)
            or deployment.get(key) != expected
        ):
            errors.append(
                f"deployment_policy.{key} must preserve the single-group, unchanged-API contract"
            )
    target_profile = deployment.get("target_profile")
    if target_profile not in ("r0", "r1"):
        errors.append("deployment_policy.target_profile must be r0 or r1")
    accounts = config.get("service_accounts", {})
    if not isinstance(accounts, dict):
        errors.append("service_accounts must be an object")
        accounts = {}
    for role in ("pipeline", "training", "serving"):
        value = accounts.get(role)
        if role == "serving" and value is None:
            pending.append(
                "Discover and pin the serving service account from the VM inventory."
            )
        elif not isinstance(value, str) or not re.fullmatch(
            rf"[a-z0-9-]+@{re.escape(str(project))}\.iam\.gserviceaccount\.com", value
        ):
            errors.append(
                f"service_accounts.{role} must be an explicit account in this project"
            )
    apis = config.get("required_apis")
    if (
        not isinstance(apis, list)
        or not apis
        or any(
            not isinstance(api, str)
            or not re.fullmatch(r"[a-z0-9]+\.googleapis\.com", api)
            for api in apis
        )
    ):
        errors.append("required_apis must contain explicit Google API names")
    protected = config.get("protected_current_uris", [])
    if (
        not isinstance(protected, list)
        or not protected
        or any(not _gcs(uri, bucket) for uri in protected)
    ):
        errors.append(
            "protected_current_uris must identify the existing operational pointers"
        )
        protected = []
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict) or set(profiles) != {"r0", "r1"}:
        errors.append("profiles must define exactly r0 and r1")
        profiles = {}
    roots: list[tuple[str, str]] = []
    databases: list[str] = []
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            errors.append(f"profiles.{name} must be an object")
            continue
        if profile.get("namespace_status") not in ("planned", "provisioned"):
            errors.append(f"{name}: namespace_status must be planned or provisioned")
        elif profile["namespace_status"] == "provisioned":
            try:
                _validate_provisioning_evidence(
                    name, profile, accounts, evidence_root or Path.cwd()
                )
            except (TypeError, ValueError, OSError) as exc:
                errors.append(
                    f"{name}: provisioning evidence invalid ({type(exc).__name__})"
                )
        for role in ROOT_ROLES:
            uri = profile.get(f"{role}_root_uri")
            if not _gcs(uri, bucket):
                errors.append(
                    f"{name}.{role}_root_uri must be a normalized gs:// URI in the allowed bucket"
                )
                continue
            roots.append((f"{name}.{role}", uri))
            if any(_overlap(uri, current.rsplit("/", 1)[0]) for current in protected):
                errors.append(
                    f"{name}.{role} overlaps the existing operational registry"
                )
        registry = profile.get("registry_root_uri")
        if profile.get("current_pointer_uri") != f"{registry}/CURRENT.json":
            errors.append(f"{name}.current_pointer_uri must belong to its own registry")
        db = profile.get("sqlite_path")
        if (
            not isinstance(db, str)
            or not db.startswith(f"/app/var/{name}/")
            or ".." in Path(db).parts
        ):
            errors.append(f"{name}.sqlite_path must be under /app/var/{name}/")
        else:
            databases.append(db)
        for key in (
            "gate_rules_uri",
            "baseline_metrics_uri",
            "approved_baseline_uri",
            "pipeline_template_uri",
        ):
            uri = profile.get(key)
            if uri is None:
                if name == target_profile:
                    pending.append(
                        f"{name}: pin {key} and its immutable evidence before release."
                    )
            elif not _gcs(uri, bucket) or any(
                word in uri.lower()
                for word in ("smoke", "replace", "baseline_metrics_empty")
            ):
                errors.append(
                    f"{name}.{key} contains a placeholder, smoke gate, empty baseline, or invalid URI"
                )
            else:
                parent_role = (
                    "registry"
                    if key == "approved_baseline_uri"
                    else "pipelines"
                    if key == "pipeline_template_uri"
                    else "config"
                )
                if not uri.startswith(
                    str(profile.get(f"{parent_role}_root_uri")) + "/"
                ):
                    errors.append(
                        f"{name}.{key} must belong to its own {parent_role} root"
                    )
    for index, (left_name, left) in enumerate(roots):
        for right_name, right in roots[index + 1 :]:
            if _overlap(left, right):
                errors.append(f"Overlapping namespaces: {left_name} and {right_name}")
    if len(set(databases)) != len(databases):
        errors.append("R0 and R1 must not share SQLite")
    images = config.get("images", {})
    if not isinstance(images, dict):
        errors.append("images must be an object")
        images = {}
    prefix = f"{config.get('region')}-docker.pkg.dev/{project}/{config.get('artifact_repository')}/"
    for role in IMAGE_ROLES:
        value = images.get(role)
        if value is None:
            pending.append(
                f"Pin {role} image digest, then compile and hash matching IR."
            )
        elif (
            not isinstance(value, str)
            or not value.startswith(prefix)
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value)
        ):
            errors.append(
                f"images.{role} must be an immutable digest in the configured Artifact Registry repository"
            )
    return errors, pending


def cloud_commands(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    project, region = config["project"], config["region"]
    commands = [
        ("apis", ["services", "list", "--enabled"]),
        (
            "vm",
            [
                "compute",
                "instances",
                "describe",
                config["serving_instance"],
                f"--zone={config['serving_zone']}",
            ],
        ),
        ("bucket", ["storage", "buckets", "describe", f"gs://{config['bucket']}"]),
        (
            "bucket_iam",
            ["storage", "buckets", "get-iam-policy", f"gs://{config['bucket']}"],
        ),
        ("project_iam", ["projects", "get-iam-policy", project]),
        (
            "repository",
            [
                "artifacts",
                "repositories",
                "describe",
                config["artifact_repository"],
                f"--location={region}",
            ],
        ),
        (
            "repository_iam",
            [
                "artifacts",
                "repositories",
                "get-iam-policy",
                config["artifact_repository"],
                f"--location={region}",
            ],
        ),
    ]
    for role in ("pipeline", "training"):
        account = config["service_accounts"][role]
        commands.extend(
            [
                (f"{role}_account", ["iam", "service-accounts", "describe", account]),
                (f"{role}_iam", ["iam", "service-accounts", "get-iam-policy", account]),
            ]
        )
    return [
        (
            name,
            [
                "gcloud",
                *args,
                f"--project={project}",
                "--quiet",
                "--format=json(name,status,machineType,serviceAccounts)"
                if name == "vm"
                else "--format=json",
            ],
        )
        for name, args in commands
    ]


def _sanitize(name: str, value: Any) -> Any:
    """Allowlist inventory fields; VM metadata and command stderr never escape."""
    if name == "apis":
        if not isinstance(value, list):
            raise ValueError("Expected an API list")
        return sorted(item.get("config", {}).get("name", "") for item in value)
    if not isinstance(value, dict):
        raise TypeError("Expected an object")
    if name.endswith("_iam"):
        return {
            "bindings": [
                {
                    key: binding[key]
                    for key in ("role", "members", "condition")
                    if key in binding
                }
                for binding in value.get("bindings", [])
            ],
            "version": value.get("version", 1),
        }
    fields = {
        "vm": ("name", "status", "machineType", "serviceAccounts"),
        "bucket": (
            "name",
            "location",
            "storage_class",
            "uniform_bucket_level_access",
            "public_access_prevention",
            "lifecycle_config",
            "soft_delete_policy",
            "versioning_enabled",
            "retention_policy",
            "iamConfiguration",
            "lifecycle",
            "versioning",
            "softDeletePolicy",
        ),
        "repository": ("name", "format", "mode", "dockerConfig"),
    }.get(name, ("email", "disabled", "uniqueId"))
    return {key: value[key] for key in fields if key in value}


def _run(command: list[str]) -> Any:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("gcloud unavailable or timed out") from exc
    if result.returncode:
        message = result.stderr.lower()
        category = (
            "reauthentication_required"
            if any(
                word in message
                for word in ("reauth", "invalid_grant", "login", "credentials")
            )
            else "permission_or_command_failure"
        )
        raise RuntimeError(
            f"{category} (exit {result.returncode}); raw output withheld"
        )
    return json.loads(result.stdout)


def audit(
    config: dict[str, Any],
    *,
    cloud: bool = False,
    runner: Callable[[list[str]], Any] = _run,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    errors, pending = validate_config(config, evidence_root=evidence_root)
    report: dict[str, Any] = {
        "schema_version": "router-cloud-foundation-audit-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if errors else "local_verified",
        "local_errors": errors,
        "cloud_errors": [],
        "inventory": {},
        "cloud_inventory_verified": False,
        "iam_effective_access_verified": False,
        "deployment_ready": False,
        "release_inputs_pending": pending,
        "predeploy_checks_pending": IAM_CHECKS
        + [
            "Confirm billing credit/budget, GPU quota and bounded job concurrency before runs.",
            "Review lifecycle retention against current/previous/anchor and durable evidence.",
            "Bind actual image digests, IR/config/data hashes and objective-specific gate evidence.",
        ],
        "postdeploy_checks_pending": POST_DEPLOY_CHECKS,
        "mutations_performed": False,
    }
    if not cloud or errors:
        return report
    for name, command in cloud_commands(config):
        try:
            report["inventory"][name] = _sanitize(name, runner(command))
        except Exception as exc:  # noqa: BLE001 -- never expose raw provider errors
            # Unknown runner errors may contain credentials; expose only our known categories.
            category = (
                str(exc)
                if type(exc) is RuntimeError
                and str(exc).startswith(
                    (
                        "gcloud unavailable",
                        "reauthentication_required",
                        "permission_or_command_failure",
                    )
                )
                else "invalid or unavailable inventory; raw output withheld"
            )
            report["cloud_errors"].append(f"{name}: {category}")
    inv = report["inventory"]
    for name, fields in {
        "vm": ("name", "status", "serviceAccounts"),
        "bucket": ("name", "location"),
        "repository": ("name", "format"),
        "pipeline_account": ("email",),
        "training_account": ("email",),
    }.items():
        if name in inv and any(not inv[name].get(field) for field in fields):
            report["cloud_errors"].append(f"{name}: incomplete inventory response")
    for name, expected in (
        ("vm", config["serving_instance"]),
        ("bucket", config["bucket"]),
        ("repository", config["artifact_repository"]),
    ):
        actual = inv.get(name, {}).get("name")
        if actual and str(actual).rstrip("/").rsplit("/", 1)[-1] != expected:
            report["cloud_errors"].append(
                f"{name}: resource name differs from the configuration"
            )
    if "apis" in inv:
        for api in sorted(set(config["required_apis"]) - set(inv["apis"])):
            report["cloud_errors"].append(f"Required API is not enabled: {api}")
    vm_accounts = [
        account.get("email") for account in inv.get("vm", {}).get("serviceAccounts", [])
    ]
    expected_serving = config["service_accounts"].get("serving")
    if "vm" in inv and (
        not vm_accounts or expected_serving and expected_serving not in vm_accounts
    ):
        report["cloud_errors"].append(
            "VM service account is missing or differs from the pinned account"
        )
    if inv.get("repository", {}).get("format") not in (None, "DOCKER"):
        report["cloud_errors"].append(
            "Artifact Registry repository must use DOCKER format"
        )
    for role in ("pipeline", "training"):
        account = inv.get(f"{role}_account", {})
        if account and (
            account.get("disabled") is True
            or account.get("email") != config["service_accounts"][role]
        ):
            report["cloud_errors"].append(
                f"{role} service account is disabled or does not match"
            )
    if not report["cloud_errors"]:
        report["status"] = "cloud_verified"
        report["cloud_inventory_verified"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("ml/config/cloud-foundation-v1.json")
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Read GCP inventory; never modify resources or submit jobs",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path.cwd(),
        help="Root for relative provisioning evidence paths (default: working directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a new local audit file; existing files are never overwritten",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        parser.error("config must be a JSON object")
    report = audit(config, cloud=args.cloud, evidence_root=args.evidence_root)
    report["config_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(text)
    print(text, end="")
    raise SystemExit(1 if report["local_errors"] or report["cloud_errors"] else 0)
