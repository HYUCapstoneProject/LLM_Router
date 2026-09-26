"""Check local release images; publish only with --publish.

Manifest: {"schema_version":"release-images-v1", "images":[
 {"local_ref":"name:tag", "expected_image_id":"sha256:<64 hex>",
  "remote_ref":"us-west1-docker.pkg.dev/project-763753da-fa8f-4afd-b31/llm-router/name:unique-tag"}]}
Uses an isolated Docker config with the gcloud credential helper. No credentials
are copied, printed or included in the receipt. A failed publish retains a partial
receipt; use its verified immutable references and inspect the failure before retrying.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REGISTRY = "us-west1-docker.pkg.dev"
PREFIX = f"{REGISTRY}/project-763753da-fa8f-4afd-b31/llm-router/"
DOCKER_HOST = "unix://" + str(Path.home() / ".orbstack/run/docker.sock")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def load_manifest(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "images"}:
        raise ValueError("manifest requires schema_version and images")
    images = payload["images"]
    if (
        payload["schema_version"] != "release-images-v1"
        or not isinstance(images, list)
        or not images
    ):
        raise ValueError("invalid release-images-v1 manifest")
    for item in images:
        if not isinstance(item, dict) or set(item) != {
            "local_ref",
            "expected_image_id",
            "remote_ref",
        }:
            raise ValueError(
                "image requires local_ref, expected_image_id and remote_ref"
            )
        if any(not isinstance(value, str) or not value for value in item.values()):
            raise ValueError("image fields must be nonempty strings")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", item["local_ref"]):
            raise ValueError("invalid local image reference")
        if not DIGEST.fullmatch(item["expected_image_id"]):
            raise ValueError("expected_image_id must be a full sha256 image ID")
        remote = item["remote_ref"]
        if (
            not remote.startswith(PREFIX)
            or not re.fullmatch(
                r"[a-z0-9][a-z0-9._-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}",
                remote[len(PREFIX) :],
            )
            or remote.rsplit(":", 1)[1] in {"latest", "main"}
        ):
            raise ValueError(
                "remote_ref must use the release registry and a unique version tag"
            )
    for field in ("local_ref", "remote_ref"):
        if len({item[field] for item in images}) != len(images):
            raise ValueError(f"duplicate {field}")
    return images


def docker(config: str, *args: str) -> str:
    result = subprocess.run(
        ["docker", "--config", config, "--host", DOCKER_HOST, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        # Credential helper output can contain credentials or account details.
        raise RuntimeError(
            f"docker {args[0]} failed (exit {result.returncode}); inspect Docker/GCP access"
        )
    return result.stdout


def inspect(config: str, reference: str) -> dict:
    result = json.loads(docker(config, "image", "inspect", reference))
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError("unexpected docker image inspect response")
    return result[0]


def verify_local(config: str, item: dict[str, str]) -> None:
    result = inspect(config, item["local_ref"])
    if result.get("Id") != item["expected_image_id"]:
        raise RuntimeError(f"local image ID changed: {item['local_ref']}")
    if (result.get("Os"), result.get("Architecture")) != ("linux", "amd64"):
        raise RuntimeError(f"image is not linux/amd64: {item['local_ref']}")


def publish_one(config: str, item: dict[str, str]) -> str:
    # Recheck immediately before each mutation, including after earlier pushes.
    verify_local(config, item)
    docker(config, "tag", item["expected_image_id"], item["remote_ref"])
    output = docker(config, "push", item["remote_ref"])
    digests = set(re.findall(r"\bdigest: (sha256:[0-9a-f]{64})\b", output))
    if len(digests) != 1:
        raise RuntimeError("push did not return exactly one registry manifest digest")
    repository = item["remote_ref"].rsplit(":", 1)[0]
    immutable = repository + "@" + digests.pop()
    result = inspect(config, item["remote_ref"])
    if result.get("Id") != item["expected_image_id"] or immutable not in result.get(
        "RepoDigests", []
    ):
        raise RuntimeError(
            "pushed digest does not match the local image and repository receipt"
        )
    return immutable


def save_receipt(path: Path, receipt: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--receipt", type=Path, help="New output path; required for --publish"
    )
    parser.add_argument(
        "--publish", action="store_true", help="Tag and upload every verified image"
    )
    args = parser.parse_args()
    receipt = None
    try:
        images = load_manifest(args.manifest)
        if args.publish and (args.receipt is None or args.receipt.exists()):
            raise ValueError(
                "--publish requires a new --receipt path; existing receipts are preserved"
            )
        with tempfile.TemporaryDirectory(prefix="router-release-docker-") as config:
            Path(config, "config.json").write_text(
                json.dumps({"credHelpers": {REGISTRY: "gcloud"}})
            )
            for item in images:
                verify_local(config, item)
            if not args.publish:
                print(
                    json.dumps(
                        {
                            "status": "local_checked",
                            "image_count": len(images),
                            "published": False,
                        }
                    )
                )
                return 0
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            # Reserve before the first mutation, so interruption cannot hide an attempted publish.
            with args.receipt.open("x"):
                pass
            receipt = {
                "schema_version": "release-image-receipt-v1",
                "status": "publishing",
                "started_at": datetime.now(UTC).isoformat(),
                "images": [],
            }
            save_receipt(args.receipt, receipt)
            for item in images:
                print(f"Publishing {item['remote_ref']}", flush=True)
                entry = {**item, "status": "publishing"}
                receipt["images"].append(entry)
                save_receipt(args.receipt, receipt)
                entry.update(immutable_ref=publish_one(config, item), status="verified")
                save_receipt(args.receipt, receipt)
            receipt.update(
                status="published", completed_at=datetime.now(UTC).isoformat()
            )
            save_receipt(args.receipt, receipt)
            print(
                json.dumps(
                    {
                        "status": "published",
                        "receipt": str(args.receipt),
                        "image_count": len(images),
                    }
                )
            )
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        if receipt is not None:
            receipt["status"] = "failed"
            save_receipt(args.receipt, receipt)
        # Validation errors contain only field names/references, never tool stderr.
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
