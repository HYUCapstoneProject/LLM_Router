#!/usr/bin/env python3
"""Train G1 arms with mounted Drive, drive.file uploads, or verified host mirroring."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from llm_router_labeling.generalization.training import (
    ARMS,
    _verify_completed,
    run_four_arms,
)
from llm_router_labeling.shared.dataset_io import sha256_file, write_json
from llm_router_labeling.training.modernbert_checkpointing import verify_checkpoint

CHUNK_SIZE = 16 * 1024 * 1024


class BackupError(RuntimeError):
    """Sanitized transfer failure: never contains a capability URL."""


def upload_file(path, slot, session=None):
    """Resume a Drive upload and verify its final object identity, size and checksum."""
    import requests

    try:
        url = slot["url"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "www.googleapis.com":
            raise BackupError("Unapproved upload host")
        total = path.stat().st_size
        with path.open("rb") as handle:
            md5 = hashlib.file_digest(handle, "md5").hexdigest()
        digest = sha256_file(path)
        client = session or requests.Session()
        response = client.put(
            url,
            data=b"",
            headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"},
            timeout=60,
        )
        with path.open("rb") as handle:
            while response.status_code == 308:
                received = response.headers.get("Range")
                if received is None:
                    offset = 0
                else:
                    match = re.fullmatch(r"bytes=0-(\d+)", received)
                    if not match:
                        raise BackupError("Invalid upload acknowledgement")
                    offset = int(match.group(1)) + 1
                if offset >= total:
                    raise BackupError("Upload acknowledged bytes without completion")
                handle.seek(offset)
                chunk = handle.read(CHUNK_SIZE)
                end = offset + len(chunk) - 1
                response = client.put(
                    url,
                    data=chunk,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{total}",
                    },
                    timeout=180,
                )
                if (
                    response.status_code == 308
                    and response.headers.get("Range") != f"bytes=0-{end}"
                ):
                    raise BackupError("Upload did not acknowledge the sent chunk")
        if response.status_code not in (200, 201):
            raise BackupError(f"Upload HTTP {response.status_code}")
        result = response.json()
        if result.get("id") != slot["id"] or int(result.get("size", -1)) != total:
            raise BackupError("Uploaded object identity or size mismatch")
        if result.get("sha256Checksum"):
            valid = result["sha256Checksum"] == digest
        else:
            valid = result.get("md5Checksum") == md5
        if not valid:
            raise BackupError("Uploaded object checksum mismatch")
        return {
            "file_id": result["id"],
            "size_bytes": total,
            "sha256": digest,
            "md5": md5,
            "verified": True,
        }
    except BackupError:
        raise
    except Exception:  # noqa: BLE001 - network exceptions may contain secret upload URLs.
        raise BackupError(
            "Drive upload failed; credentials and URLs withheld"
        ) from None


def make_archive(destination, files, extra_json=None):
    """Deterministic file-only tar with explicit restore paths and no secret slots."""
    with tarfile.open(destination, "w") as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or not path.is_file():
                raise BackupError("Archive source is not a regular file")
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
        for name, payload in sorted((extra_json or {}).items()):
            data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))


class DriveBackup:
    def __init__(self, root, output, arms, slots_path, *, expected_account=None):
        if not isinstance(expected_account, str) or not expected_account.strip():
            raise BackupError("An explicit backup account is required")
        self.root, self.output, self.arms = root, output, arms
        # Keep the capability file outside every archive and output directory.
        if slots_path.resolve().is_relative_to(output.resolve()):
            raise BackupError("Backup slots must be outside the output directory")
        payload = json.loads(slots_path.read_text())
        if payload.get("account") != expected_account:
            raise BackupError("Unexpected backup account")
        self.slots = payload["slots"]
        self.receipt_path = output / "backup-receipts.json"
        self.receipts = (
            json.loads(self.receipt_path.read_text())
            if self.receipt_path.exists()
            else {}
        )
        self.transfer_dir = output / ".transfer"
        self.transfer_dir.mkdir(parents=True, exist_ok=True)

    def preserve(self, key, files, extra_json=None):
        if key in self.receipts:
            return
        if key not in self.slots:
            raise BackupError("Required backup slot is absent")
        archive = self.transfer_dir / (key.replace(":", "-") + ".tar")
        make_archive(archive, files, extra_json)
        receipt = self.transfer(key, archive)
        self.receipts[key] = receipt
        write_json(self.receipt_path, self.receipts)
        archive.unlink()
        print(json.dumps({"backup_key": key, **receipt}), flush=True)

    def transfer(self, key, archive):
        return upload_file(archive, self.slots[key])

    def base_files(self):
        return {
            "run-plan.json": self.output / "run-plan.json",
            "input-manifest.json": self.root / "input-manifest.json",
        }

    def completed_arms(self):
        for arm in self.arms:
            directory = self.output / arm
            if not (directory / "_SUCCESS").exists():
                continue
            manifest = _verify_completed(
                directory, sha256_file(self.output / "run-plan.json")
            )
            files = self.base_files()
            # Includes training.parquet because the sealed arm manifest covers it.
            for name in [*manifest["artifacts"], "manifest.json", "_SUCCESS"]:
                files[f"{arm}/{name}"] = directory / name
            self.preserve(
                f"{arm}:final",
                files,
                {
                    "execution-status.json": {
                        "status": "arm_complete",
                        "arm": arm,
                        "optimizer_steps": manifest["optimizer_steps"],
                        "comparison_complete": False,
                        "production_activated": False,
                    }
                },
            )

    def checkpoint(self, arm, path):
        metadata = verify_checkpoint(path)["metadata"]
        step = int(metadata["optimizer_steps"])
        if path.name != f"step-{step:08d}" or path.parent.parent.name != arm:
            raise BackupError("Checkpoint path and metadata differ")
        self.completed_arms()
        files = self.base_files()
        for file in path.rglob("*"):
            if file.is_file():
                files[
                    f"{arm}/checkpoints/{path.name}/{file.relative_to(path).as_posix()}"
                ] = file
        self.preserve(f"{arm}:{step}", files)


class LocalMirrorBackup(DriveBackup):
    """Block until the host confirms the archive is durably mirrored and verified."""

    def __init__(self, root, output, arms, mirror_dir, timeout_seconds=900):
        self.root, self.output, self.arms = root, output, arms
        self.mirror_dir = mirror_dir.resolve()
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.receipt_path = output / "backup-receipts.json"
        self.receipts = (
            json.loads(self.receipt_path.read_text())
            if self.receipt_path.exists()
            else {}
        )
        self.transfer_dir = output / ".transfer"
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        # preserve's whitelist check accepts every valid selected-arm transfer key.
        self.slots = _MirrorKeys(arms)

    def transfer(self, key, archive):
        safe_key = key.replace(":", "-")
        with archive.open("rb") as handle:
            md5 = hashlib.file_digest(handle, "md5").hexdigest()
        request = {
            "key": key,
            "archive_path": str(archive.resolve()),
            "sha256": sha256_file(archive),
            "md5": md5,
            "size_bytes": archive.stat().st_size,
        }
        request_path = self.mirror_dir / f"request-{safe_key}.json"
        temporary = request_path.with_suffix(".json.partial")
        write_json(temporary, request)
        temporary.replace(request_path)
        acknowledgement = self.mirror_dir / f"ack-{safe_key}.json"
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if acknowledgement.exists():
                try:
                    ack = json.loads(acknowledgement.read_text())
                except (ValueError, OSError):
                    raise BackupError("Invalid host mirror acknowledgement") from None
                if ack.get("status") in ("error", "failed") or ack.get("error"):
                    raise BackupError("Host mirror reported a failure")
                if any(
                    ack.get(field) != request[field]
                    for field in ("key", "sha256", "size_bytes")
                ):
                    raise BackupError("Host mirror acknowledgement checksum mismatch")
                if "md5" in ack and ack["md5"] != md5:
                    raise BackupError("Host mirror acknowledgement MD5 mismatch")
                return {
                    "backend": "host-local-mirror",
                    "verified": True,
                    "sha256": request["sha256"],
                    "md5": md5,
                    "size_bytes": request["size_bytes"],
                }
            if time.monotonic() >= deadline:
                raise BackupError("Host mirror acknowledgement timed out")
            time.sleep(2)


class _MirrorKeys:
    def __init__(self, arms):
        self.arms = set(arms)

    def __contains__(self, key):
        parts = key.split(":")
        return (
            len(parts) == 2
            and parts[0] in self.arms
            and (parts[1] == "final" or parts[1].isdigit())
        )


class MountedDriveBackup:
    """Checkpoint writes already target the user-approved mounted Drive."""

    def __init__(self, root, output, arms, drive_root):
        self.output, self.arms = output.resolve(), arms
        drive = drive_root.resolve()
        if not drive.is_dir() or not self.output.is_relative_to(drive):
            raise BackupError("Output must be inside the approved mounted Drive root")
        probe = self.output / "storage-probe.txt"
        expected = str(time.time_ns())
        with probe.open("w") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_text() != expected:
            raise BackupError("Mounted Drive write/read probe failed")

    def completed_arms(self):
        for arm in self.arms:
            directory = self.output / arm
            if (directory / "_SUCCESS").exists():
                _verify_completed(directory, sha256_file(self.output / "run-plan.json"))

    def checkpoint(self, arm, path):
        if not path.resolve().is_relative_to(self.output / arm / "checkpoints"):
            raise BackupError("Checkpoint is outside the mounted arm directory")
        metadata = verify_checkpoint(path)["metadata"]
        if path.name != f"step-{int(metadata['optimizer_steps']):08d}":
            raise BackupError("Checkpoint path and metadata differ")
        self.completed_arms()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", choices=list(ARMS), nargs="+", required=True)
    backup_args = parser.add_mutually_exclusive_group(required=True)
    backup_args.add_argument("--backup-slots", type=Path)
    backup_args.add_argument("--mirror-dir", type=Path)
    backup_args.add_argument("--drive-root", type=Path)
    parser.add_argument(
        "--backup-account",
        help="Expected account in backup slots; required for --backup-slots",
    )
    args = parser.parse_args()
    if args.backup_slots and not (args.backup_account and args.backup_account.strip()):
        parser.error("--backup-slots requires --backup-account")
    if args.backup_account is not None and not args.backup_slots:
        parser.error("--backup-account applies only to --backup-slots")
    root, out = args.root.resolve(), args.output_dir.resolve()
    manifest = json.loads((root / "input-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError("Frozen bundle checksum mismatch")
    if (root / "data/dataset/task_holdout.parquet").exists():
        raise ValueError("Holdout must not be delivered to training sessions")
    out.mkdir(parents=True, exist_ok=True)
    if args.backup_slots:
        backup = DriveBackup(
            root,
            out,
            args.arms,
            args.backup_slots,
            expected_account=args.backup_account,
        )
    elif args.mirror_dir:
        backup = LocalMirrorBackup(root, out, args.arms, args.mirror_dir)
    else:
        backup = MountedDriveBackup(root, out, args.arms, args.drive_root)
    status_path = out / "execution-status.json"
    status = {
        "started_at_unix": time.time(),
        "status": "running",
        "arms": args.arms,
        "input_manifest_sha256": sha256_file(root / "input-manifest.json"),
        "task_holdout_accessed": False,
        "production_activated": False,
    }
    write_json(status_path, status)

    def checkpoint_saved(arm, path):
        backup.checkpoint(arm, path)  # A failed backup aborts training immediately.
        status.update(
            last_checkpoint_arm=arm,
            last_checkpoint=str(path),
            updated_at_unix=time.time(),
        )
        write_json(status_path, status)
        print(json.dumps(status), flush=True)

    try:
        result = run_four_arms(
            dataset_dir=root / "data/dataset",
            config_path=root / "ml/config/routing_generalization_v1.json",
            runtime_config_path=root / "ml/config/modernbert_runtime_cu124_v1.json",
            output_dir=out,
            arms_to_run=args.arms,
            on_checkpoint_saved=checkpoint_saved,
        )
        backup.completed_arms()
        status.update(
            status="assigned_arms_complete",
            finished_at_unix=time.time(),
            comparison_complete=result.get("comparison_complete", False),
        )
    except BaseException as error:  # noqa: BLE001 - sanitize credentials even on interruption.
        status.update(
            status="failed",
            error_type=type(error).__name__,
            finished_at_unix=time.time(),
        )
        raise BackupError("Run stopped; inspect sanitized execution status") from None
    finally:
        write_json(status_path, status)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
