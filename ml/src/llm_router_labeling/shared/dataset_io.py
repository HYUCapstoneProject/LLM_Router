from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Mapping


class DatasetIOError(RuntimeError):
    """Raised when a source or artifact violates the shared I/O contract."""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, source_config: Mapping[str, Any]) -> None:
    expected_size = int(source_config["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DatasetIOError(
            f"Unexpected size for {path}: expected {expected_size}, got {actual_size}"
        )
    expected_hash = str(source_config["sha256"])
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise DatasetIOError(
            f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
        )


def download_source(
    source_config: Mapping[str, Any], destination_dir: Path, timeout: int = 120
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / str(source_config["filename"])
    if destination.exists():
        verify_file(destination, source_config)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    expected_size = int(source_config["size_bytes"])
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > expected_size:
        partial.unlink()
        partial_size = 0
    headers = {"User-Agent": "llm-router-dataset-tools/0.1"}
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
    request = urllib.request.Request(str(source_config["url"]), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_status = getattr(response, "status", response.getcode())
        resumed = bool(partial_size and response_status == 206)
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_size = partial.stat().st_size
    if actual_size < expected_size:
        raise DatasetIOError(
            f"Incomplete download for {partial}: expected {expected_size}, got "
            f"{actual_size}; re-run to resume"
        )
    try:
        verify_file(partial, source_config)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
