"""Minimal object-store abstraction for publishing model packages.

The training and evaluation code paths deliberately stay on local paths because
Vertex AI mounts pipeline artifacts through GCS FUSE. Publishing an approved
model is the one place that must address immutable ``gs://`` locations directly,
because the serving deployment reads them by URI and never sees the pipeline's
mounted view.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Protocol
from urllib.parse import urlparse


class ObjectStoreError(RuntimeError):
    """Raised when an object-store operation violates the publishing contract."""


@dataclass(frozen=True)
class GcsUri:
    bucket: str
    prefix: str

    def child(self, *parts: str) -> "GcsUri":
        joined = PurePosixPath(self.prefix or "").joinpath(*parts)
        return GcsUri(bucket=self.bucket, prefix=str(joined).strip("/"))

    def __str__(self) -> str:
        return (
            f"gs://{self.bucket}/{self.prefix}"
            if self.prefix
            else f"gs://{self.bucket}"
        )


def parse_gcs_uri(uri: str) -> GcsUri:
    parsed = urlparse(str(uri))
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ObjectStoreError(f"Not a GCS URI: {uri}")
    return GcsUri(bucket=parsed.netloc, prefix=parsed.path.strip("/"))


class ObjectStore(Protocol):
    def upload_file(self, source: Path, destination_uri: str) -> None: ...

    def download_file(self, source_uri: str, destination: Path) -> None: ...

    def read_text(self, uri: str) -> str | None: ...

    def write_text(
        self, uri: str, text: str, *, if_generation_match: int | None = None
    ) -> None: ...

    def generation(self, uri: str) -> int | None: ...

    def list_uris(self, prefix_uri: str) -> list[str]: ...

    def delete_uri(self, uri: str) -> None: ...


class LocalObjectStore:
    """Filesystem-backed store used by tests and by local dry runs."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, uri: str) -> Path:
        parsed = parse_gcs_uri(uri)
        return self._root / parsed.bucket / parsed.prefix

    def upload_file(self, source: Path, destination_uri: str) -> None:
        target = self._path(destination_uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def download_file(self, source_uri: str, destination: Path) -> None:
        source = self._path(source_uri)
        if not source.is_file():
            raise ObjectStoreError(f"Object does not exist: {source_uri}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def read_text(self, uri: str) -> str | None:
        path = self._path(uri)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write_text(
        self, uri: str, text: str, *, if_generation_match: int | None = None
    ) -> None:
        path = self._path(uri)
        current = self.generation(uri)
        if current != if_generation_match:
            raise ObjectStoreError(
                f"Generation precondition failed for {uri}: "
                f"expected {if_generation_match}, found {current}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def generation(self, uri: str) -> int | None:
        path = self._path(uri)
        if not path.is_file():
            return None
        return int(path.stat().st_mtime_ns)

    def list_uris(self, prefix_uri: str) -> list[str]:
        parsed = parse_gcs_uri(prefix_uri)
        base = self._path(prefix_uri)
        if not base.is_dir():
            return []
        return sorted(
            f"gs://{parsed.bucket}/{PurePosixPath(parsed.prefix) / path.relative_to(base)}"
            for path in base.rglob("*")
            if path.is_file()
        )

    def delete_uri(self, uri: str) -> None:
        path = self._path(uri)
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            raise ObjectStoreError(f"Failed to delete {uri}: {exc}") from exc


class GcsObjectStore:
    """Google Cloud Storage store used by the pipeline and approval CLI."""

    def __init__(
        self, *, project: str | None = None, client: object | None = None
    ) -> None:
        self._project = project
        self._client = client

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._project)
        return self._client

    def _blob(self, uri: str):
        parsed = parse_gcs_uri(uri)
        if not parsed.prefix:
            raise ObjectStoreError(f"GCS URI has no object name: {uri}")
        return self._get_client().bucket(parsed.bucket).blob(parsed.prefix)

    def upload_file(self, source: Path, destination_uri: str) -> None:
        blob = self._blob(destination_uri)
        # Bound each resumable transfer chunk on slower developer uplinks. This
        # does not change immutable package identity or CURRENT generation CAS.
        blob.chunk_size = 8 * 1024 * 1024
        blob.upload_from_filename(str(source), timeout=(20, 120))

    def download_file(self, source_uri: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = self._blob(source_uri)
        if not blob.exists():
            raise ObjectStoreError(f"Object does not exist: {source_uri}")
        blob.download_to_filename(str(destination))

    def read_text(self, uri: str) -> str | None:
        blob = self._blob(uri)
        if not blob.exists():
            return None
        return blob.download_as_bytes().decode("utf-8")

    def write_text(
        self, uri: str, text: str, *, if_generation_match: int | None = None
    ) -> None:
        from google.api_core.exceptions import PreconditionFailed

        blob = self._blob(uri)
        try:
            blob.upload_from_string(
                text.encode("utf-8"),
                content_type="application/json",
                # Generation 0 is GCS shorthand for "only if the object does not
                # exist yet", which is what a None precondition means here.
                if_generation_match=0
                if if_generation_match is None
                else if_generation_match,
            )
        except PreconditionFailed as exc:
            raise ObjectStoreError(
                f"Concurrent update detected while writing {uri}"
            ) from exc

    def generation(self, uri: str) -> int | None:
        parsed = parse_gcs_uri(uri)
        blob = self._get_client().bucket(parsed.bucket).get_blob(parsed.prefix)
        return int(blob.generation) if blob is not None else None

    def list_uris(self, prefix_uri: str) -> list[str]:
        parsed = parse_gcs_uri(prefix_uri)
        prefix = parsed.prefix.rstrip("/") + "/" if parsed.prefix else ""
        blobs: Iterable = self._get_client().list_blobs(parsed.bucket, prefix=prefix)
        return sorted(f"gs://{parsed.bucket}/{blob.name}" for blob in blobs)

    def delete_uri(self, uri: str) -> None:
        from google.api_core.exceptions import GoogleAPICallError

        try:
            self._blob(uri).delete()
        except GoogleAPICallError as exc:
            raise ObjectStoreError(f"Failed to delete {uri}: {exc}") from exc


def build_object_store(uri: str, *, project: str | None = None) -> ObjectStore:
    """Return a GCS store, or a local store when the URI points at a directory."""
    parsed = urlparse(str(uri))
    if parsed.scheme == "gs":
        return GcsObjectStore(project=project)
    raise ObjectStoreError(
        f"Model publishing requires a gs:// URI; got {uri!r}. "
        "Use LocalObjectStore explicitly for local dry runs."
    )
