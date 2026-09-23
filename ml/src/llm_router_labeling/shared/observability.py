from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .object_store import ObjectStore, ObjectStoreError, build_object_store


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_observability_json(
    uri: str,
    payload: Mapping[str, Any],
    *,
    mutable: bool = False,
    object_store: ObjectStore | None = None,
) -> None:
    """Publish a privacy-safe stage snapshot to its deterministic run URI."""
    store = object_store or build_object_store(uri)
    value = {"schema_version": "retraining-observability-v1", **dict(payload)}
    value.setdefault("updated_at", utc_now())
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    existing = store.read_text(uri)
    if not mutable and existing is not None:
        try:
            existing_value = json.loads(existing)
        except json.JSONDecodeError as exc:
            raise ObjectStoreError(
                f"Immutable observability object is invalid JSON: {uri}"
            ) from exc
        comparable = dict(value)
        comparable.pop("updated_at", None)
        if any(existing_value.get(key) != item for key, item in comparable.items()):
            raise ObjectStoreError(f"Immutable observability object already exists: {uri}")
        return
    generation = store.generation(uri) if mutable else None
    store.write_text(uri, text, if_generation_match=generation)
