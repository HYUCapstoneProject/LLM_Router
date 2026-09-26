from pathlib import Path

import pytest
from llm_router_labeling.shared.object_store import GcsObjectStore


class FakeBlob:
    def __init__(self, error=None):
        self.chunk_size = None
        self.calls = []
        self.error = error

    def upload_from_filename(self, filename, *, timeout):
        self.calls.append((filename, self.chunk_size, timeout))
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("size", [3, 598 * 1024 * 1024])
def test_upload_configures_bounded_chunks_and_timeout(tmp_path, monkeypatch, size):
    source = tmp_path / "model.safetensors"
    # Sparse file: test configuration without allocating a model-sized buffer.
    with source.open("wb") as handle:
        handle.truncate(size)
    blob = FakeBlob()
    store = GcsObjectStore(project="fixture")
    seen = []

    def resolve(uri):
        seen.append(uri)
        return blob

    monkeypatch.setattr(store, "_blob", resolve)
    destination = "gs://fixture/models/example/candidate/model.safetensors"
    store.upload_file(source, destination)
    assert seen == [destination]
    assert blob.calls == [(str(source), 8 * 1024 * 1024, (20, 120))]
    assert blob.chunk_size % (256 * 1024) == 0


def test_upload_propagates_failure_for_safe_publisher_retry(monkeypatch):
    failure = TimeoutError("synthetic write timeout")
    blob = FakeBlob(failure)
    store = GcsObjectStore(project="fixture")
    monkeypatch.setattr(store, "_blob", lambda uri: blob)
    with pytest.raises(TimeoutError, match="synthetic write timeout"):
        store.upload_file(Path("weights"), "gs://fixture/model.safetensors")
