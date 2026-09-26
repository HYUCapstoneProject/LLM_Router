"""No registry access: verify fail-closed local checks and immutable push receipts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/publish-release-images.py"
spec = importlib.util.spec_from_file_location("publish_release_images", SCRIPT)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
IMAGE_ID = "sha256:" + "a" * 64
REMOTE = publisher.PREFIX + "serving:r0-20260927-local"
DIGEST = "sha256:" + "b" * 64


def item():
    return {
        "local_ref": "serving:local",
        "expected_image_id": IMAGE_ID,
        "remote_ref": REMOTE,
    }


def fake_process(
    monkeypatch, *, arch="amd64", actual_id=IMAGE_ID, wrong_repo=False, fail_push=False
):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[:2] == ["docker", "--config"]
        assert command[3:5] == ["--host", publisher.DOCKER_HOST]
        operation = command[5:]
        if operation[:2] == ["image", "inspect"]:
            repository = "wrong/repository" if wrong_repo else REMOTE.rsplit(":", 1)[0]
            output = json.dumps(
                [
                    {
                        "Id": actual_id,
                        "Os": "linux",
                        "Architecture": arch,
                        "RepoDigests": [repository + "@" + DIGEST],
                    }
                ]
            )
        elif operation[0] == "push":
            if fail_push:
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="credential-secret-must-not-leak"
                )
            output = "tag: digest: " + DIGEST + " size: 1234\n"
        else:
            assert operation[0] == "tag"
            assert operation[1] == IMAGE_ID
            output = ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", run)
    return commands


def write_manifest(tmp_path):
    path = tmp_path / "images.json"
    path.write_text(
        json.dumps({"schema_version": "release-images-v1", "images": [item()]})
    )
    return path


def test_check_only_never_tags_or_pushes(tmp_path, monkeypatch, capsys):
    commands = fake_process(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["publish", "--manifest", str(write_manifest(tmp_path))]
    )
    assert publisher.main() == 0
    assert len(commands) == 1
    assert json.loads(capsys.readouterr().out)["published"] is False


@pytest.mark.parametrize(
    "changes", [{"arch": "arm64"}, {"actual_id": "sha256:" + "c" * 64}]
)
def test_changed_or_wrong_platform_image_blocks_all_mutation(monkeypatch, changes):
    commands = fake_process(monkeypatch, **changes)
    with pytest.raises(RuntimeError):
        publisher.publish_one("/unused", item())
    assert len(commands) == 1


def test_publish_verifies_matching_repository_digest(monkeypatch):
    fake_process(monkeypatch)
    assert (
        publisher.publish_one("/unused", item())
        == REMOTE.rsplit(":", 1)[0] + "@" + DIGEST
    )


def test_wrong_repository_digest_is_not_accepted(monkeypatch):
    fake_process(monkeypatch, wrong_repo=True)
    with pytest.raises(RuntimeError, match="does not match"):
        publisher.publish_one("/unused", item())


def test_failed_push_preserves_partial_receipt_without_credentials(
    tmp_path, monkeypatch, capsys
):
    fake_process(monkeypatch, fail_push=True)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish",
            "--manifest",
            str(write_manifest(tmp_path)),
            "--publish",
            "--receipt",
            str(receipt),
        ],
    )
    assert publisher.main() == 1
    data = json.loads(receipt.read_text())
    assert data["status"] == "failed"
    assert data["images"][0]["status"] == "publishing"
    assert "immutable_ref" not in data["images"][0]
    assert "credential-secret" not in capsys.readouterr().out + receipt.read_text()


def test_success_receipt_contains_immutable_reference(tmp_path, monkeypatch):
    fake_process(monkeypatch)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish",
            "--manifest",
            str(write_manifest(tmp_path)),
            "--publish",
            "--receipt",
            str(receipt),
        ],
    )
    assert publisher.main() == 0
    data = json.loads(receipt.read_text())
    assert data["status"] == "published"
    assert data["images"][0]["immutable_ref"].endswith("@" + DIGEST)
    assert publisher.main() == 1  # Never overwrite an existing receipt.


def test_remote_registry_and_latest_are_rejected(tmp_path):
    path = write_manifest(tmp_path)
    for remote in ["evil.example/repo:tag", publisher.PREFIX + "serving:latest"]:
        entry = {**item(), "remote_ref": remote}
        path.write_text(
            json.dumps({"schema_version": "release-images-v1", "images": [entry]})
        )
        with pytest.raises(ValueError):
            publisher.load_manifest(path)
