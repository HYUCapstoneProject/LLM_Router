from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run-generalization-colab.py"
    spec = importlib.util.spec_from_file_location("g1_colab_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(status, payload=None, headers=None):
    return Mock(status_code=status, headers=headers or {}, json=lambda: payload)


def object_info(data):
    return {
        "id": "object-id",
        "size": str(len(data)),
        "md5Checksum": hashlib.md5(data).hexdigest(),
    }


def test_upload_resumes_from_acknowledged_bytes_and_checks_object(driver, tmp_path):
    data = b"0123456789"
    path = tmp_path / "archive"
    path.write_bytes(data)
    session = Mock()
    session.put.side_effect = [
        response(308, headers={"Range": "bytes=0-3"}),
        response(200, object_info(data)),
    ]
    slot = {"url": "https://www.googleapis.com/upload/secret", "id": "object-id"}
    result = driver.upload_file(path, slot, session=session)
    assert result["verified"]
    sent = session.put.call_args_list[1].kwargs
    assert sent["data"] == data[4:]
    assert sent["headers"]["Content-Range"] == "bytes 4-9/10"


@pytest.mark.parametrize(
    "change",
    [
        {"id": "wrong"},
        {"size": "1"},
        {"md5Checksum": "wrong"},
        {"sha256Checksum": "wrong"},
    ],
)
def test_bad_uploaded_object_rejected(driver, tmp_path, change):
    data = b"payload"
    path = tmp_path / "archive"
    path.write_bytes(data)
    payload = object_info(data)
    payload.update(change)
    session = Mock()
    session.put.return_value = response(200, payload)
    with pytest.raises(driver.BackupError):
        driver.upload_file(
            path,
            {"url": "https://www.googleapis.com/secret", "id": "object-id"},
            session=session,
        )


def test_network_exception_does_not_expose_upload_capability(driver, tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"payload")
    session = Mock()
    session.put.side_effect = RuntimeError("secret-url?token=private")
    with pytest.raises(driver.BackupError) as failure:
        driver.upload_file(
            path,
            {"url": "https://www.googleapis.com/secret", "id": "object-id"},
            session=session,
        )
    assert "private" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_nonprogressing_ack_rejected(driver, tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"payload")
    session = Mock()
    session.put.side_effect = [response(308), response(308)]
    with pytest.raises(driver.BackupError, match="acknowledge"):
        driver.upload_file(
            path,
            {"url": "https://www.googleapis.com/secret", "id": "object-id"},
            session=session,
        )


def test_archive_restore_paths_and_secret_exclusion(driver, tmp_path):
    source = tmp_path / "model"
    source.write_bytes(b"weights")
    secret = tmp_path / "slots.json"
    secret.write_text("private-capability")
    first, second = tmp_path / "first.tar", tmp_path / "second.tar"
    for target in (first, second):
        driver.make_archive(
            target,
            {"r1-natural/model/weights.bin": source},
            {"execution-status.json": {"status": "arm_complete"}},
        )
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert archive.getnames() == [
            "r1-natural/model/weights.bin",
            "execution-status.json",
        ]
    assert b"private-capability" not in first.read_bytes()


def test_backup_failure_propagates_and_receipt_written_only_after_success(
    driver, tmp_path, monkeypatch
):
    root, output = tmp_path / "root", tmp_path / "output"
    root.mkdir()
    output.mkdir()
    slots = tmp_path / "slots.json"
    slots.write_text(
        json.dumps(
            {
                "account": "researcher@example.com",
                "slots": {
                    "r1-natural:500": {
                        "url": "https://www.googleapis.com/secret",
                        "id": "object-id",
                    }
                },
            }
        )
    )
    backup = driver.DriveBackup(
        root, output, ["r1-natural"], slots, expected_account="researcher@example.com"
    )
    source = tmp_path / "checkpoint"
    source.write_bytes(b"checkpoint")

    def fail(*args):
        raise driver.BackupError("checksum mismatch")

    monkeypatch.setattr(driver, "upload_file", fail)
    with pytest.raises(driver.BackupError):
        backup.preserve("r1-natural:500", {"checkpoint": source})
    assert not backup.receipt_path.exists()
    uploader = Mock(return_value={"verified": True, "file_id": "object-id"})
    monkeypatch.setattr(driver, "upload_file", uploader)
    backup.preserve("r1-natural:500", {"checkpoint": source})
    backup.preserve("r1-natural:500", {"checkpoint": source})
    assert uploader.call_count == 1
    assert backup.receipt_path.is_file()


def test_local_mirror_acknowledgement_preserves_until_verified(
    driver, tmp_path, monkeypatch
):
    root, output, mirror = tmp_path / "root", tmp_path / "output", tmp_path / "mirror"
    root.mkdir()
    output.mkdir()
    backup = driver.LocalMirrorBackup(root, output, ["r1-natural"], mirror)
    source = tmp_path / "checkpoint"
    source.write_bytes(b"checkpoint")

    def acknowledge(_):
        request = json.loads((mirror / "request-r1-natural-500.json").read_text())
        archive = Path(request["archive_path"])
        assert archive.exists()
        assert driver.sha256_file(archive) == request["sha256"]
        assert not backup.receipt_path.exists()
        (mirror / "ack-r1-natural-500.json").write_text(json.dumps(request))

    monkeypatch.setattr(driver.time, "sleep", acknowledge)
    backup.preserve("r1-natural:500", {"checkpoint": source})
    receipt = json.loads(backup.receipt_path.read_text())["r1-natural:500"]
    assert receipt["backend"] == "host-local-mirror"
    assert receipt["verified"]
    assert not list(backup.transfer_dir.glob("*.tar"))


@pytest.mark.parametrize("mode", ["timeout", "wrong_hash", "host_error"])
def test_local_mirror_failure_keeps_archive_and_no_receipt(
    driver, tmp_path, monkeypatch, mode
):
    root, output, mirror = tmp_path / "root", tmp_path / "output", tmp_path / "mirror"
    root.mkdir()
    output.mkdir()
    backup = driver.LocalMirrorBackup(
        root,
        output,
        ["r3-natural"],
        mirror,
        timeout_seconds=0 if mode == "timeout" else 900,
    )
    source = tmp_path / "checkpoint"
    source.write_bytes(b"checkpoint")

    def acknowledge(_):
        request = json.loads((mirror / "request-r3-natural-500.json").read_text())
        if mode == "wrong_hash":
            request["sha256"] = "wrong"
        else:
            request["status"] = "error"
        (mirror / "ack-r3-natural-500.json").write_text(json.dumps(request))

    monkeypatch.setattr(driver.time, "sleep", acknowledge)
    with pytest.raises(driver.BackupError):
        backup.preserve("r3-natural:500", {"checkpoint": source})
    assert list(backup.transfer_dir.glob("*.tar"))
    assert not backup.receipt_path.exists()


def test_mounted_drive_probe_and_checkpoint_without_transfer(
    driver, tmp_path, monkeypatch
):
    drive = tmp_path / "MyDrive"
    output = drive / "run"
    output.mkdir(parents=True)
    backup = driver.MountedDriveBackup(tmp_path, output, ["r1-natural"], drive)
    assert (output / "storage-probe.txt").is_file()
    checkpoint = output / "r1-natural/checkpoints/step-00000500"
    checkpoint.mkdir(parents=True)
    verify = Mock(return_value={"metadata": {"optimizer_steps": 500}})
    monkeypatch.setattr(driver, "verify_checkpoint", verify)
    backup.checkpoint("r1-natural", checkpoint)
    verify.assert_called_once_with(checkpoint)
    assert not (output / ".transfer").exists()
    with pytest.raises(driver.BackupError, match="outside"):
        backup.checkpoint("r1-natural", tmp_path / "other-checkpoint")
    with pytest.raises(driver.BackupError, match="inside"):
        driver.MountedDriveBackup(tmp_path, tmp_path, ["r1-natural"], drive)


def test_mounted_drive_completed_arms_verified(driver, tmp_path, monkeypatch):
    drive = tmp_path / "MyDrive"
    output = drive / "run"
    output.mkdir(parents=True)
    backup = driver.MountedDriveBackup(tmp_path, output, ["r1-natural"], drive)
    arm = output / "r1-natural"
    arm.mkdir()
    (arm / "_SUCCESS").write_text("complete")
    (output / "run-plan.json").write_text("{}")
    verifier = Mock()
    monkeypatch.setattr(driver, "_verify_completed", verifier)
    backup.completed_arms()
    verifier.assert_called_once_with(arm, driver.sha256_file(output / "run-plan.json"))


@pytest.mark.parametrize("expected", [None, "", " ", "another@example.com"])
def test_drive_backup_requires_explicit_matching_account(driver, tmp_path, expected):
    slots = tmp_path / "slots.json"
    slots.write_text(json.dumps({"account": "researcher@example.com", "slots": {}}))
    with pytest.raises(driver.BackupError):
        driver.DriveBackup(
            tmp_path, tmp_path / "output", [], slots, expected_account=expected
        )
    assert not (tmp_path / "output").exists()


def test_slots_cli_rejects_missing_account_before_reading_inputs(driver, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--root",
            "/missing",
            "--output-dir",
            "/missing-output",
            "--arms",
            "r1-natural",
            "--backup-slots",
            "/missing-slots",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        driver.main()
