"""Verify (or restore) the pinned R0 anchor from the existing personal Drive archive only.

The R1->R0 transfer run already restored it under inputs/experiment-A/models/r0.
This script only re-checks those files and restores them from the pinned tar if
they are missing. It never downloads anything or touches other models.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--drive-root', type=Path, default=Path('/content/drive/MyDrive'))
    args = parser.parse_args()
    pins = json.loads((args.inputs / 'source_pins.json').read_text())
    archive_pin = json.loads((args.inputs / 'model-archives.json').read_text())['r0']
    model = args.inputs / pins['models']['r0']['path']
    expected = pins['models']['r0']['files']
    if model.exists() and all((model / n).is_file() and sha(model / n) == h for n, h in expected.items()):
        print('anchor r0 verified existing model')
        return
    archive = args.drive_root / archive_pin['drive_relative_path']
    if not archive.is_file():
        raise FileNotFoundError(f'Expected existing archive on alexyoo0506 Drive: {archive}')
    if sha(archive) != archive_pin['sha256']:
        raise ValueError(f'Archive checksum differs: {archive}')
    # Inspect regular members only; never extract links or archive paths.
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        candidates = [n[:-len('model.safetensors')] for n in members
                      if n.endswith('/model.safetensors') or n == 'model.safetensors']
        matched = None
        for prefix in candidates:
            if not all(prefix + n in members for n in expected):
                continue
            with tar.extractfile(members[prefix + 'model.safetensors']) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() == expected['model.safetensors']:
                    matched = prefix
                    break
        if matched is None:
            raise ValueError(f'Pinned anchor not found inside {archive}')
        model.mkdir(parents=True, exist_ok=True)
        for name, checksum in expected.items():
            dest = model / name
            if dest.is_file() and sha(dest) == checksum:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name(dest.name + '.partial')
            with tar.extractfile(members[matched + name]) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if sha(temporary) != checksum:
                raise ValueError(f'Extracted checksum differs: {name}')
            temporary.replace(dest)
    print('anchor r0 restored and verified')


if __name__ == '__main__':
    main()
