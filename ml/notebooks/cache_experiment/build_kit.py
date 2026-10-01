"""Build the Colab kit for the cache experiment (upload to MyDrive/LLM_Router/).

Layout:
  cache-experiment/
    START_HERE.md
    source/cache-source.zip, cache-source.sha256   code + config, per-file SHA manifest inside
    notebooks/                                       the notebooks to open in Colab
    data-root/<repo-relative input paths>            inputs pinned in experiment_config.json
    runs/                                            notebook outputs (written by Colab)

data-root mirrors repository paths, so experiment_config.json works unchanged with
DATA_ROOT=data-root. Two inputs are read from earlier kits already on Drive and
are not copied: the router logger log (retraining-effect) and the R0 package
(transfer-experiments, byte-identical to the approved package; SHA-pinned). Usage: python build_kit.py --data-root <checkout with ml/data, ml/artifacts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE_FILES = sorted([p for p in HERE.iterdir() if p.suffix in (".py", ".json", ".txt", ".md", ".ipynb")]) + [
    REPO / "backend/src/main/java/com/llmrouter/service/CacheService.java",
    REPO / "infra/postgres/init/01-schema.sql",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads((HERE / "experiment_config.json").read_text())
    out = (args.out or args.data_root / "ml/artifacts/cache-experiment-v1-user-kit") / "cache-experiment"
    if out.exists():
        shutil.rmtree(out)
    (out / "source").mkdir(parents=True)
    (out / "runs").mkdir()

    manifest = {}
    zip_path = out / "source/cache-source.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in SOURCE_FILES:
            name = str(path.relative_to(REPO))
            z.write(path, name)
            manifest[name] = sha(path)
        z.writestr("source-bundle-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    (out / "source/cache-source.sha256").write_text(f"{sha(zip_path)}  cache-source.zip\n")

    copied = {}
    for name, spec in config["inputs"].items():
        if name == "logger_log":
            continue
        source = args.data_root / spec["path"]
        if sha(source) != spec["sha256"]:
            raise SystemExit(f"{name}: SHA mismatch at {source}")
        target = out / "data-root" / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[name] = spec["path"]
    (out / "notebooks").mkdir()
    for notebook in sorted(HERE.glob("*.ipynb")):
        shutil.copy2(notebook, out / "notebooks" / notebook.name)

    (out / "START_HERE.md").write_text(START.format(files="\n".join(f"- `{p}`" for p in copied.values())))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"kit: {out} ({size:.0f} MB)")


START = """# 캐시 실험 Colab kit

이 `cache-experiment` 폴더를 개인 Drive의 `MyDrive/LLM_Router/` 아래에 그대로 올린다.
다음 두 입력은 이미 Drive에 있는 것을 읽기 전용으로 쓴다(SHA 확인).
- 라우터 로그: `MyDrive/LLM_Router/retraining-effect/logger/logger-log.parquet`
- 운영 R0 패키지: `MyDrive/LLM_Router/transfer-experiments/inputs/experiment-A/models/r0`

1. Colab 런타임: 런타임 버전 2026.07(Python 3.12), GPU T4.
2. `notebooks/`의 노트북을 Colab에서 열고 **전체 실행**한다. 첫 셀이 Drive 연결, 소스 ZIP SHA 확인과 압축 해제,
   서빙과 같은 torch 2.5.1 / transformers 4.48.3 설치를 한다.
3. 순서: 00 → 01 → (사람 작업 02) → 03 → 04 → 06. 05(지연·pgvector)는 로컬 Mac에서 실행한다
   (Drive 동기화 폴더의 `runs/`를 읽는다).

산출물은 `runs/<RUN_ID>/`에 쌓이고 write-once로 봉인된다. 다시 하려면 새 RUN_ID를 쓴다.

포함된 입력(SHA는 experiment_config.json에 고정):
{files}
- 운영 R0 라우터 패키지 (`ml/artifacts/r0-release-20260927/approved`)
"""

if __name__ == "__main__":
    main()
