"""Build the user-run Drive kit: source ZIP, notebooks and START_HERE.md.

Usage: python build_bundle.py --kit-dir /path/to/retraining-effect
No models, data, credentials, Git internals or cloud transfers are included.
Inputs are reused read-only from the R1->R0 transfer kit already on Drive.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTEBOOKS = ["00_prepare.ipynb", "01_f1_cycle_only.ipynb", "02_f2_replay.ipynb", "03_compare_results.ipynb"]
PUBLIC = ["README.md", "experiment.py", "experiment_config.json", "requirements-colab.txt",
          "prepare_models.py", "build_bundle.py", *NOTEBOOKS]
START_HERE = """# 직접 실행 — R0 재학습 효과 실험 (F0/F1/F2)

1. 이 `retraining-effect` 폴더 전체를 **alexyoo0506@gmail.com** Google Drive의 `내 드라이브/LLM_Router/` 아래에 업로드합니다. 입력 데이터와 R0(운영 1.0) 모델은 R1→R0 전이 실험 때 올린 `LLM_Router/transfer-experiments/inputs/experiment-A`를 그대로 읽습니다. 그 폴더는 지우거나 옮기지 마세요.
2. `notebooks/00_prepare.ipynb`를 Colab에서 열고, **런타임 버전 2026.07(Python 3.12)·T4 GPU**로 **모두 실행**합니다. Drive 연결 승인만 직접 합니다. anchor 확인, 로거 추론, 피드백 시뮬레이션과 계획 봉인이 자동으로 진행됩니다.
3. 00이 정상 종료하면 `01_f1_cycle_only.ipynb`와 `02_f2_replay.ipynb`를 **각각 다른 GPU 세션에서 모두 실행**합니다. 한 세션만 가능하면 01 완료 후 02를 실행해도 됩니다. 같은 노트북을 두 세션에서 동시에 실행하지 않습니다.
4. **01과 02가 모두 끝나면** `03_compare_results.ipynb`를 GPU 세션 하나에서 **모두 실행**합니다. 마지막에 내려받아지는 `results-for-review.zip` 하나만 대화에 첨부합니다.

중간 확인 보고는 필요 없습니다. 연결이 끊기면 같은 노트북을 다시 실행하면 저장된 지점부터 이어서 합니다. 오류가 나면 다음 노트북으로 넘어가지 말고 오류 셀을 전달합니다.

운영 서버·GCS·`CURRENT`는 바꾸지 않으며, 모든 결과는 개인 Drive의 `retraining-effect/runs/` 아래에 저장됩니다. 예상 GPU 시간은 T4 기준 총 3~5시간입니다.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-dir", type=Path, required=True)
    args = parser.parse_args()
    kit = args.kit_dir
    if kit.exists():
        raise ValueError("Use a new kit directory")
    paths = [ROOT / "ml" / "pyproject.toml", ROOT / "docs/ml/README.md",
             ROOT / "docs/ml/RETRAINING_EFFECT_EXPERIMENT.md"]
    paths += list((ROOT / "ml/src/llm_router_labeling").rglob("*.py"))
    paths += list((ROOT / "ml/config").glob("*.json"))
    paths += [HERE / name for name in PUBLIC]
    source = kit / "source" / "retraining-source.zip"
    source.parent.mkdir(parents=True)
    records = {}
    with zipfile.ZipFile(source, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(paths)):
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
            records[name] = hashlib.sha256(data).hexdigest()
        info = zipfile.ZipInfo("source-bundle-manifest.json", date_time=(2026, 9, 29, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(info, json.dumps(records, sort_keys=True, indent=2))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix(".sha256").write_text(checksum + "  " + source.name + "\n")
    (kit / "notebooks").mkdir()
    for name in NOTEBOOKS:
        shutil.copyfile(HERE / name, kit / "notebooks" / name)
    (kit / "START_HERE.md").write_text(START_HERE, encoding="utf-8")
    print(json.dumps({"kit": str(kit), "source_sha256": checksum, "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
