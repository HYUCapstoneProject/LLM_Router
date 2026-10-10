"""Build the user-run Drive kit: source ZIP, notebooks and START_HERE.md.

Usage: python build_bundle.py --kit-dir /path/to/retraining-effect [--variant f3|f4|f5]
No models, credentials, Git internals or cloud transfers are included. Inputs are
reused read-only from the R1->R0 transfer kit already on Drive; only the f4/f5 kits
add the sealed test (inputs/test.parquet), rebuilt from SHA-checked local artifacts.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VARIANTS = {
    "v1": {"zip": "retraining-source.zip",
           "notebooks": ["00_prepare.ipynb", "01_f1_cycle_only.ipynb", "02_f2_replay.ipynb", "03_compare_results.ipynb"]},
    "f3": {"zip": "retraining-source-f3.zip",
           "notebooks": ["00_prepare_f3.ipynb", "01_f3_balanced_easy.ipynb", "02_compare_f3.ipynb"]},
    "f4": {"zip": "retraining-source-f4.zip",
           "notebooks": ["00_prepare_f4.ipynb", "01_f4_bidirectional.ipynb", "02_compare_f4.ipynb"]},
    "f5": {"zip": "retraining-source-f5.zip",
           "notebooks": ["00_prepare_f5.ipynb", "01_f5_matched_prior.ipynb", "02_compare_f5.ipynb"]},
}
SEALED_TEST = "ml/artifacts/frozen-evaluation-outcomes/continual-v1-20260924-28cbc342c510"
TEST_PARTS = ("test_routerbench.parquet", "test_xroutebench.parquet")
PUBLIC = ["README.md", "experiment.py", "experiment_config.json", "experiment_config_f3.json", "experiment_config_f4.json", "experiment_config_f5.json",
          "requirements-colab.txt", "prepare_models.py", "build_bundle.py",
          *[name for variant in VARIANTS.values() for name in variant["notebooks"]]]
START_HERE = """# 직접 실행 — R0 재학습 효과 실험 (F0/F1/F2)

1. 이 `retraining-effect` 폴더 전체를 **alexyoo0506@gmail.com** Google Drive의 `내 드라이브/LLM_Router/` 아래에 업로드합니다. 입력 데이터와 R0(운영 1.0) 모델은 R1→R0 전이 실험 때 올린 `LLM_Router/transfer-experiments/inputs/experiment-A`를 그대로 읽습니다. 그 폴더는 지우거나 옮기지 마세요.
2. `notebooks/00_prepare.ipynb`를 Colab에서 열고, **런타임 버전 2026.07(Python 3.12)·T4 GPU**로 **모두 실행**합니다. Drive 연결 승인만 직접 합니다. anchor 확인, 로거 추론, 피드백 시뮬레이션과 계획 봉인이 자동으로 진행됩니다.
3. 00이 정상 종료하면 `01_f1_cycle_only.ipynb`와 `02_f2_replay.ipynb`를 **각각 다른 GPU 세션에서 모두 실행**합니다. 한 세션만 가능하면 01 완료 후 02를 실행해도 됩니다. 같은 노트북을 두 세션에서 동시에 실행하지 않습니다.
4. **01과 02가 모두 끝나면** `03_compare_results.ipynb`를 GPU 세션 하나에서 **모두 실행**합니다. 마지막에 내려받아지는 `results-for-review.zip` 하나만 대화에 첨부합니다.

중간 확인 보고는 필요 없습니다. 연결이 끊기면 같은 노트북을 다시 실행하면 저장된 지점부터 이어서 합니다. 오류가 나면 다음 노트북으로 넘어가지 말고 오류 셀을 전달합니다.

운영 서버·GCS·`CURRENT`는 바꾸지 않으며, 모든 결과는 개인 Drive의 `retraining-effect/runs/` 아래에 저장됩니다. 예상 GPU 시간은 T4 기준 총 3~5시간입니다.
"""
START_HERE_F3 = """# 직접 실행 — F3: easy 보충 재학습

1. 이 `retraining-effect-f3` 폴더 전체를 **alexyoo0506@gmail.com** Google Drive의 `내 드라이브/LLM_Router/` 아래에 업로드합니다. 첫 실험 폴더 `LLM_Router/retraining-effect`(로거 결과와 run001)와 `LLM_Router/transfer-experiments/inputs/experiment-A`를 읽기 전용으로 재사용하므로 두 폴더는 그대로 둡니다.
2. `notebooks/00_prepare_f3.ipynb`를 **런타임 버전 2026.07(Python 3.12)** 으로 **모두 실행**합니다. GPU 추론 없이 계획만 봉인하며, window별 easy 보충 수가 표로 출력됩니다.
3. `01_f3_balanced_easy.ipynb`를 **T4 GPU**에서 **모두 실행**합니다. seed 3개 × 재학습 6회와 window 평가를 합니다(약 1.5~2시간).
4. 끝나면 `02_compare_f3.ipynb`를 T4 GPU에서 **모두 실행**합니다(약 40분). F3 최종 평가 후 첫 실험의 F1/F2와 한 보고서로 비교하고, `results-for-review.zip`을 내려받습니다. 이 파일은 대화에 첨부하지 않아도 Drive에서 바로 확인할 수 있습니다.

연결이 끊기면 같은 노트북을 다시 실행하면 이어서 합니다. 오류가 나면 다음 노트북으로 넘어가지 말고 오류 셀을 전달합니다. 운영 서버·GCS·`CURRENT`는 바꾸지 않습니다.
"""
START_HERE_F4 = """# 직접 실행 — F4: 양방향 오류 교정 재학습

1. 이 `retraining-effect-f4` 폴더 전체를 **alexyoo0506@gmail.com** Google Drive의 `내 드라이브/LLM_Router/` 아래에 업로드합니다. `inputs/test.parquet`(봉인 test 7,202문항)이 함께 들어 있습니다. 첫 실험 폴더 `LLM_Router/retraining-effect`(로거 결과와 run001), F3 폴더 `LLM_Router/retraining-effect-f3`, 입력 `LLM_Router/transfer-experiments/inputs/experiment-A`는 읽기 전용으로 재사용하므로 그대로 둡니다.
2. `notebooks/00_prepare_f4.ipynb`를 **런타임 버전 2026.07(Python 3.12)** 으로 **모두 실행**합니다. GPU 추론 없이 계획만 봉인하며, window별로 고른 문항 수(원래 라벨보다 높게/낮게)가 표로 출력됩니다.
3. `01_f4_bidirectional.ipynb`를 **T4 GPU**에서 **모두 실행**합니다. seed 3개 × 재학습 6회와 window 평가를 합니다(약 1.5~2시간).
4. 끝나면 `02_compare_f4.ipynb`를 T4 GPU에서 **모두 실행**합니다(약 1~1.5시간). 초기 R0(F0)와 F4 최종 모델을 test·RouterArena에서 평가해 비교하고 `results-for-review.zip`을 내려받습니다. Drive의 `retraining-effect-f4/runs/` 아래에도 그대로 남습니다.

연결이 끊기면 같은 노트북을 다시 실행하면 이어서 합니다. 오류가 나면 다음 노트북으로 넘어가지 말고 오류 셀을 전달합니다. 운영 서버·GCS·`CURRENT`는 바꾸지 않습니다.
"""
START_HERE_F5 = """# 직접 실행 — F5: 라벨 비율을 맞춘 오류 교정 재학습

1. 이 `retraining-effect-f5` 폴더 전체를 **alexyoo0506@gmail.com** Google Drive의 `내 드라이브/LLM_Router/` 아래에 업로드합니다. `inputs/test.parquet`(봉인 test 7,202문항, F4와 같은 파일)이 함께 들어 있습니다. `LLM_Router/retraining-effect`(로거·run001), `LLM_Router/retraining-effect-f3`, `LLM_Router/retraining-effect-f4`, `LLM_Router/transfer-experiments/inputs/experiment-A`는 읽기 전용으로 재사용하므로 그대로 둡니다.
2. `notebooks/00_prepare_f5.ipynb`를 **런타임 버전 2026.07(Python 3.12)** 으로 **모두 실행**합니다. GPU 추론 없이 계획만 봉인하며, window별 선택 수와 학습 데이터의 라벨 비율이 표로 나옵니다.
3. `01_f5_matched_prior.ipynb`를 **T4 GPU**에서 **모두 실행**합니다. seed 3개 × 재학습 6회와 window 평가를 합니다(F4보다 학습량이 약 1.7배라 약 2.5~3.5시간).
4. 끝나면 `02_compare_f5.ipynb`를 T4 GPU에서 **모두 실행**합니다(약 1~1.5시간). 초기 R0(F0)·F4·F5를 test·RouterArena에서 같은 문항으로 비교하고 `results-for-review.zip`을 내려받습니다. Drive의 `retraining-effect-f5/runs/` 아래에도 그대로 남습니다.

연결이 끊기면 같은 노트북을 다시 실행하면 이어서 합니다. 오류가 나면 다음 노트북으로 넘어가지 말고 오류 셀을 전달합니다. 운영 서버·GCS·`CURRENT`는 바꾸지 않습니다.
"""


def build_test_input(kit):
    """Concatenate the sealed RouterBench/xRouteBench test after checking the frozen manifest SHAs."""
    import pandas as pd
    held = ROOT / SEALED_TEST
    manifest = json.loads((held / "manifest.json").read_text())
    parts = []
    for name in TEST_PARTS:
        path = held / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Sealed test checksum differs: {path}")
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    out = kit / "inputs" / "test.parquet"
    out.parent.mkdir(parents=True)
    frame.to_parquet(out, index=False)
    record = {"schema_version": "retraining-extra-inputs-v1",
              "roles": {"test": {"path": "test.parquet", "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                 "rows": int(len(frame)),
                                 "source_rows": {k: int(v) for k, v in frame.source_dataset.value_counts().items()},
                                 "built_from": {f"{SEALED_TEST}/{n}": manifest["artifacts"][n]["sha256"] for n in TEST_PARTS}}}}
    (kit / "inputs" / "extra-inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="v1")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    kit = args.kit_dir
    if kit.exists():
        raise ValueError("Use a new kit directory")
    paths = [ROOT / "ml" / "pyproject.toml", ROOT / "docs/ml/README.md",
             ROOT / "docs/ml/RETRAINING_EFFECT_EXPERIMENT.md"]
    paths += list((ROOT / "ml/src/llm_router_labeling").rglob("*.py"))
    paths += list((ROOT / "ml/config").glob("*.json"))
    paths += [HERE / name for name in PUBLIC]
    source = kit / "source" / variant["zip"]
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
    for name in variant["notebooks"]:
        shutil.copyfile(HERE / name, kit / "notebooks" / name)
    start = {"v1": START_HERE, "f3": START_HERE_F3, "f4": START_HERE_F4, "f5": START_HERE_F5}[args.variant]
    (kit / "START_HERE.md").write_text(start, encoding="utf-8")
    extra = build_test_input(kit) if args.variant in {"f4", "f5"} else None
    print(json.dumps({"kit": str(kit), "source_sha256": checksum, "files": len(records), "extra_inputs": extra}, indent=2))


if __name__ == "__main__":
    main()
