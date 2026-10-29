"""Build 05_f6_gain_metric.ipynb: 04_gain_metric plus the F6 training-free arms.

The G calculation, bootstrap seeds and set order are copied unchanged from
04_gain_metric.ipynb, so F0 and every F6 arm share the same resamples as F1-F3.

Usage: python build_f6_gain_notebook.py <path to 04_gain_metric.ipynb>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# 05 — 학습 없이 갱신(F6)을 첫 시도 이득 $G$로 평가

[04_gain_metric](04_gain_metric.ipynb)의 계산을 그대로 쓰고, 비교군에 F6을 더합니다. F6은 운영 1.0의 가중치를 고정한 채 순서형 문턱 파라미터 4개만 window마다 다시 맞춥니다([training_free.py](training_free.py)). F6의 단계 선택은 저장된 logit에서 결정적으로 다시 계산하므로 GPU가 필요 없습니다. **Colab CPU 런타임**에서 실행합니다.

| 비교군 | 내용 |
|---|---|
| F0 ~ F5 | 04와 같음 |
| F6 | 학습 없이 갱신. 상향 없이 끝난 요청과 상향 기록을 모두 씀(고정 로거) |
| F6-on | F6과 같지만 F6이 직접 고른 단계의 결과로 갱신 |
| F6-esc | F6과 같은 모델에 상향 기록만 씀(현재 `/escalate` 계약으로 받을 수 있는 정보) |

bootstrap 난수 seed와 평가 데이터 순서가 04와 같아 F0~F5의 값과 구간이 04 결과와 같게 나와야 합니다. 마지막 셀에서 확인합니다. F6은 구간을 따라가며 갱신하므로 F1~F3과 같은 다음 구간 문항(prequential)이 주 평가이고, 구간 6까지 맞춘 파라미터를 최종 평가 세트(test, 7,202문항)에도 적용해 F4·F5와 같은 문항으로 비교합니다.
"""

SETUP = """# 환경 설정. Colab이면 Drive를 붙입니다. 다른 곳에서 돌릴 때는 LLM_ROUTER_DRIVE에 Drive의 LLM_Router 폴더 경로를 넣습니다.
import hashlib, json, os, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    display
except NameError:  # 노트북 밖에서 실행할 때
    display = print

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = Path('/content/drive/MyDrive/LLM_Router')
else:
    DRIVE = Path(os.environ['LLM_ROUTER_DRIVE'])

RUNS = {
    'run001': (DRIVE / 'retraining-effect/runs/retraining-effect-v1-run001', ['f1_cycle_only', 'f2_replay']),
    'f3': (DRIVE / 'retraining-effect-f3/runs/retraining-effect-v2-f3-run001', ['f3_balanced_easy']),
    'f4': (DRIVE / 'retraining-effect-f4/runs/retraining-effect-v3-f4-run001', ['f4_bidirectional']),
    'f5': (DRIVE / 'retraining-effect-f5/runs/retraining-effect-v4-f5-run001', ['f5_matched_prior']),
}
OUT_DIR = Path(os.environ.get('F6_GAIN_OUT', DRIVE / 'retraining-gain-f6-20261021'))
REFERENCE_DIR = DRIVE / 'retraining-gain-20261021'

FROZEN = 'f0_frozen'
F6_ARMS = ['f6_training_free', 'f6_training_free_on_policy', 'f6_escalation_only']
ARMS = [FROZEN, 'f1_cycle_only', 'f2_replay', 'f3_balanced_easy', 'f4_bidirectional', 'f5_matched_prior',
        *F6_ARMS]
SHORT = dict(zip(ARMS, ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F6-on', 'F6-esc']))
NAME = {FROZEN: '재학습 안 함', 'f1_cycle_only': '새 피드백만', 'f2_replay': '과거 데이터 섞기',
        'f3_balanced_easy': 'easy 보충', 'f4_bidirectional': '양방향 교정(오답만)',
        'f5_matched_prior': '양방향 교정 + 비율 맞춤', 'f6_training_free': '학습 없이 갱신',
        'f6_training_free_on_policy': '학습 없이 갱신(자기 선택으로 갱신)',
        'f6_escalation_only': '학습 없이 갱신(상향 기록만)'}
SEEDS = (42, 43, 44)
WINDOWS = range(2, 7)
SETS = ['prequential', 'gate', 'forget', 'test']   # 04와 같은 순서(bootstrap 난수 seed가 순서에 묶여 있음)

# F6 helper: training_free.py와, 그것이 쓰는 experiment.py(F3 kit에 든 판)를 불러온다.
TF_PIN = '7edd05c67d74d9b45bb43a4d8affbf59fa276aefb1c3ecf66c0c507c29de1cb2'
helper_dir = Path('/content/f6-gain') if IN_COLAB else Path(os.environ.get('F6_HELPER_DIR', '/tmp/f6-gain'))
helper_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DRIVE / 'retraining-effect-f3/source/retraining-source-f3.zip') as z:
    (helper_dir / 'experiment.py').write_bytes(z.read('ml/notebooks/retraining_effect/experiment.py'))
source = (DRIVE / 'retraining-effect-f6/training_free.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == TF_PIN, 'training_free.py가 고정한 판과 다릅니다'
(helper_dir / 'training_free.py').write_bytes(source)
sys.path.insert(0, str(helper_dir))
import training_free as tf

for key, (root, _) in RUNS.items():
    print(f'{key:7s}', root, '있음' if root.is_dir() else '없음')
missing_runs = [k for k, (root, _) in RUNS.items() if not root.is_dir()]
if missing_runs:
    raise FileNotFoundError(f'run 폴더가 없습니다: {missing_runs}. DRIVE 경로를 확인하세요.')"""

F6_CELL_MD = """## F6 단계 선택 다시 계산

run001의 봉인된 stream(logit 포함)으로 F6 세 가지를 다시 돌립니다. window $j+1$은 window 1~$j$의 관측으로 맞춘 파라미터로 고릅니다. 최종 모델 평가(gate·forget)는 window 6까지 맞춘 파라미터를 F0의 최종 logit에 적용합니다. 결정적 계산이라 seed는 0 하나입니다."""

F6_CELL = """run_dir = RUNS['run001'][0]
plan = json.loads((run_dir / 'plan.json').read_text())
full_stream = pd.read_parquet(run_dir / 'shared-data/stream-windows.parquet')
frozen = full_stream[tf.LOGIT_COLUMNS].to_numpy(float).argmax(1)
assert np.array_equal(frozen, as_tier(full_stream['initial_label'])), 'logit이 봉인된 로거 예측을 재현하지 않습니다'
modes = {'f6_training_free': 'fixed_logger', 'f6_training_free_on_policy': 'on_policy',
         'f6_escalation_only': 'escalation_only'}
f6_params, f6_fits = {}, {}
for arm, mode in modes.items():
    decisions, params, fits = tf.run_stream(full_stream, plan['config']['windows'], mode)
    f6_params[arm], f6_fits[arm] = params, fits
    later = full_stream.window.isin(WINDOWS).to_numpy()
    frame = full_stream.loc[later, KEEP].copy()
    frame['prediction'] = np.asarray(LABELS)[decisions[later]]
    add('prequential', arm, 0, frame)

for role in ('gate', 'forget', 'test'):
    # F0의 최종 예측(logit)이 있는 첫 run을 쓴다. test는 F4·F5 run에만 있다.
    directory = next(root / 'scores' / FROZEN / 'seed-0' / 'final' / role for root, _ in RUNS.values()
                     if (root / 'scores' / FROZEN / 'seed-0' / 'final' / role / 'receipt.json').exists())
    path = directory / 'predictions.parquet'
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert sha256(path) == receipt['predictions_sha256'], path
    final = pd.read_parquet(path)
    for arm, params in f6_params.items():
        frame = final[KEEP].copy()
        frame['prediction'] = np.asarray(LABELS)[tf.decide(params, tf.features(final))]
        add(role, arm, 0, frame)
print({SHORT[a]: tf.params_record(p) for a, p in f6_params.items()})"""

CHECK_MD = """## 04 결과와 같은지 확인

F0~F5의 $G$는 04의 `gain-summary.csv`와 같아야 하고, 95% 구간은 float32 합산 순서 차이(10⁻⁵ 미만)만 허용합니다. 그래야 bootstrap 재표집이 공유된 것입니다."""

CHECK = """reference = pd.read_csv(REFERENCE_DIR / 'gain-summary.csv')
mine = summary[summary.arm.isin(['F0', 'F1', 'F2', 'F3', 'F4', 'F5'])][['set', 'arm', 'G', 'G_lo', 'G_hi']]
merged = mine.merge(reference[['set', 'arm', 'G', 'G_lo', 'G_hi']], on=['set', 'arm'], suffixes=('', '_04'))
assert len(merged) == len(mine), '04 결과에 없는 행이 있습니다'
# 점 추정은 같아야 하고, 구간은 float32 합산 순서 차이만 허용한다.
for col, tol in (('G', 1e-12), ('G_lo', 1e-5), ('G_hi', 1e-5)):
    gap = (merged[col] - merged[f'{col}_04']).abs().max()
    assert gap < tol, f'{col}이 04와 {gap} 다릅니다'
print('F0~F5가 04 결과와 일치:', len(merged), '행')"""


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def main(reference):
    nb = json.loads(Path(reference).read_text())
    cells = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in nb["cells"]]
    assert text(cells[1]).startswith("# 환경 설정") and "RUNS = {" in text(cells[1])
    assert "streams = {}" in text(cells[3]) and "def add(" in text(cells[3])
    cells[0] = markdown(INTRO)
    cells[1] = code(SETUP)
    # F4/F5 runs are gone; 'streams' now compares run001 with f3 only, unchanged code handles that.
    cells[3:4] = [cells[3], markdown(F6_CELL_MD), code(F6_CELL)]
    save = next(i for i, c in enumerate(cells) if "OUT_DIR.mkdir" in text(c))
    cells[save] = code(text(cells[save])
                       .replace("ml/notebooks/retraining_effect/04_gain_metric.ipynb",
                                "ml/notebooks/retraining_effect/05_f6_gain_metric.ipynb")
                       .replace("'runs': {k: str(root.relative_to(DRIVE)) for k, (root, _) in RUNS.items()},",
                                "'runs': {k: str(root.relative_to(DRIVE)) for k, (root, _) in RUNS.items()},\n"
                                "    'f6_helper_sha256': TF_PIN,\n"
                                "    'f6_params': {SHORT[a]: tf.params_record(p) for a, p in f6_params.items()},"))
    cells[save:save] = [markdown(CHECK_MD), code(CHECK)]
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and "colors = [" in text(cell):
            cells[i] = code(text(cell).replace("'#4c43c4' if a == 'F0' else '#686d78'",
                                               "'#4c43c4' if a.startswith('F6') else '#686d78'"))
    nb["cells"] = cells
    (HERE / "05_f6_gain_metric.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
