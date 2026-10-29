"""Build 04_f6_training_free.ipynb (CPU runtime).

training_free.py is uploaded to MyDrive/LLM_Router/retraining-effect-f6/ and
experiment.py is taken from an existing retraining kit ZIP on Drive; both are pinned
by the SHA-256 of the local files at build time.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def main():
    pins = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("experiment.py", "training_free.py")}
    setup = f"""import hashlib, pathlib, shutil, sys, zipfile
DRIVE = pathlib.Path('/content/drive/MyDrive/LLM_Router')
root = pathlib.Path('/content/f6'); root.mkdir(exist_ok=True)
PINS = {json.dumps(pins)}
member = 'ml/notebooks/retraining_effect/experiment.py'
found = {{}}
for archive in sorted(DRIVE.glob('retraining-effect*/source/*.zip')):
    with zipfile.ZipFile(archive) as z:
        if member in z.namelist():
            data = z.read(member)
            found[hashlib.sha256(data).hexdigest()] = (archive, data)
print({{k[:12]: str(v[0].name) for k, v in found.items()}})
if PINS['experiment.py'] in found:
    chosen = PINS['experiment.py']
else:
    # Fall back to the helper that produced the F3 report; F6 imports only metric/replay functions.
    chosen = next(k for k, v in found.items() if v[0].name == 'retraining-source-f3.zip')
    print('WARNING: pinned experiment.py not in any kit; using F3 kit helper', chosen)
(root / 'experiment.py').write_bytes(found[chosen][1])
source = DRIVE / 'retraining-effect-f6' / 'training_free.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == PINS['training_free.py'], 'training_free.py differs from the pinned build'
shutil.copy(source, root / 'training_free.py')
sys.path.insert(0, str(root))
print('helpers ready')"""
    cells = [
        markdown(
            "# F6 학습 없이 갱신 (training-free score update)\n\n"
            "운영 1.0(ModernBERT)은 그대로 두고, 고정 logit 위의 순서형 모델 파라미터 4개만 window마다 다시 맞춘다. "
            "run001의 봉인된 stream과 F0 최종 예측을 Drive에서 읽기 전용으로 읽고, 결과는 새 디렉터리에만 쓴다. "
            "**CPU 런타임으로 충분하다**(GPU·모델 추론 없음). 전체 실행(Run all)."),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        code(setup),
        code(
            "import json\nimport training_free as tf\n"
            "SOURCE = DRIVE / 'retraining-effect/runs/retraining-effect-v1-run001'\n"
            "OUT = DRIVE / 'retraining-effect-f6/runs/f6-training-free-v1-20261021'\n"
            "assert not OUT.exists(), 'Output exists; use a new run name instead of overwriting'\n"
            "plan, stream, finals = tf.load_run(SOURCE)\n"
            "print(len(stream), {k: len(v) for k, v in finals.items()})\n"
            "result = tf.evaluate(plan, stream, finals)\n"
            "tf.write_report(result, OUT, SOURCE)\n"
            "print(json.dumps(result['judgments'], indent=2))"),
        code("print((OUT / 'REPORT.md').read_text())"),
    ]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}}}
    (HERE / "04_f6_training_free.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
