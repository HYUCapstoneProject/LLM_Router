"""Build the three R2/R3 Colab notebooks and a self-contained source delivery ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result |= {"execution_count": None, "outputs": []}
    return result


def notebook(cells, gpu=False):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []},
        **({"accelerator": "GPU"} if gpu else {})},
        "cells": [dict(c, id=f"r23-{i:02d}") for i, c in enumerate(cells)]}


def setup(digest, gpu):
    install = """
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'scikit-learn==1.6.1', 'numpy==2.1.2', 'pandas==2.3.3', 'pyarrow==21.0.0'], check=True)
    """
    if gpu:
        install = """
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.5.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                        str(REPO / 'ml/notebooks/router_final/requirements-train-colab.txt')], check=True)
        """
    return [cell("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
            cell("code", f"""
            from pathlib import Path
            import hashlib, json, os, subprocess, sys, zipfile
            RUN = Path('/content/drive/MyDrive/LLM_Router/router-final/runs/run001')
            REPO = Path('/content/router-final-r23-source')
            SOURCE_ZIP = RUN / 'code/router-final-source-r23.zip'
            EXPECTED_SHA = '{digest}'
            if not SOURCE_ZIP.is_file():
                from google.colab import files
                print('배포 묶음 안의 router-final-source-r23.zip을 선택하세요.')
                uploaded = files.upload()
                SOURCE_ZIP = Path('/content/router-final-source-r23.zip')
                SOURCE_ZIP.write_bytes(uploaded['router-final-source-r23.zip'])
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            assert sha(SOURCE_ZIP) == EXPECTED_SHA, '노트북과 소스 ZIP의 버전이 다릅니다.'
            with zipfile.ZipFile(SOURCE_ZIP) as bundle:
                for name in bundle.namelist():
                    assert (REPO / name).resolve().is_relative_to(REPO.resolve()), name
                bundle.extractall(REPO)
            for name, digest in json.loads((REPO / 'source-bundle-manifest.json').read_text()).items():
                assert sha(REPO / name) == digest, name
            assert sys.version_info[:2] in {{(3, 11), (3, 12)}}, 'Python 3.12 런타임을 선택하세요.'
            """), cell("code", install),
            cell("markdown", "계산은 별도 Python 프로세스에서 실행하므로 설치 후 세션을 다시 시작할 필요가 없습니다. 아래 경로가 기존 실험의 Drive 폴더인지 확인하세요."),
            cell("code", """
            from pathlib import Path
            import os, subprocess, sys
            SCRIPT = REPO / 'ml/notebooks/router_final/r23_extension.py'
            ENV = dict(os.environ, PYTHONPATH=str(REPO / 'ml/src'), PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false')
            def run_logged(command, *args):
                cmd = [sys.executable, str(SCRIPT), command, '--run-dir', str(RUN), '--repo', str(REPO),
                       *map(str, args)]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=ENV)
                for line in proc.stdout:
                    print(line, end='', flush=True)
                if proc.wait():
                    raise RuntimeError(f'{command} 실패 (exit {proc.returncode}). 위 오류를 확인하세요.')
            run_logged('preflight')
            """)]


def train_notebook(digest, method):
    name = "R2" if method == "r2_group" else "R3"
    target = ("집단별 해결 확률의 차이로 쌍별 순위를 학습합니다. 동률 가중치는 기존 R2와 같은 0.25입니다."
              if name == "R2" else
              "집단별 해결 확률 − λ × 비용의 최선 단계를 맞힙니다. 최선·차선 차이로 CE를 가중합니다. "
              "학습 λ는 기존 R3의 0, .05, .1, .2, .4, .8, 1.6, 3.2입니다.")
    cells = [cell("markdown", f"""
    # {name} 모델 집단 학습 + 점수 저장

    {target}

    기존 EV-10의 RouterBench·xRouteBench·LLMRouterBench·SPROUT 모델 집단과
    `ours/data/blended`를 그대로 씁니다. 집단의 **해결 확률**이 학습 target이며,
    과거 대표 모델의 연속 품질 점수와 의미가 다릅니다.
    ModernBERT-base, 512토큰, 3 epoch, seed 42, 원천별 동일 총 가중치,
    실효 batch 32, 비용 .2/.6/1.0을 기존 R0·R1과 맞춥니다.

    **런타임: T4 GPU, Python 3.12.** 새 LLM API 호출은 없습니다.
    입력은 기존 `run001`의 준비 데이터와 G 평가 산출물입니다.
    결과는 `run001/r23-extension`에 저장됩니다.
    학습은 기존 trainer와 같아 중간 재개 기능이 없습니다. 연결이 끊기면
    완료 여부를 확인하고 미완료 로컬 학습만 새 폴더에서 다시 실행하세요.
    기존 R0·R1과 논문 라우터는 다시 학습하지 않습니다.
    """)]
    cells += setup(digest, True)
    cells += [cell("code", f"""
    METHOD = '{method}'
    LOCAL = Path('/content') / ('ev10-' + METHOD)
    SAVED = RUN / 'r23-extension/models' / METHOD
    subprocess.run([sys.executable, '-c',
        "import torch; assert torch.cuda.is_available(), 'T4 GPU를 선택하세요'; "
        "print(torch.cuda.get_device_name(0), torch.__version__)"], env=ENV, check=True)
    """), cell("markdown", "## 1. 시험 학습\n\n64행·2 step으로 loss와 런타임을 확인합니다. 시험 모델은 본 평가에 들어갈 수 없습니다."),
    cell("code", """
    import tempfile
    if not (SAVED / '_SUCCESS').exists():
        SMOKE = Path(tempfile.mkdtemp(prefix=METHOD + '-smoke-', dir='/content'))
        run_logged('train', '--method', METHOD, '--output-dir', SMOKE, '--device', 'cuda', '--smoke')
    """), cell("markdown", "## 2. 본 학습과 Drive 보존\n\n완료된 모델이 있으면 무결성을 확인하고 재사용합니다. 새 모델은 로컬에서 학습을 마친 뒤 Drive에 복사·재검증합니다."),
    cell("code", """
    if (SAVED / '_SUCCESS').exists():
        run_logged('train', '--method', METHOD, '--output-dir', SAVED, '--device', 'cuda')
    else:
        run_logged('train', '--method', METHOD, '--output-dir', LOCAL, '--device', 'cuda')
        run_logged('publish', '--method', METHOD, '--local-model', LOCAL)
    """), cell("markdown", "## 3. dev·test 질문의 점수 저장\n\n각 질문의 세 logit을 저장합니다. test에서는 질문 텍스트와 ID만 읽으며 성패 정답은 읽지 않습니다. 기존 07a와 같은 512토큰 추론입니다."),
    cell("code", "run_logged('score', '--method', METHOD, '--device', 'cuda')"),
    cell("code", """
    import json
    print(json.dumps(json.loads((SAVED / 'metrics.json').read_text()), indent=1)[:4000])
    print('점수:', RUN / 'r23-extension/scores' / METHOD / 'scores.parquet')
    """), cell("markdown", "R2와 R3의 점수 저장이 모두 끝나면 `13_r23_gain.ipynb`를 CPU 런타임에서 실행하세요.")]
    return notebook(cells, gpu=True)


def gain_notebook(digest):
    cells = [cell("markdown", """
    # R2·R3 집단 라우터 추가 — G 평가

    **런타임: CPU, Python 3.12.** 학습·모델 추론·LLM API 호출이 없습니다.
    `12a_r2_group.ipynb`와 `12b_r3_group.ipynb`의 학습·점수 저장을 먼저 끝냅니다.

    기존 비교군은 `run001/gain-lock`에 고정된 설정·보정·λ를 그대로 재사용합니다.
    R2·R3만 dev로 같은 5분할 isotonic 보정을 하고 비용 수준별 λ를 고정합니다.
    새로운 설정 탐색은 하지 않습니다. 주 결과는 τ=.8, seed 42입니다.

    **G = A − H(C)**: A는 첫 시도 해결 확률, C는 첫 호출 비용,
    H(C)는 같은 비용 이하의 고정 단계 또는 무작위 혼합이 얻는 최선의 해결 확률입니다.
    기존 G 구현을 그대로 사용합니다. 에스컬레이션을 포함하는 효용 U와는 다른 지표입니다.

    비용 1:3:5가 주 결과이며 τ=.8에서는 집단 평균 비용과 길이 반영 비용도 계산합니다.
    τ 곡선·비용 가중치 λ별 비교·원천별/전체 표·쌍별 차이·95% 구간을 출력합니다.
    paired bootstrap 10,000회이며 기존 비교군의 점수와 구간 재현을 검사합니다.

    기존 test를 재사용하는 후속 비교입니다. 결과를 보고 학습 설정이나 λ를 다시 맞추지 않습니다.
    """)]
    cells += setup(digest, False)
    cells += [cell("markdown", "## 1. R2·R3 보정과 λ 고정 — dev만 사용\n\n기존 라우터는 다시 선택하거나 보정하지 않습니다. 아래 요약은 test 평가 전에 기록됩니다."),
    cell("code", """
    LOCK = RUN / 'r23-extension/gain-lock'
    if not LOCK.exists():
        run_logged('lock')
    print((LOCK / 'gain-lock-summary.md').read_text())
    """), cell("markdown", "## 2. dev 리허설\n\n경로·평가 코드를 검사합니다. dev로 보정한 값을 dev에서 평가하므로 성능 결과로 사용하지 않습니다."),
    cell("code", """
    import tempfile
    REHEARSAL = Path(tempfile.mkdtemp(prefix='r23-rehearsal-', dir='/content')) / 'results'
    run_logged('rehearse', '--output-dir', REHEARSAL)
    print((REHEARSAL / 'gain-summary.md').read_text()[:3000])
    """), cell("markdown", "## 3. 고정한 정책으로 test 평가\n\n원본 `gain-test`를 유지하고 `r23-extension/gain-test`에 새 결과를 씁니다. 기존 라우터의 원천별 A·C·H·G·선택 비율·95% 구간이 이전과 다르면 검사에서 실패합니다."),
    cell("code", """
    OUT = RUN / 'r23-extension/gain-test'
    assert not OUT.exists(), '이미 출력이 있습니다. 완료 결과를 읽거나, 실패 로그를 확인한 뒤 미완료 폴더를 별도 보관하세요.'
    run_logged('test')
    """), cell("code", """
    import pandas as pd, json
    print((OUT / 'gain-summary.md').read_text())
    display(pd.read_csv(OUT / 'r0-r3-comparison.csv'))
    print('기존 결과 재현:', json.loads((OUT / 'manifest.json').read_text())['baseline_point_and_interval_reproduced'])
    """), cell("markdown", "## 4. 결과 다운로드\n\n표와 manifest, 요약만 묶습니다. 모델 가중치나 질문 원문은 포함하지 않습니다."),
    cell("code", """
    import zipfile
    from google.colab import files
    RESULT_ZIP = Path('/content/r23-group-gain-results.zip')
    with zipfile.ZipFile(RESULT_ZIP, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.suffix in {'.csv', '.md', '.json', '.parquet'}:
                bundle.write(path, path.name)
    files.download(str(RESULT_ZIP))
    """)]
    return notebook(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / 'build_kit.py'), '--output-dir', str(out)], check=True)
    source = out / 'router-final-source-r23.zip'
    (out / 'router-final-source.zip').replace(source)
    (out / 'router-final-source.sha256').unlink()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix('.sha256').write_text(f'{digest}  {source.name}\n')
    notebooks = {'12a_r2_group.ipynb': train_notebook(digest, 'r2_group'),
                 '12b_r3_group.ipynb': train_notebook(digest, 'r3_group'),
                 '13_r23_gain.ipynb': gain_notebook(digest)}
    for name, nb in notebooks.items():
        path = HERE / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        shutil.copyfile(path, out / name)
    shutil.copyfile(HERE / 'R23_COLAB.md', out / 'README.md')
    files = [out / n for n in [*notebooks, source.name, source.with_suffix('.sha256').name, 'README.md']]
    write_manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / 'manifest.json').write_text(json.dumps(write_manifest, indent=2) + '\n')
    target = out / 'r23-colab-kit.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files + [out / 'manifest.json']:
            bundle.write(path, path.name)
    print(target)


if __name__ == '__main__':
    main()
