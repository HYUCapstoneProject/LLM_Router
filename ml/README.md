# RouterBench Labeling

RouterBench 0-shot/5-shot 배포본을 검증하고, 최소 충분 모델 성능 티어 기준으로 `easy`, `medium`, `hard` 라벨을 생성한다. 상세 계약과 체크리스트는 `docs/routerbench-data-labeling-spec.md`를 따른다.

## 소스 구조

데이터 용도와 원본 데이터셋이 파일명에서 드러나도록 package를 나눈다.

```text
src/llm_router_labeling/
├── training/
│   ├── routerbench_labeling.py       # RouterBench 모델 성능 → 난이도 라벨
│   ├── routerbench_preprocessing.py  # 라벨 데이터 → 학습 Parquet
│   ├── routerbench_cli.py            # routerbench-labeling 명령
│   ├── modernbert_training.py        # 로컬/Kubeflow 공통 학습 코어
│   └── modernbert_training_cli.py    # modernbert-train 명령
├── evaluation/
│   ├── routerarena_preprocessing.py  # RouterArena → 평가 Parquet
│   ├── routerarena_cli.py            # routerarena-preprocessing 명령
│   ├── routerarena_routing_evaluator.py # checkpoint 예측과 라벨 비교
│   └── routerarena_evaluation_cli.py # routerarena-evaluate 명령
└── shared/
    ├── dataset_io.py                 # 다운로드/checksum/config/JSON
    └── text_normalization.py         # 데이터셋 공통 중복 탐지 정규화
```

`training`은 모델 학습에 들어갈 RouterBench 데이터를 만들고, `evaluation`은 학습에
섞지 않는 RouterArena 검증 데이터를 만든다. `shared`에는 어느 한 데이터셋의
라벨 의미나 schema에 의존하지 않는 기능만 둔다.

## 환경 구성

Python 3.11-3.13을 사용한다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r ml/requirements-lock.txt
.venv/bin/pip install --no-deps -e ml
```

## 테스트

```bash
.venv/bin/python -m unittest discover -s ml/tests -v
```

## 공식 데이터 다운로드

다운로드 중 중단되면 `.part` 파일에서 이어받는다. 완료된 파일은 설정에 고정된 크기와 SHA-256을 모두 확인한다.

```bash
.venv/bin/routerbench-labeling download --data-dir ml/data/raw
```

## 100행 smoke labeling

공식 Pickle의 URL과 checksum을 확인한 실행자만 `--trust-pickle`을 사용한다.

```bash
.venv/bin/routerbench-labeling smoke \
  --data-dir ml/data/raw \
  --output-dir ml/artifacts/routerbench-smoke-100 \
  --sample-size 100 \
  --seed 42 \
  --trust-pickle
```

산출물은 다음 위치에 생성되며 Git에서 제외된다.

```text
ml/artifacts/routerbench-smoke-100/labeled.parquet
ml/artifacts/routerbench-smoke-100/metrics.json
ml/artifacts/routerbench-smoke-100/model_tiers.json
```

## 전체 데이터 labeling

MVP 핵심 eval 34,778행 전체에는 `label` 명령을 사용한다. `--output-dir`를 생략하면
dataset/tier/label rule version을 조합한 경로를 자동으로 사용하므로 서로 다른 규칙의
산출물을 덮어쓰지 않는다.

```bash
.venv/bin/routerbench-labeling label \
  --data-dir ml/data/raw \
  --trust-pickle
```

```text
ml/artifacts/routerbench-labeled/
  withmartian-routerbench-2024-03-27__routerbench-capability-tier-v1__minimum-sufficient-tier-v1/
    labeled.parquet
    metrics.json
    model_tiers.json
```

## 학습 전처리

라벨 산출물에서 RouterBench 전용 객관식 지시문을 제거하고, 동일·유사 질문을
같은 split에 묶은 뒤 신뢰도 가중치를 적용한다. 모델 입력에 분석용 성공률 컬럼이
섞이지 않도록 학습 projection과 분석 metadata를 분리한다.

```bash
.venv/bin/routerbench-labeling preprocess \
  --input-path ml/artifacts/routerbench-smoke-100/labeled.parquet \
  --output-dir ml/artifacts/routerbench-preprocessed-smoke-100 \
  --seed 42
```

```text
ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet
ml/artifacts/routerbench-preprocessed-smoke-100/metadata.parquet
ml/artifacts/routerbench-preprocessed-smoke-100/metrics.json
```

전체 라벨 산출물은 같은 전처리 구현에 `full` profile을 지정한다. 입력과 출력 경로를
생략하면 config version과 일치하는 전체 라벨만 읽고, dataset/tier/label version이
다르면 실행을 중단한다.

```bash
.venv/bin/routerbench-labeling preprocess --profile full --seed 42
```

```text
ml/artifacts/routerbench-training/
  withmartian-routerbench-2024-03-27__minimum-sufficient-tier-v1__routerbench-training-preprocess-v1/
    training.parquet
    metadata.parquet
    metrics.json
```

`training.parquet`은 `sample_id`, `model_input_text`, `difficulty_label`,
`sample_weight`, `split`, `duplicate_group_id`만 포함한다. tokenizer 기반 길이 제한과
truncation은 모델 tokenizer가 확정되는 학습 컴포넌트에서 수행한다.

## ModernBERT 로컬 학습 smoke

로컬과 향후 Kubeflow component는 같은 `modernbert-train` CLI를 사용한다. 차이는
데이터·출력 경로와 device/resource config뿐이다. 설계 근거와 계약은
`docs/modernbert-training-spec.md`에 기록한다.

```bash
python3.12 -m venv .venv-training
.venv-training/bin/pip install -r ml/requirements-training-macos.txt
.venv-training/bin/pip install -r ml/requirements-training.lock
.venv-training/bin/pip install --no-deps -e ml

HF_HOME=ml/.cache/huggingface .venv-training/bin/modernbert-train \
  --config ml/config/modernbert_smoke.json \
  --runtime-contract ml/config/modernbert_runtime_cu124_v1.json \
  --training-path ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet \
  --output-dir ml/artifacts/modernbert-smoke
```

smoke config는 `answerdotai/ModernBERT-base`의 immutable revision, 512 token,
`easy=0, medium=1, hard=2`, FP32와 2 optimizer step을 고정한다. 학습은 Parquet의
`sample_weight`를 cross-entropy에 반영한다. 출력에는 Hugging Face model/tokenizer,
`metrics.json`, `training_config.json`이 포함되며 저장 직후 checkpoint를 재로딩해
라벨 순서와 1행 추론을 다시 검증한다. 실제 Python/PyTorch/CUDA/Transformers/data
library 버전도 `metrics.json.runtime`에 기록하며 계약과 다르면 학습을 시작하지 않는다.

T4 전체 학습은 `ml/config/modernbert_t4_full.json`을 사용한다. native 8,192 token,
FP16, batch 1, gradient accumulation 8과 gradient checkpointing을 고정한다.

로컬 GPU가 없으면 VS Code에 설치된 공식 `google.colab` 확장과
`ml/notebooks/modernbert_smoke_colab.ipynb`를 사용한다. notebook 실행 전에 현재
소스/config와 smoke Parquet을 묶은 아래 archive를 Colab `/content`로 업로드한다.

```text
ml/artifacts/modernbert-colab-smoke-input.tar.gz
```

archive는 아래 명령으로 현재 source/config/runtime lock에서 다시 만든다.

```bash
./ml/scripts/build-modernbert-colab-smoke-archive.sh
```

## ModernBERT 학습 Docker image

Kubeflow/GKE에서는 Colab과 같은 `modernbert-train`을 entrypoint로 제공하는 T4용
linux/amd64 image를 사용한다. 데이터와 checkpoint는 image에 넣지 않고 Artifact로
마운트한다.

```bash
./ml/scripts/build-modernbert-training-image.sh

docker run --rm \
  llm-router-modernbert-training:cu124-v1 \
  --help
```

base image는 PyTorch 2.5.1, CUDA 12.4, cuDNN 9 runtime digest로 고정되어 있다. 로컬
Mac이 arm64여도 build script의 기본 target은 GKE용 `linux/amd64`다. image에는
ModernBERT 학습 코드와 runtime 검사만 포함하며 RouterBench CPU 라벨링 및 RouterArena
평가 코드는 포함하지 않는다.

## RouterArena 평가 전처리

RouterArena는 RouterBench 학습 데이터와 합치지 않고 일반화·강건성 평가 전용으로
전처리한다. 공식 revision과 checksum은
`ml/config/routerarena_preprocessing.json`에 고정돼 있다.

```bash
.venv/bin/routerarena-preprocessing download \
  --data-dir ml/data/raw/routerarena \
  --split all

.venv/bin/routerarena-preprocessing preprocess \
  --data-dir ml/data/raw/routerarena \
  --output-dir ml/artifacts/routerarena-evaluation-smoke-100 \
  --split sub_10 \
  --sample-size 100 \
  --seed 42 \
  --training-path ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet
```

`evaluation.parquet`에는 모델 입력, 난이도 정답과 평가 식별자만 들어간다. 원문
context/question/options/answer, 제외 사유와 robustness 원 질문은 `metadata.parquet`에
분리한다. 상세 계약은 `docs/routerarena-evaluation-preprocessing-spec.md`를 따른다.

## RouterArena routing 평가

RouterBench로 학습한 Hugging Face 호환 3-class checkpoint를 RouterArena에 실행하고
공식 `Difficulty`의 `easy/medium/hard`와 비교한다. 공개 Parquet에는 질문별 Bloom
6단계 컬럼이 없으므로 별도의 6→3 변환은 수행하지 않는다.

```bash
.venv/bin/pip install -e 'ml[model-evaluation]'

.venv/bin/routerarena-evaluate \
  --model-path /path/to/routerbench-modernbert-checkpoint \
  --evaluation-path ml/artifacts/routerarena-evaluation-full/evaluation.parquet \
  --output-dir ml/artifacts/routerarena-routing-evaluation
```

`predictions.parquet`, `metrics.json`, `model_info.json`을 생성하며 accuracy, macro-F1,
confusion matrix, under/over-routing, domain/source dataset별 결과와 truncation/latency를
기록한다. 예측 그룹별 RouterArena 라벨 비율과 시각화는 명세됐지만 아직 자동 생성
기능을 구현하지 않았다. 상세 계약은 `docs/routerarena-routing-evaluation-spec.md`를
따른다.
