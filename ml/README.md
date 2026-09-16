# RouterBench Labeling

RouterBench 0-shot/5-shot 배포본을 검증하고, 최소 충분 모델 성능 티어 기준으로 `easy`, `medium`, `hard` 라벨을 생성한다. 라벨 판정 근거는 `docs/ml/routerbench-data-labeling-spec.md`, 현행 ML 구조와 단일 체크리스트는 `docs/ml/workstream-checklist.md`를 따른다.

## 소스 구조

데이터 용도와 원본 데이터셋이 파일명에서 드러나도록 package를 나눈다.

```text
src/llm_router_labeling/
├── training/
│   ├── routerbench_labeling.py       # RouterBench 모델 성능 → 난이도 라벨
│   ├── routerbench_preprocessing.py  # 라벨 데이터 → 학습 Parquet
│   ├── routerbench_cli.py            # routerbench-labeling 명령
│   ├── feedback_retraining_preprocessing.py # 피드백 검증·canonical 변환·병합
│   ├── feedback_retraining_cli.py    # escalation-feedback-preprocessing 명령
│   ├── modernbert_training.py        # 환경 공통 학습 코어와 선택적 resume hook
│   ├── modernbert_checkpointing.py   # 완결 checkpoint·checksum·원자적 복제
│   ├── modernbert_training_cli.py    # Docker/Vertex용 modernbert-train 명령
│   ├── modernbert_colab_training.py  # Colab 로컬↔Drive 동기화·자동 재개
│   └── modernbert_colab_training_cli.py # modernbert-colab-train 명령
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

## Legacy v1 전체 데이터 labeling

아래 명령과 경로는 과거 v1 재현용이다. 신규 학습에는 다음 절의 v2 config를 사용한다.
MVP 핵심 eval 34,778행 전체에는 `label` 명령을 사용한다. `--output-dir`를 생략하면
dataset/tier/label rule version을 조합한 경로를 자동으로 사용하므로 서로 다른 규칙의
산출물을 덮어쓰지 않는다.

```bash
.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling.json \
  label \
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

### v2 균형 학습 데이터

GSM8K partial-credit 해석, unresolved 제외, 자연분포 validation/test와 클래스별
6,256행 train 선택을 적용하려면 v2 config를 명시한다.

```bash
.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling_v2.json \
  label --data-dir ml/data/raw --trust-pickle

.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling_v2.json \
  preprocess --profile full --seed 42
```

```text
ml/artifacts/routerbench-training/
  withmartian-routerbench-2024-03-27__minimum-sufficient-tier-v2__routerbench-training-preprocess-v2-balanced/
    training.parquet  # train 18,768 + validation 3,218 + test 3,218
    metadata.parquet
    metrics.json
```

Colab용 입력은 `./ml/scripts/build-modernbert-colab-balanced-v2-archive.sh`로 만들고
`ml/notebooks/modernbert_full_colab.ipynb`에서 새 Drive run directory로 학습한다.

### v3 solve-rate 실험 데이터

v2 결과에서 확인된 tier별 양자화 노이즈와 전역 균형화 손실을 비교하기 위한 별도
프로파일이다. 11개 모델 전체 solve rate에 RouterBench 내부 probe로 선택한
`easy >= 0.64`, `medium >= 0.27`, 그 미만 `hard`를 적용하고 train도 자연분포를
유지한다. RouterArena는 threshold 선택에 사용하지 않는다.

```bash
.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling_v3.json \
  label --data-dir ml/data/raw --trust-pickle

.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling_v3.json \
  preprocess --profile full --seed 42
```

현재 Colab 초기학습과 Vertex feedback 재학습의 production 입력은 v2로 유지한다. v3는
같은 ModernBERT 설정의 비교 학습과 Claude tier 표본 검증을 통과한 뒤 전환한다.

### Colab 초기학습과 Vertex feedback 재학습 bundle

아래 `build-full` 명령은 Colab 초기학습 입력 bundle을 만드는 공통 core다. 기존 KFP 초기
학습 pipeline도 같은 component를 포함하지만 compile regression용으로만 보존하며 실제
PipelineJob으로 제출하지 않는다.

```bash
.venv/bin/routerbench-labeling \
  --config ml/config/routerbench_labeling_v2.json \
  build-full \
  --zero-shot-path ml/data/raw/routerbench_0shot.pkl \
  --five-shot-path ml/data/raw/routerbench_5shot.pkl \
  --output-dir ml/artifacts/routerbench-kfp-v2-bundle \
  --seed 42 \
  --trust-pickle
```

bundle root에는 `training.parquet`, `metadata.parquet`, `metrics.json`,
`build_metrics.json`이 생기며 `labeling/` 아래에 감사용 labeled Parquet와 model tier를
보존한다. KFP IR은 다음 명령으로 컴파일한다.

```bash
.venv/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-v2
```

GCS에는 RouterBench 0/5-shot 원본, `routerbench_labeling_v2.json`,
`modernbert_t4_balanced_v2.json`, feedback 재학습용
`feedback_retraining_preprocessing.json`과 평가 config/data를 versioned object로 올린다.
초기 pipeline은 원본→v2 bundle을 만들고, feedback pipeline은 그 v2
`training.parquet`을 base로 import한다.

## ModernBERT 로컬 학습 smoke

로컬과 Vertex CustomJob은 같은 `modernbert-train` CLI를 사용한다. 차이는
데이터·출력 경로와 device/resource config뿐이다. 설계 근거와 계약은
`docs/ml/modernbert-training-spec.md`에 기록한다.

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

## Colab 무료 세션 전체 학습과 자동 재개

RouterBench 전체 baseline은 `ml/notebooks/modernbert_full_colab.ipynb`와 Colab 전용
`modernbert-colab-train`을 사용한다. loss, tokenizer와 optimizer 의미는
`modernbert-train`과 같은 공통 코어를 사용하고, 다음 세션 내구성 기능만 별도 계층에
둔다.

- Colab 임시 디스크에서 학습해 Google Drive 직접 I/O를 피한다.
- Colab 로컬 SSD에는 기본 50 optimizer step마다 model, optimizer, scheduler, FP16
  scaler, epoch/batch cursor와 Python/NumPy/PyTorch RNG 상태를 저장한다.
- Google Drive에는 500 optimizer step마다 동기화하며, 500 배수가 아니어도 각 epoch
  종료와 최종 step은 반드시 동기화한다.
- checkpoint를 `.partial-*`에 복사하고 모든 파일의 SHA-256을 검증한 뒤에만 Drive의
  `step-XXXXXXXX`로 승격한다.
- 새 세션은 `_SUCCESS`와 manifest가 있는 최신 checkpoint만 로컬로 내려받아 재개한다.
- 데이터·config·runtime·step limit fingerprint가 다르면 잘못된 재개를 거부한다.
- 최근 checkpoint 2개만 유지하고 최종 모델도 checksum manifest와 `_SUCCESS`로
  완결성을 표시한다.
- RouterBench validation split은 각 epoch의 약 1/3·2/3·3/3 optimizer step에서
  inference-only로 측정해 3 epoch 동안 총 9개 지점을 Drive에 저장한다.
- 3개 epoch가 각각 끝나면 같은 RouterArena full 8,375행을 inference-only로 평가하고
  predictions, metrics, predicted-group SVG와 epoch 변화 SVG를 Drive에 저장한다.
- RouterArena label은 loss, gradient, optimizer, scheduler, early stopping과 checkpoint
  선택에 전달되지 않는다. 최종 모델은 RouterBench에 정한 3 epoch의 마지막 모델이다.

전체 입력 archive를 만든 뒤 Colab `/content`에 업로드한다.

```bash
./ml/scripts/build-modernbert-colab-full-archive.sh
```

세션이 종료되면 새 GPU server에서 같은 archive와 notebook을 처음부터 실행한다. 같은
Google Drive run directory를 사용하면 `--resume-mode auto`가 자동으로 이어서 학습한다.
config나 데이터를 바꾸는 실험은 기존 경로를 재사용하지 않고 새 run directory를 쓴다.

epoch별 결과는 다음 위치에 생성된다.

```text
<Drive run>/routerbench-validation-progress/
├── observation-01..09/
│   ├── metrics.json
│   ├── usage_contract.json
│   ├── artifact_manifest.json
│   └── _SUCCESS
├── validation_progress.json
└── validation_progress.svg
```

현재 전체 설정은 epoch당 3,476 optimizer step이므로 관찰 지점은 global step
1,159/2,317/3,476, 4,635/5,793/6,952, 8,111/9,269/10,428이다. 중간 검증은
accuracy·macro-F1·loss를 기록하지만 early stopping이나 best checkpoint 선택에는 쓰지
않는다.

```text
<Drive run>/routerarena-epoch-evaluations/
├── epoch-01/
│   ├── predictions.parquet
│   ├── metrics.json
│   ├── usage_contract.json
│   ├── predicted_group_distribution.svg
│   └── predicted_group_heatmap.svg
├── epoch-02/
├── epoch-03/
├── epoch_progress.json
└── epoch_progress.svg
```

`epoch_progress.svg`는 accuracy, macro-F1, under-routing rate의 변화를 함께 보여준다.
평가 중 세션이 끊기면 epoch 종료 checkpoint와 `_SUCCESS`가 있는 평가 artifact를 대조해
누락된 epoch 평가부터 복구한다. 이 곡선을 보고 epoch 수나 hyperparameter를 선택하면
RouterArena가 사실상 개발 데이터가 되므로, 현재 baseline에서는 관찰·보고에만 사용한다.

## ModernBERT 학습 Docker image

Vertex AI Custom Training에서는 Colab과 같은 `modernbert-train`을 entrypoint로 제공하는 T4용
linux/amd64 image를 사용한다. 데이터와 checkpoint는 image에 넣지 않고 Artifact로
마운트한다.

```bash
./ml/scripts/build-modernbert-training-image.sh

docker run --rm \
  llm-router-modernbert-training:cu124-v1 \
  --help
```

base image는 PyTorch 2.5.1, CUDA 12.4, cuDNN 9 runtime digest로 고정되어 있다. 로컬
Mac이 arm64여도 build script의 기본 target은 Vertex용 `linux/amd64`다. image에는
ModernBERT 학습 코드와 runtime 검사만 포함하며 RouterBench CPU 라벨링 및 RouterArena
평가 코드는 포함하지 않는다.

## Vertex AI Pipelines v2 데이터·학습 workflow

KFP SDK는 `2.17.0`, Google Cloud Pipeline Components는 `2.22.0`으로 고정한다.
파이프라인은 KFP v2 IR로 컴파일해 관리형 Vertex AI Pipelines에서 실행한다. 일반
component는 CPU이고, 학습 component만 기존 GPU image의 `modernbert-train`을 Vertex AI
CustomJob으로 실행한다.

```bash
python3.12 -m venv .venv-kubeflow
.venv-kubeflow/bin/pip install -r ml/requirements-kubeflow.txt

.venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-v2
```

production IR은 GPU image digest를 compile 시점에 고정한다.

```bash
LLM_ROUTER_VERTEX_TRAINING_IMAGE="us-west1-docker.pkg.dev/<project>/<repo>/llm-router-modernbert-training@sha256:<digest>" \
  .venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-v2
```

생성되는 IR YAML은 Git에서 제외되는 로컬 artifact다.

```text
feedback-prepare-component.yaml
feedback-merge-component.yaml
routerbench-v2-preprocessing-component.yaml
vertex-modernbert-training-job-component.yaml
initial-training-pipeline.yaml
feedback-retraining-pipeline.yaml
```

`initial-training-pipeline.yaml`은 회귀 검사용 역사적 IR이다. production에 게시·제출하는
학습 IR은 `feedback-retraining-pipeline.yaml`뿐이다.

component 계약:

```text
Input[Dataset] training_dataset    # v2 bundle 또는 feedback merge bundle
Input[Artifact] training_config    # modernbert_t4_balanced_v2.json 또는 smoke config
str training_path_suffix           # direct file은 "", bundle은 "/training.parquet"
Output[Model] trained_model        # checkpoint, tokenizer, metrics.json
Output[str] gcp_resources          # Vertex CustomJob resource 추적 정보
```

초기학습은 Colab notebook이 검증된 bundle의 `training.parquet`와 config를 읽어 실행하고
Drive checkpoint를 원자적으로 동기화한다. 재학습 pipeline은 versioned v2 base artifact와
feedback manifest/events/config를 가져와
`prepare(CPU) → merge(CPU) → train`으로 실행하며 기본 학습 설정은
`modernbert_t4_balanced_v2.json`이다.

학습은 Google 공식 CustomJob wrapper를 사용하며 worker pool은 `n1-standard-4`,
`NVIDIA_TESLA_T4=1`, replica 1, restart false, timeout 6시간이다. KFP task retry를 설정하지
않아 학습 재시도는 0회다. IR에는 `nvidia.com/gpu`가 없고 T4 사양은 CustomJob
`worker_pool_specs`에만 존재한다. 라벨링·전처리·feedback prepare/merge·RouterArena
평가·package·gate에는 accelerator를 요청하지 않는다.

기본 image URI의 `replace-project`는 실제 GCP project와 immutable digest URI로 컴파일
전에 교체해야 한다. dataset과 config URI도 내용을 덮어쓰지 않는 immutable version
경로여야 KFP cache와 Vertex ML Metadata 계보를 안전하게 사용할 수 있다.

## CPU 데이터 처리 Docker image

초기 RouterBench 라벨링·학습 전처리, RouterArena 평가 데이터 전처리와 운영
에스컬레이션 피드백의 검증·병합은 GPU 학습 image와 분리한 공통 CPU 전용 image에서
실행한다. 이 image에는 PyTorch, Transformers, checkpoint 추론용
`routerarena-evaluate`를 넣지 않는다.

```bash
./ml/scripts/build-data-processing-image.sh

docker run --rm \
  llm-router-data-processing:py312-v2 \
  routerbench-labeling --help

docker run --rm \
  llm-router-data-processing:py312-v2 \
  routerarena-preprocessing --help

docker run --rm \
  llm-router-data-processing:py312-v2 \
  escalation-feedback-preprocessing --help
```

기본 target은 Vertex managed runtime과 맞춘 `linux/amd64`다. 데이터는 image layer에 넣지 않고
Kubeflow Dataset Artifact 또는 로컬 read-only volume으로 전달한다. container는
UID/GID `10001`의 non-root 사용자로 실행되며 출력 경로만 쓰기 가능하게 제공해야 한다.

RouterArena의 checkpoint 추론·metric 계산은 PyTorch/Transformers가 필요한 별도 평가
component로 유지한다. 따라서 이 CPU 데이터 image의 RouterArena 범위는 공식 데이터
검증, 입력 조립, 중복·누수 제거와 evaluation Parquet 생성까지다.

에스컬레이션 피드백은 두 단계로 처리한다.

```bash
escalation-feedback-preprocessing prepare \
  --manifest-path /input/manifest.json \
  --events-path /input/events.parquet \
  --output-dir /output/feedback

escalation-feedback-preprocessing merge \
  --base-training-path /input/routerbench/training.parquet \
  --feedback-training-path /input/feedback/feedback_training.parquet \
  --output-dir /output/retraining
```

로컬 통합 검증에서는 `build` 명령에 manifest, events와 base training 경로를 한 번에
전달할 수 있다. Kubeflow에서는 캐시·재시도·계보를 단계별로 남기기 위해 `prepare`와
`merge`를 반드시 별도 component로 실행한다.

`prepare`는 schema version, checksum, 행 수와 정정 라벨 채택 조건을 검사하고
`feedback_training.parquet`, 전체 event 감사용 metadata, metrics와 manifest를 만든다.
`merge`는 validation/test와 겹치는 피드백을 제외하고, train의 정확히 같은 질문에 더
높은 피드백 라벨이 있으면 정확히 겹치는 기존 train 행만 교체하고 duplicate group은
이어받는다. 실제 운영 schema 합의 전에는
`ml/tests/fixtures/feedback_v0_1`의 합성 batch로 이 계약을 검증한다.

Kubeflow CPU component 계약은 다음과 같다. 두 component는 동일한
`llm-router-data-processing:py312-v2` image를 사용하지만 서로 다른 command와 resource,
cache 단위를 갖는다.

```text
feedback_prepare_component
  Input[Artifact] feedback_manifest
  Input[Dataset]  feedback_events
  Input[Artifact] preprocessing_config
  Output[Dataset] prepared_feedback/
    ├── feedback_training.parquet
    ├── feedback_metadata.parquet
    ├── metrics.json
    └── manifest.json

feedback_merge_component
  Input[Dataset]  base_training_dataset        # direct training.parquet
  Input[Dataset]  prepared_feedback/           # 위 bundle
  Input[Artifact] preprocessing_config
  Output[Dataset] merged_training_dataset/
    ├── training.parquet
    ├── merge_metadata.parquet
    ├── metrics.json
    └── manifest.json
```

`prepare`는 CPU request/limit `1/2`, memory `2Gi/4Gi`, `merge`는 CPU `2/4`, memory
`4Gi/8Gi`로 시작한다. 두 task 모두 retry 2회와 cache를 사용하며 GPU resource를 요청하지
않는다. 재학습 GPU component에는 merged bundle의 `/training.parquet` suffix를 명시해
기존 GPU image의 파일 기반 CLI 계약을 그대로 유지한다.

v1 이미지의 2026-08-08 로컬 smoke 기준선:

| 항목 | 결과 |
| --- | --- |
| image tag | `llm-router-routerbench-routerarena-data:py312-v1` |
| target | `linux/amd64` |
| image ID | `sha256:34b043ea2007a4e6f67df6362f2537754b1eae3ddda0405d18b847b176a8307d` |
| build-time CLI 검사 | `routerbench-labeling`, `routerarena-preprocessing` 통과 |
| RouterBench smoke | 기존 labeled 100행 → train 80 / validation 10 / test 10 |
| RouterArena smoke | `sub_10`에서 3행 evaluation artifact 생성 |

Apple Silicon에서는 `linux/amd64` emulation 경고가 출력되지만 두 command 모두 exit
code 0으로 완료됐다. 실제 데이터와 생성 artifact는 image layer에 포함하지 않았다.

v2 이미지의 2026-08-08 구현 결과:

| 항목 | 결과 |
| --- | --- |
| image tag | `llm-router-data-processing:py312-v2` |
| target | `linux/amd64` |
| image ID | `sha256:ddfabf9d54934999e1f88676590f3967a2e9c5ab6e615e39e468816868d3bdb7` |
| build-time CLI 검사 | RouterBench, RouterArena, feedback 3개 명령 통과 |
| feedback prepare | 합성 8 event → 채택 3 event → 학습 2행 |
| full merge | RouterBench 34,764행 + feedback 2행 → 34,766행 |
| holdout 보존 | validation 3,478 / test 3,478 유지 |

v2 Docker 통합 smoke는 2026-08-10에 `linux/amd64` image로 통과했다.

| 흐름 | 결과 |
| --- | --- |
| RouterBench preprocess | 100행 → train 80 / validation 10 / test 10 |
| RouterArena preprocess | `sub_10` 809행 중 seed 42로 3행 artifact 생성 |
| feedback prepare | 8 event → 채택 3 event → 학습 2행 |
| feedback merge | RouterBench smoke 100행 + feedback 2행 → 102행 |
| holdout 보존 | validation 10 / test 10 유지 |

네 명령은 모두 non-root UID/GID `10001`로 실행했고, 입력은 read-only
volume, 출력은 별도 writable volume으로 전달했다. 실제 운영 feedback schema는
온라인 팀 합의 전이므로 합성 `feedback-export-v0.1` fixture 결과만 검증한다.

## Docker image 재빌드·재배포

4개 image는 image tag의 `v1`/`v2` 숫자가 코드 최신 여부를 보장하지 않는다. Docker
build는 그 순간 디스크에 있는 Dockerfile과 `COPY` 대상 파일을 그대로 읽어 image를
만들 뿐, tag 이름은 사람이 붙인 라벨일 뿐이다. `ml/src/llm_router_labeling/` 아래
파일을 고쳤다면, 그 파일을 실제로 쓰는 image의 Dockerfile을 다시 build·push해야
반영된다.

### 어떤 파일을 고치면 어떤 image를 다시 만들어야 하는가

| 고친 위치 | 다시 빌드할 image | Dockerfile |
| --- | --- | --- |
| `training/routerbench_*.py`, `training/feedback_retraining_*.py`, `evaluation/routerarena_preprocessing.py`, `evaluation/routerarena_cli.py` | `llm-router-data-processing` | `ml/docker/data-processing.Dockerfile` |
| `training/modernbert_training*.py`, `training/modernbert_checkpointing.py`, `training/modernbert_runtime.py` | `llm-router-modernbert-training` | `ml/docker/modernbert-training.Dockerfile` |
| `evaluation/routerarena_evaluation_cli.py`, `evaluation/routerarena_routing_evaluator.py`, `evaluation/routerarena_visualization.py` | `llm-router-modernbert-evaluation` | `ml/docker/modernbert-evaluation.Dockerfile` |
| `deployment/*.py`(`model_package*`, `deploy_gate*`, `model_publish*`), `shared/object_store.py` | `llm-router-model-promotion` | `ml/docker/model-promotion.Dockerfile` |
| `shared/dataset_io.py`, `shared/text_normalization.py` | 이 파일을 쓰는 image 전부(위 표에서 해당 파일이 나온 image) | 해당 Dockerfile들 |

새 파일을 추가했다면(오늘의 `model_publish.py`, `model_publish_cli.py`,
`shared/object_store.py`처럼) **파일을 옮기는 것만으로는 반영되지 않는다.**
Dockerfile에 `COPY` 줄을 새로 추가해야 하고, 새 CLI 진입점을 추가했다면
`ml/docker/<image>.pyproject.toml`의 `[project.scripts]`에도 등록해야 한다. 새
런타임 의존성(`google-cloud-storage`처럼)이 필요하면 Dockerfile의 `RUN pip
install` 줄에도 추가한다.

### build 명령

**build context는 반드시 저장소 루트(`.`)다.** 아래처럼 저장소 루트에서 실행한다.

```bash
docker buildx build --platform linux/amd64 \
  -f ml/docker/model-promotion.Dockerfile \
  -t "<registry>/<repo>/llm-router-model-promotion:py312-v1" --push .
```

`-f`로 넘긴 Dockerfile의 모든 `COPY` 경로가 `ml/src/...`, `ml/config/...`처럼
저장소 루트를 기준으로 적혀 있기 때문이다. context를 `ml`로 주면(흔한 실수)
Docker가 `ml/ml/src/...`를 찾다가 다음처럼 실패한다.

```text
ERROR: failed to build: failed to solve: failed to compute cache key:
failed to calculate checksum of ref ...: "/ml/src/llm_router_labeling/.../<file>.py": not found
```

이 에러가 나면 build context를 확인한다. 로컬에서 결과만 확인하고 싶으면
`--push` 대신 `--load`를 쓴다.

### push 후 확인할 것

1. **build-time smoke test 통과 여부.** 각 Dockerfile은 `RUN` 단계에서 해당 image의
   CLI `--help`를 실행해 entry point가 살아있는지 검증한다(예: `modernbert-publish
   --help`). 이게 실패하면 push 전에 이미 build가 멈춘다.
2. **digest 확보.**
   ```bash
   gcloud artifacts docker images list "<registry>/<repo>" --include-tags --format="table(package,tags,version)"
   ```
   `version` 열의 `sha256:...` 값을 기록한다. **tag가 아니라 digest를 써야** 어떤
   코드가 실제로 도는지 재현 가능하다.
3. **KFP IR 재컴파일.** image를 다시 push했다면 그걸로 끝이 아니다. 컴파일된
   pipeline IR은 build 시점 digest를 값으로 박아 넣으므로, 새 digest로
   `LLM_ROUTER_VERTEX_*_IMAGE` 환경변수를 다시 설정하고
   `python -m ml.kubeflow.compile_training_workflows`를 다시 실행해 GCS의 IR도
   갱신해야 한다. 자세한 환경변수 목록과 명령은
   `docs/ml/gcp-rollout-and-serving-integration-plan.md`의 G-3/G-7절을 참고한다.

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
분리한다. 상세 계약은 `docs/ml/routerarena-evaluation-preprocessing-spec.md`를 따른다.

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
기록한다. 또한 예측 easy/medium/hard 그룹별 RouterArena 공식 라벨 count·비율·전체
분포 대비 %p 차이와 `predicted_group_distribution.svg`,
`predicted_group_heatmap.svg`를 자동 생성한다. 상세 계약은
`docs/ml/routerarena-routing-evaluation-spec.md`를 따른다.

## 모델 패키지와 deploy gate

학습 checkpoint에 평가·학습 metadata를 결합한 `model-package-v1` manifest를 만들고
필수 파일, `easy=0/medium=1/hard=2` 순서와 파일별 SHA-256을 검증한다.

```bash
.venv/bin/modernbert-package create \
  --package-dir /path/to/checkpoint \
  --metadata /path/to/package-metadata.json

.venv/bin/modernbert-package validate \
  --package-dir /path/to/checkpoint
```

`modernbert-deploy-gate`는 후보 metric의 절대 최소·최대 기준과 기존 승인 모델 대비
delta를 규칙 JSON으로 검사한다. 실패 시 결과 JSON을 먼저 저장하고 exit code 2로
종료하므로 Kubeflow 승격 단계를 차단할 수 있다. 실제 metric 임계값과 온라인 배포
호출은 기준 모델 및 백엔드 계약이 확정된 뒤 설정한다.

최초 모델처럼 비교할 기존 승인 모델이 없다면 gate 규칙의 `minimum_delta`와
`maximum_delta`를 비우고, `baseline_metrics_uri`에는 빈 JSON 객체 파일을 전달한다.
이 경우 절대 최소·최대 기준과 package 무결성만 검사한다.

Kubeflow 실행 이미지는 역할별로 분리한다.

| image | 자원 | 역할 | 2026-08-10 로컬 build |
| --- | --- | --- | --- |
| `llm-router-data-processing:py312-v2` | CPU | 초기 데이터와 feedback prepare/merge | 통과 |
| `llm-router-modernbert-training:cu124-v1` | GPU | ModernBERT 학습·checkpoint 저장 | 통과 |
| `llm-router-modernbert-evaluation:cu124-v1` | GPU | RouterArena 추론·metric·SVG | `c854108b...`, 통과 |
| `llm-router-model-promotion:py312-v1` | CPU | package/checksum/deploy gate | `66f21b23...`, 통과 |

평가와 승격 두 image는 모두 `linux/amd64`, UID/GID `10001`로 빌드했으며 컨테이너
내부 CLI smoke를 통과했다. Colab 초기 checkpoint는 독립
`model_qualification_pipeline`으로 RouterArena 평가 → candidate package → deploy gate를
수행한다. feedback 재학습 pipeline은 재학습 뒤 같은 단계를 직접 실행한다.

이 gate는 온라인 서비스 프로세스에 모델을 직접 밀어 넣지 않는다. pipeline은 gate를
통과한 package만 GCS candidate 경로에 게시하고, 같은 pipeline의 다음 component가 이를
재검증해 `approved/`로 승격하고 `CURRENT.json`을 원자적으로 갱신한다. FastAPI 반영은
아래의 pull 방식 전달 경로를 따른다.

## 승인 모델 게시와 FastAPI 전달

FastAPI로 모델을 밀어 넣는 API는 없다. pipeline이 GCS에 게시·승격하고, FastAPI가
`CURRENT.json`을 polling해 새 version을 내려받는 pull 구조다.

```text
deploy gate 통과
  -> model_publish_component: gs://<bucket>/models/router-modernbert/<version>/candidate/
  -> model_auto_approve_component: manifest·checksum 재검증
  -> approved/ 복사 + gate_status=approved + CURRENT.json 원자 갱신
  -> FastAPI pointer watcher: 다운로드·검증 -> 요청 중단 없는 model swap
```

`model_publish_component`는 gate 뒤에 실행되며 caching을 끈다. gate가 실패하면 candidate
경로에 아무것도 올라가지 않는다. 자동 승인 component 역시 caching을 끄며, 바로 앞
publish 결과의 version과 candidate URI가 정확히 일치하는지 확인한 다음 candidate를
다시 내려받아 manifest·checksum·라벨 순서를 검증한다. 검증이 끝난 package만
`approved/`에 올라가고 `CURRENT.json`은 GCS generation precondition으로 갱신된다.
candidate 사본은 감사와 rollback 분석을 위해 그대로 둔다.

자동 반영을 사용하더라도 수동 rollback 명령은 장애 복구 수단으로 유지한다.

```bash
.venv/bin/modernbert-publish show-current --root-uri gs://<bucket>/models/router-modernbert
.venv/bin/modernbert-publish rollback \
  --root-uri gs://<bucket>/models/router-modernbert --approved-by <승인자>
```

GCS 없이 동작을 확인하려면 `--local-store-root`로 파일시스템 store를 쓴다. 이 CLI는
`google-cloud-storage`가 필요하므로 `pip install -e 'ml[model-publishing]'` 또는 승격
image를 사용한다.

FastAPI 쪽 기본 계약은 `LLM_Router-serving`의
`LLM_ROUTER_ROUTER_MODEL_CURRENT_POINTER_URI`다. 이 값을 생략하면
`LLM_ROUTER_MODEL_PUBLISH_ROOT_URI/CURRENT.json`을 사용한다. 시작 시와 이후 기본 60초
간격으로 pointer를 읽고 package의 schema version, 라벨 순서, `model_version`, manifest
SHA-256과 파일별 SHA-256을 검증한다. `gate_status`가 `approved`가 아니면 적재하지
않는다. 새 model을 완전히 적재한 뒤 한 번에 참조를 교체하므로 기존 요청은 기존 model로
완료되고, 다운로드·검증 실패 시에는 마지막 정상 model을 계속 제공한다.
