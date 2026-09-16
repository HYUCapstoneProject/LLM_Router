# ModernBERT 학습 명세와 로컬·Colab·Vertex AI 실행 결정

기준일: 2026-08-14
상태: 승인
관련 브랜치: `feat/data-labeling-training`

> **2026-09-16 실행 경계 보정:** ModernBERT 최초 전체 학습은 Colab GPU에서 수행한다.
> 공통 학습 core/CLI와 checkpoint 계약은 유지하고, periodic checkpoint·Google Drive
> 원자 동기화·자동 재개는 `modernbert-colab-train` wrapper가 담당한다. 관리형 Vertex AI
> Pipelines와 T4 CustomJob은 **feedback 재학습부터** 사용한다. 기존 initial-training IR은
> compile regression용으로 보존하되 실제 PipelineJob으로 제출하지 않는다. GKE와 자체
> 설치 Kubeflow는 사용하지 않는다.

## 1. 결정이 필요했던 이유

Vertex AI resource가 준비되기 전에 ModernBERT 학습 경로를 검증해야 한다. 이때
로컬 smoke 코드를 임시 구현한 뒤 클라우드용 학습 코드를 다시 만들면 두 환경의
전처리, 라벨 순서, loss와 checkpoint 형식이 달라질 위험이 있다.

핵심 질문은 다음과 같다.

> 로컬 smoke, Colab 초기학습과 Vertex AI feedback 재학습 코드를 별도로 만들 것인가,
> 아니면 같은 학습 코어를 실행 환경만 바꿔 재사용할 것인가?

## 2. 결정

**모든 환경이 하나의 학습 코어를 공유한다.** 로컬 smoke와 feedback 재학습용 Vertex
container는 `modernbert-train`을 사용한다. Colab 초기 전체 학습은 같은 코어를 호출하는
`modernbert-colab-train`을 사용하며, 세션 내구성에 필요한 checkpoint·Drive 동기화만
추가한다. tokenizer, loss, optimizer와 metric을 Colab용으로 다시 구현하지 않는다.

```text
공통 학습 코어
  ├─ config 및 Parquet 계약 검증
  ├─ tokenizer와 truncation
  ├─ sample-weighted classification loss
  ├─ train/validation metric
  ├─ save_pretrained checkpoint
  └─ checkpoint 재로딩 및 label 계약 검증
       ↑                         ↑                         ↑
  로컬 smoke CLI          Colab 초기학습 wrapper       feedback 재학습 CustomJob
  로컬 파일·CPU/MPS       임시 disk↔Drive·T4          GCS URI·T4 CUDA
                                                           ↑
                                             Vertex AI PipelineJob
```

CustomJob wrapper 안에는 tokenizer, loss, 학습 loop 또는 metric을 다시 구현하지 않는다.
Vertex AI Pipelines는 feedback 검증·merge, Artifact 전달, step caching, 재학습 CustomJob
제출·감시, GCS 저장과 Vertex ML Metadata 계보를 담당한다. GPU 자원은 feedback 재학습
CustomJob worker pool에만 요청한다.

## 3. 판단 근거

- 로컬에서 검증한 데이터·loss·checkpoint 계약을 Vertex에서도 그대로 사용해 환경 간
  동작 차이를 줄일 수 있다.
- 버그 수정과 실험 설정 변경 지점이 한 곳이므로 유지보수와 재현성이 좋아진다.
- Vertex AI 없이도 동일 CLI를 실행할 수 있어 인프라 장애가 모델 코드 검증을 막지 않는다.
- 파일 기반 입출력은 로컬 경로와 KFP Artifact 경로를 동일하게 다룰 수 있다.
- 로컬과 클라우드의 차이는 코드가 아니라 config의 데이터 규모, batch, precision,
  device와 출력 경로로 표현할 수 있다.

검토했지만 선택하지 않은 방식:

| 대안 | 선택하지 않은 이유 |
| --- | --- |
| 로컬 notebook과 Vertex 학습 코드를 각각 작성 | 학습 의미론이 중복되고 두 구현이 쉽게 달라진다. |
| Google Drive 경로에서 model을 직접 학습·저장 | Drive 지연·I/O 오류가 학습 loop에 영향을 주고 미완성 checkpoint를 정상 파일로 오인할 수 있다. |
| 학습 코드에서 GCS를 직접 읽고 쓴다 | 로컬 테스트가 어려워지고 저장소 책임이 학습 코어에 섞인다. |
| 학습 CustomJob이 split과 label을 다시 계산한다 | 이미 승인한 versioned 데이터와 다른 split이 생길 수 있다. |

## 4. 고정 모델과 라벨 계약

- base model: `answerdotai/ModernBERT-base`
- model revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`
- task: single-label, 3-class sequence classification
- `label2id`: `easy=0`, `medium=1`, `hard=2`
- `id2label`: `0=easy`, `1=medium`, `2=hard`
- smoke tokenizer max length: 512
- T4 full tokenizer max length: 8,192
- 긴 입력은 오른쪽을 자르는 tokenizer 기본 truncation을 사용하고 잘린 행 수를 metric에
  기록한다.

ModernBERT 공식 모델 카드는 base가 149M 파라미터이고 최대 8,192 token 문맥을
지원한다고 설명한다. smoke의 목적은 긴 문맥 성능 측정이 아니라 end-to-end 학습
경로 검증이므로 초기 로컬 실행은 512 token으로 제한한다. 전체 T4 baseline은
`modernbert_t4_balanced_v2.json`에서 8,192 token, batch 1, gradient accumulation 8,
gradient checkpointing과 FP16을 사용한다. 100행 Colab smoke에서 OOM이면 model이나
학습 코드를 바꾸지 않고 batch/길이 config version을 올려 조정한다.

참고:

- [ModernBERT-base 공식 모델 카드](https://huggingface.co/answerdotai/ModernBERT-base)
- [Transformers ModernBERT 공식 문서](https://huggingface.co/docs/transformers/model_doc/modernbert)

## 5. 공통 CLI 계약

```bash
modernbert-train \
  --config <training-config.json> \
  --runtime-contract <runtime-contract.json> \
  --training-path <training.parquet> \
  --output-dir <checkpoint-output-directory>
```

로컬 smoke, Colab 초기학습과 Vertex feedback 재학습은 위 세 입력의 의미가 같다.

| 항목 | 로컬 smoke | Colab 초기학습 | Vertex feedback 재학습 |
| --- | --- | --- | --- |
| training path | 저장소의 로컬 Parquet | 업로드한 immutable Parquet | 입력 Artifact가 마운트된 경로 |
| output directory | `ml/artifacts/` 하위 | 임시 SSD + Drive 동기화 | 출력 Artifact 경로 |
| device | auto CPU/MPS/CUDA | CUDA T4 등 할당 GPU | CUDA T4 |
| precision | FP32 | FP16 | FP16 |
| max length | 512 | production config 값 | production config 값 |
| 데이터 규모 | smoke 100행 | 승인된 초기 전체 split | base rehearsal + feedback |
| 실행 관리 | 사용자가 CLI 실행 | notebook + resume wrapper | component가 같은 CLI 실행 |

학습 코어는 GCS bucket, Kubeflow run ID 또는 Kubernetes API를 알지 못해야 한다.

### 5-1. Colab 전용 실행 계약

```bash
modernbert-colab-train \
  --config ml/config/modernbert_t4_balanced_v2.json \
  --runtime-contract ml/config/modernbert_runtime_cu124_v1.json \
  --training-path <training.parquet> \
  --local-run-dir /content/llm-router-training/modernbert-router-balanced-v2 \
  --persistent-run-dir /content/drive/MyDrive/LLM_Router/training/modernbert-router-balanced-v2 \
  --resume-mode auto \
  --save-every-optimizer-steps 50 \
  --drive-sync-every-optimizer-steps 500 \
  --routerbench-validations-per-epoch 3 \
  --keep-last-checkpoints 2
```

Colab 전용 wrapper와 checkpoint 계약은 다음과 같다.

1. 학습은 Colab 임시 SSD에서 실행한다.
2. optimizer step 경계에서만 resume checkpoint를 만든다. 따라서 미저장 gradient를
   복구해야 하는 중간 accumulation 상태가 생기지 않는다.
3. model/tokenizer뿐 아니라 optimizer, scheduler, AMP scaler, 다음 epoch·batch cursor,
   누적 loss·시간과 Python/NumPy/PyTorch/CUDA RNG 상태를 저장한다.
4. 로컬 임시 SSD에는 50 optimizer step마다 checkpoint를 완성하고 manifest의
   size·SHA-256을 검증한다.
5. Drive에는 500 optimizer step마다 `.partial-<uuid>`로 복사한 뒤 재검증하고
   `step-XXXXXXXX`로 원자적으로
   rename한다. `_SUCCESS` 없는 경로는 재개 후보에서 제외한다.
6. 새 세션은 가장 높은 optimizer step의 정상 checkpoint를 임시 SSD로 복사하고
   data SHA-256, training config, runtime version과 max-step fingerprint가 같을 때만
   재개한다.
7. 기본 보존 수는 최근 2개다. 최종 model에도 manifest와 `_SUCCESS`를 남기며 이미
   완료된 run은 다시 학습하지 않는다.
8. 각 epoch 종료와 최종 optimizer step에는 500 step 주기와 무관하게 Drive checkpoint를
   남기고, epoch checkpoint가 봉인된 뒤 RouterArena inference-only 관찰을 실행한다.
9. RouterBench validation은 각 epoch의 약 33%·66%·100% optimizer step에서 먼저
   실행하고 작은 JSON·SVG artifact를 Drive에 저장한 뒤 해당 step checkpoint를 만든다.
   검증 도중 세션이 종료되면 이전 checkpoint에서 재개해 같은 관찰 지점을 다시 통과한다.

`gradient_checkpointing=true`는 GPU memory 절약 기능이고 위 resume checkpoint와 다른
개념이다. 같은 Colab 세션에서 프로세스만 다시 시작하면 최대 49 optimizer step 전의
로컬 checkpoint를 사용할 수 있다. Colab VM 자체가 종료되면 마지막 Drive 동기화 이후
최대 499 optimizer step을 다시 수행할 수 있다. Drive checkpoint 한 개가 model과 Adam
상태를 포함해 크므로 50 step마다 원격 복사하지 않는다.

v2 균형 train은 각 클래스 6,256행, 총 18,768행이다. T4 micro-batch 1과 gradient
accumulation 8을 사용하므로 한 optimizer 단위에는 세 클래스가 `3/3/2`로 포함된다.
추가 2자리를 받는 클래스는 단위마다 순환하고, 각 클래스 내부 행 순서는 epoch seed로
다시 섞는다. 따라서 epoch 전체는 각 클래스가 정확히 6,256회 나타나며 어떤 완전한
optimizer 단위도 특정 클래스가 빠지지 않는다.

confidence sample weight는 micro-batch별 평균으로 나누지 않는다. batch 1에서 그렇게
나누면 weight가 정확히 상쇄되기 때문이다. 각 행의 weighted loss 합을 train 전체 평균
weight와 해당 accumulation window의 표본 수로 정규화해 optimizer 단위의 gradient에
weight가 실제로 반영되도록 한다.

#### 5-1-1. epoch 내부 RouterBench validation 관찰 계약

고정된 RouterBench validation split을 epoch마다 세 번 추론해 총 9개 학습 곡선 지점을
남긴다. v2 epoch당 optimizer step은 2,346이며 global step은 다음과 같다.

```text
epoch 1: 782 / 1,564 / 2,346
epoch 2: 3,128 / 3,910 / 4,692
epoch 3: 5,474 / 6,256 / 7,038
```

- 출력: 관찰별 `metrics.json`, `usage_contract.json`, checksum manifest와 `_SUCCESS`
- 요약: `validation_progress.json`, accuracy·macro-F1 변화 `validation_progress.svg`
- loss 비교: 직전 관찰 이후 train 행의 unweighted/weighted mean cross-entropy와
  validation의 unweighted/weighted mean cross-entropy를 같은 관찰 JSON과 stdout에
  기록한다. 직접 비교용 unweighted train/validation 곡선은 `loss_progress.svg`로 만든다.
- 저장: `<Drive run>/routerbench-validation-progress/observation-01..09/`
- 격리: inference mode로 실행하고 Python·NumPy·PyTorch·CUDA RNG를 복구한다.
- 의사결정 금지: metric은 gradient·optimizer·scheduler·early stopping·checkpoint 선택과
  최종 epoch 결정에 사용하지 않는다. 최종 100% 관찰값은 최종 `metrics.json`에 재사용해
  같은 validation 전체 추론을 중복하지 않는다.

#### 5-1-2. epoch별 RouterArena 관찰 계약

전체 3 epoch 각각이 끝나면 RouterArena full 8,375행을 동일한 설정으로 추론한다. 이
단계는 모델이 epoch에 따라 어떻게 변하는지 보여주는 진단 단계이며 학습 검증 split을
대체하지 않는다.

- 입력: RouterArena `evaluation.parquet`과 선택적 `metadata.parquet`
- Colab T4 batch: RouterArena에는 최대 54,601자 입력이 있으므로 15GB GPU의 OOM을 피하기
  위해 `--routerarena-batch-size 1`을 사용한다. 학습 batch도 1이지만 gradient
  accumulation 8로 effective batch 8을 유지한다.
- 출력: epoch별 `predictions.parquet`, `metrics.json`, model info, predicted-group 누적 막대,
  heatmap, 세 epoch accuracy·macro-F1·under-routing 변화 SVG
- 저장: `<Drive run>/routerarena-epoch-evaluations/epoch-01..03/`
- 내구성: epoch checkpoint를 먼저 Drive에 저장하고 평가한다. 평가 도중 세션이 종료되면
  새 세션이 checkpoint cursor와 평가 `_SUCCESS`를 비교해 누락된 평가를 다시 수행한다.
- 격리: RouterArena label과 metric은 loss, gradient, optimizer, scheduler, early stopping,
  checkpoint 선택 및 최종 epoch 결정으로 반환되지 않는다.
- 난수 격리: 관찰 전 Python·NumPy·PyTorch·CUDA RNG를 저장하고 관찰 뒤 복구한다.

최종 모델은 사전에 고정한 RouterBench 3 epoch 학습의 마지막 모델이다. 사람이 이 곡선을
보고 epoch 수·learning rate·기타 hyperparameter를 변경하면 RouterArena가 사실상 개발
데이터가 되어 독립 외부평가 의미가 약해진다. 그러한 튜닝을 하려면 별도 최종 holdout을
확보하고 평가 명세를 새 버전으로 승인해야 한다.

공식 Vertex 구조에서 RouterArena 평가는 Vertex AI Pipelines CPU component의 책임이다.
이번 Colab baseline만 이미 활성화된 학습 GPU를 epoch 사이에 재사용해 별도 GPU 자원을
추가로 만들지 않는다.

### 5-2. runtime 재현성 계약

`modernbert_runtime_cu124_v1.json`은 다음 핵심 버전을 고정한다.

```text
PyTorch 2.5.1 / CUDA 12.4 / Transformers 4.48.3
NumPy 2.1.2 / Pandas 2.3.3 / PyArrow 21.0.0
Python minor 3.11 또는 3.12
```

Docker/Vertex CustomJob이 기준 runtime이다. Colab은 PyTorch 공식 CUDA 12.4 wheel과
`requirements-training.lock`을 설치하고, macOS는 같은 PyTorch base version과 공통
lock을 사용한다. CUDA 유무와 build suffix만 장치에 따라 다르다. 학습 시작 전에 실제
runtime을 계약과 비교하고 하나라도 다르면 model download 전에 실패한다. 통과한 버전은
`metrics.json.runtime`에 저장한다.

## 6. smoke 검증 계약

smoke는 immutable base revision, `easy=0/medium=1/hard=2`, tokenizer 설정,
sample-weighted loss와 학습 hyperparameter 기록을 검사한다. 실제 100행에서 CUDA
forward/backward, validation metric, checkpoint 저장과 재로딩이 모두 성공해야 한다.
진행 상태는 `docs/ml/workstream-checklist.md`에서만 관리한다.

## 7. Kubeflow component

학습 component는 Dataset/config 입력, Model 출력, GPU 1개, CPU·memory request/limit,
retry 2회와 task cache를 선언한다. GCS URI Importer를 사용하는 pipeline IR은 컴파일돼
있다. 실제 MLMD 계보, node selector, taint/toleration, timeout과 Spot resume는 cluster
설정 뒤 검증한다. 체크 상태는 총괄 문서에서만 관리한다.

이 항목들은 공통 학습 코어를 변경하는 이유가 아니다. 동일 CLI를 container entrypoint로
호출하는 방식으로 추가한다.

### 7-1. 학습 container 계약

- Dockerfile: `ml/docker/modernbert-training.Dockerfile`
- build script: `ml/scripts/build-modernbert-training-image.sh`
- target platform: `linux/amd64`
- base: PyTorch `2.5.1`, CUDA `12.4`, cuDNN 9 runtime
- base image는 registry digest로 고정한다.
- entrypoint: `modernbert-train`
- runtime user: UID/GID `10001`
- working directory: `/workspace`
- Hugging Face cache: `/cache/huggingface`
- runtime contract: `modernbert-runtime-cu124-v1`

image에는 ModernBERT 학습 CLI, runtime 검사와 공통 JSON writer만 넣는다. RouterBench
라벨링 및 RouterArena 평가 CLI는 넣지 않아 CPU 전처리 image와 GPU 학습 image의
책임을 분리한다. Docker build 단계에서 핵심 버전 계약과 `modernbert-train --help`를
검증한다.

runtime 자체를 설명하는 version contract만 image에 고정한다. 데이터, 실험별 training
config, checkpoint와 cache는 image에 넣지 않고 Kubeflow input/output Artifact와
volume으로 전달한다. secret도 build argument나 image layer에 포함하지 않는다.

```bash
./ml/scripts/build-modernbert-training-image.sh

docker run --rm \
  llm-router-modernbert-training:cu124-v1 \
  --help
```

### 7-2. 2026-08-08 image build 결과

| 항목 | 결과 |
| --- | --- |
| image tag | `llm-router-modernbert-training:cu124-v1` |
| image ID | `sha256:8585bf3d39c1b09578a011cf73827f80f89b8e572430b2058de9010085b5c849` |
| platform | `linux/amd64` |
| local size | 6,487,268,550 bytes |
| entrypoint | `modernbert-train` |
| user/workdir | `10001:10001`, `/workspace` |
| runtime contract | PyTorch/CUDA/Transformers/data library 통과 |
| container CLI | `modernbert-train --help` 통과 |
| installed project scripts | `modernbert-train`만 존재 |
| excluded modules | RouterBench 라벨링, RouterArena 평가 |

Apple Silicon 로컬 Docker에서 amd64 emulation으로 image metadata, process 실행과 설치
module 목록을 확인했다. 초기학습 CUDA/길이 계약은 Colab smoke에서 검증하고, Vertex
container 계약은 feedback 100행 재학습 cloud smoke에서 검증한다.

### 7-3. KFP v2 학습 component 계약

KFP SDK는 `2.17.0`으로 고정한다. component는 GPU image를 재구현하지 않고 다음 명령을
Artifact path placeholder와 함께 실행한다.

```text
modernbert-train
  --config <Input[Artifact].path>
  --training-path <Input[Dataset].path><training_path_suffix>
  --runtime-contract /opt/llm-router/config/modernbert_runtime_cu124_v1.json
  --output-dir <Output[Model].path>
  --device cuda
```

파일 위치:

```text
ml/kubeflow/components/modernbert_training.py
ml/kubeflow/pipelines/modernbert_training.py
ml/kubeflow/pipelines/modernbert_retraining.py
ml/kubeflow/compile_training_workflows.py
ml/kubeflow/compile_modernbert_training.py
ml/requirements-kubeflow.txt
```

`training_path_suffix`는 역사적 initial preprocessing bundle과 현행 재학습 merge bundle 모두
`/training.parquet`를 사용한다. GPU image와 `modernbert-train` 파일 입력 계약은 같다.

초기학습 pipeline IR은 GCS 원천을 CPU `build-full`에서 처리한 뒤 T4 학습으로 연결하도록
구현돼 있지만 **실행하지 않는다**. 이는 component 계약과 compile regression을 보존하기
위한 역사적 경로다. 실제 초기학습은 Colab notebook이 승인된 Parquet와 config를 검증한 뒤
같은 학습 core를 호출한다. 재학습 pipeline은 같은 GPU component 앞에 동일 CPU image를
사용하는 feedback `prepare`와 `merge` component를 각각 둔다. 출력 Model에는
Hugging Face checkpoint, tokenizer, `training_config.json`과 `metrics.json`이 함께 저장된다.
이 단계의 metrics는 학습 기록이며 최종 배포 gate metric은 이후 별도 평가 component가
만든다.

Kubernetes GPU resource 값은 운영 기준이 아니다. feedback 재학습의 현행 시작값은
Vertex AI worker pool `n1-standard-4 + NVIDIA_TESLA_T4 1장`, replica/concurrency 1,
retry 0, 최대 실행 6시간이다.
dataset, config와 image URI는 immutable GCS version/digest로 전달한다. GKE resource는
생성하지 않는다. 이 platform 변경은 공통 학습 core/CLI의
데이터·label·checkpoint 계약을 바꾸지 않는다.

initial/retraining workflow를 모두 회귀 검사하도록 컴파일하는 명령은 다음과 같다. 생성된
initial IR은 제출하지 않고 feedback-retraining IR만 production template로 게시한다.

```bash
.venv/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-v2
```

## 8. VS Code Colab GPU smoke

로컬 accelerator가 없을 때는 공식 Google Colab VS Code 확장을 사용한다.

1. `ml/notebooks/modernbert_smoke_colab.ipynb`를 VS Code에서 연다.
2. `Select Kernel → Colab → New Colab Server`에서 GPU machine type을 선택한다.
3. Google OAuth를 완료한다.
4. Explorer에서 `ml/artifacts/modernbert-colab-smoke-input.tar.gz`를 우클릭해
   `Upload to Colab`을 실행한다.
5. notebook 셀을 순서대로 실행한다.

Colab 확장은 로컬 파일을 자동 공유하지 않으므로 현재 source/config, runtime lock과
100행 `training.parquet`만 포함한 upload archive를 사용한다. 다음 명령으로 매번 현재
파일에서 archive를 다시 만든다.

```bash
./ml/scripts/build-modernbert-colab-smoke-archive.sh
```

이 archive는 운반용 snapshot일 뿐이며 Git과 학습 로직의 source of truth가 아니다.

### 8-1. 전체 baseline notebook

전체 학습 입력 archive는 다음 명령으로 만든다.

```bash
./ml/scripts/build-modernbert-colab-full-archive.sh
```

`ml/notebooks/modernbert_full_colab.ipynb`를 GPU Colab server에서 열고 archive를
`/content/modernbert-colab-full-input.tar.gz`로 업로드한다. notebook은 Google Drive를
mount하고 고정 run directory에 checkpoint를 저장한다. 세션이 끊겼다면 새 server에서
같은 archive를 올리고 notebook을 처음부터 재실행한다. `resume-mode=auto`이므로 정상
최신 checkpoint가 있으면 자동 재개하고 최종 `_SUCCESS` model이 있으면 재학습하지 않는다.

새 config·dataset·max-step으로 실험할 때는 기존 checkpoint를 덮어쓰지 않고 Drive run
directory의 version을 올린다. fingerprint 불일치는 의도하지 않은 실험 혼합으로 간주해
실패시킨다.

공식 가이드:

- [Google Colab VS Code 확장](https://marketplace.visualstudio.com/items?itemName=google.colab)
- [Colab VS Code User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)

## 9. Colab CUDA smoke 결과

2026-08-08에 100행 RouterBench smoke artifact와 `modernbert-router-smoke-v1`
설정으로 Colab CUDA 실행을 완료했다. batch size 2, optimizer step 2의 목적은 성능
확정이 아니라 GPU 학습·저장·재로딩 경로 검증이다.

이 결과는 runtime contract 도입 전 실행이므로 학습 경로 검증 근거로는 유지하지만,
Docker/Vertex와의 dependency 동일성 근거로 사용하지 않는다. 새 archive로 계약 통과 여부를
확인하는 Colab smoke를 한 번 더 실행한다.

| 항목 | 결과 |
| --- | ---: |
| device | CUDA |
| optimizer steps | 2 |
| mean train loss | 1.580744 |
| validation loss | 1.459821 |
| validation accuracy | 0.30 |
| validation macro-F1 | 0.287879 |
| checkpoint reload | 통과 |
| id2label | `0=easy, 1=medium, 2=hard` |

3-class 무작위 accuracy 기준은 약 0.333이고 실제 parameter update에 사용된 행도 약
4개뿐이므로 이 accuracy를 모델 품질로 해석하지 않는다. 이번 결과는 finite loss,
CUDA backward, optimizer update, `save_pretrained()`, 재로딩 추론과 label 계약이 모두
연결됐다는 승인 근거다. 실제 성능 비교는 Colab 전체 baseline 학습을 완료하고
RouterBench holdout·RouterArena 평가를 실행한 뒤 수행한다.
