# ModernBERT 학습 명세와 로컬·Kubeflow 실행 결정

기준일: 2026-08-08
상태: 승인
관련 브랜치: `feat/data-labeling-training`

## 1. 결정이 필요했던 이유

GKE와 Kubeflow 인프라가 준비되기 전에 ModernBERT 학습 경로를 검증해야 한다. 이때
로컬 smoke 코드를 임시 구현한 뒤 클라우드용 학습 코드를 다시 만들면 두 환경의
전처리, 라벨 순서, loss와 checkpoint 형식이 달라질 위험이 있다.

핵심 질문은 다음과 같다.

> 로컬 smoke 학습 코드와 Kubeflow/GKE 전체 학습 코드를 별도로 만들 것인가,
> 아니면 같은 학습 코어를 실행 환경만 바꿔 재사용할 것인가?

## 2. 결정

**로컬과 Kubeflow가 하나의 학습 코어와 CLI를 공유한다.** 로컬 smoke는 작은 데이터와
보수적인 자원 설정으로 이 CLI를 직접 실행하고, Kubeflow component는 GCS Artifact가
마운트된 로컬 파일 경로와 전체 학습 config를 같은 CLI에 전달하는 얇은 wrapper로
구성한다.

```text
공통 학습 코어
  ├─ config 및 Parquet 계약 검증
  ├─ tokenizer와 truncation
  ├─ sample-weighted classification loss
  ├─ train/validation metric
  ├─ save_pretrained checkpoint
  └─ checkpoint 재로딩 및 label 계약 검증
       ↑                         ↑
  로컬 smoke CLI          Kubeflow component
  로컬 파일·CPU/MPS       Artifact 경로·T4 CUDA
```

Kubeflow component 안에는 tokenizer, loss, 학습 loop 또는 metric을 다시 구현하지 않는다.
Kubeflow는 Artifact 전달, step caching, retry, CPU/GPU resource, GCS 저장과 MLMD 계보를
담당한다.

## 3. 판단 근거

- 로컬에서 검증한 데이터·loss·checkpoint 계약을 GKE에서도 그대로 사용해 환경 간
  동작 차이를 줄일 수 있다.
- 버그 수정과 실험 설정 변경 지점이 한 곳이므로 유지보수와 재현성이 좋아진다.
- Kubeflow 없이도 동일 CLI를 실행할 수 있어 인프라 장애가 모델 코드 검증을 막지 않는다.
- 파일 기반 입출력은 로컬 경로와 Kubeflow Artifact 경로를 동일하게 다룰 수 있다.
- 로컬과 클라우드의 차이는 코드가 아니라 config의 데이터 규모, batch, precision,
  device와 출력 경로로 표현할 수 있다.

검토했지만 선택하지 않은 방식:

| 대안 | 선택하지 않은 이유 |
| --- | --- |
| 로컬 notebook과 Kubeflow 학습 코드를 각각 작성 | 학습 의미론이 중복되고 두 구현이 쉽게 달라진다. |
| 학습 코드에서 GCS를 직접 읽고 쓴다 | 로컬 테스트가 어려워지고 저장소 책임이 학습 코어에 섞인다. |
| Kubeflow가 split과 label을 다시 계산한다 | 이미 승인한 versioned 데이터와 다른 split이 생길 수 있다. |

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
`modernbert_t4_full.json`에서 8,192 token, batch 1, gradient accumulation 8,
gradient checkpointing과 FP16을 사용한다. 100행 cloud smoke에서 OOM이면 model이나
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

로컬 smoke와 Kubeflow는 위 세 입력의 의미가 같다.

| 항목 | 로컬 smoke | Kubeflow/GKE |
| --- | --- | --- |
| training path | 저장소의 로컬 Parquet | 입력 Artifact가 마운트된 경로 |
| output directory | `ml/artifacts/` 하위 | 출력 Artifact 경로 |
| device | auto CPU/MPS/CUDA | CUDA T4 |
| precision | FP32 | FP16 |
| max length | 512 | 8,192 |
| 데이터 규모 | smoke 100행 | 승인된 전체 split |
| 실행 관리 | 사용자가 CLI 실행 | component가 같은 CLI 실행 |

학습 코어는 GCS bucket, Kubeflow run ID 또는 Kubernetes API를 알지 못해야 한다.

### 5-1. runtime 재현성 계약

`modernbert_runtime_cu124_v1.json`은 다음 핵심 버전을 고정한다.

```text
PyTorch 2.5.1 / CUDA 12.4 / Transformers 4.48.3
NumPy 2.1.2 / Pandas 2.3.3 / PyArrow 21.0.0
Python minor 3.11 또는 3.12
```

Docker/GKE가 기준 runtime이다. Colab은 PyTorch 공식 CUDA 12.4 wheel과
`requirements-training.lock`을 설치하고, macOS는 같은 PyTorch base version과 공통
lock을 사용한다. CUDA 유무와 build suffix만 장치에 따라 다르다. 학습 시작 전에 실제
runtime을 계약과 비교하고 하나라도 다르면 model download 전에 실패한다. 통과한 버전은
`metrics.json.runtime`에 저장한다.

## 6. smoke 검증 계약

smoke는 immutable base revision, `easy=0/medium=1/hard=2`, tokenizer 설정,
sample-weighted loss와 학습 hyperparameter 기록을 검사한다. 실제 100행에서 CUDA
forward/backward, validation metric, checkpoint 저장과 재로딩이 모두 성공해야 한다.
진행 상태는 `docs/ml-training-evaluation-workstream-checklist.md`에서만 관리한다.

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
module 목록을 확인했다. 실제 CUDA 가용성과 8,192 token 메모리는 GKE T4 100행 cloud
smoke에서 검증한다.

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

`training_path_suffix`는 초기 학습의 direct Parquet에는 빈 문자열, 재학습 merge bundle에는
`/training.parquet`를 사용한다. 따라서 기존 GPU image와 `modernbert-train` 파일 입력
계약을 바꾸지 않고 두 artifact 형태를 모두 연결한다.

초기 학습 pipeline은 GCS URI 두 개를 Importer로 `Dataset`과 config `Artifact`로 등록한
뒤 학습 component에 연결한다. 재학습 pipeline은 같은 GPU component 앞에 동일 CPU
image를 사용하는 feedback `prepare`와 `merge` component를 각각 둔다. 출력 Model에는
Hugging Face checkpoint, tokenizer, `training_config.json`과 `metrics.json`이 함께 저장된다.
이 단계의 metrics는 학습 기록이며 최종 배포 gate metric은 이후 별도 평가 component가
만든다.

task resource는 GPU 1개(`nvidia.com/gpu`), CPU request/limit 4/8, memory request/limit
24Gi/32Gi로 시작한다. retry는 2회, cache는 활성화한다. cache 오판을 피하려면 dataset,
config와 image URI를 immutable version/digest로 전달해야 한다.

현재 파이프라인의 `training_image` 기본값은 로컬 compile용 tag다. 서버 확보 후 실제
Artifact Registry URI를 전달하고, T4 node pool이 여러 GPU node와 섞여 있다면 GKE
node selector와 toleration을 추가한다. platform-specific 설정은 공통 학습 component의
파일 계약을 변경하지 않는다.

두 workflow 전체를 컴파일하는 명령은 다음과 같다.

```bash
.venv/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows
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

공식 가이드:

- [Google Colab VS Code 확장](https://marketplace.visualstudio.com/items?itemName=google.colab)
- [Colab VS Code User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)

## 9. Colab CUDA smoke 결과

2026-08-08에 100행 RouterBench smoke artifact와 `modernbert-router-smoke-v1`
설정으로 Colab CUDA 실행을 완료했다. batch size 2, optimizer step 2의 목적은 성능
확정이 아니라 GPU 학습·저장·재로딩 경로 검증이다.

이 결과는 runtime contract 도입 전 실행이므로 학습 경로 검증 근거로는 유지하지만,
Docker/GKE와의 dependency 동일성 근거로 사용하지 않는다. 새 archive로 계약 통과 여부를
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
연결됐다는 승인 근거다. batch/epoch/precision 조정과 실제 성능 비교는 Kubeflow 100행
cloud smoke 이후 수행한다.
