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

## Kubeflow ModernBERT GPU 학습 component

KFP SDK는 `ml/requirements-kubeflow.txt`에서 `2.17.0`으로 고정한다. 학습 component는
기존 GPU image의 `modernbert-train` entrypoint를 그대로 호출하고 모델 코드를 다시
구현하지 않는다.

```bash
python3.12 -m venv .venv-kubeflow
.venv-kubeflow/bin/pip install -r ml/requirements-kubeflow.txt

.venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows
```

생성되는 IR YAML은 Git에서 제외되는 로컬 artifact다.

```text
feedback-prepare-component.yaml
feedback-merge-component.yaml
modernbert-training-component.yaml
initial-training-pipeline.yaml
feedback-retraining-pipeline.yaml
```

component 계약:

```text
Input[Dataset] training_dataset   # canonical training.parquet
Input[Artifact] training_config   # modernbert_t4_full.json 또는 smoke config
str training_path_suffix          # direct file은 "", bundle은 "/training.parquet"
Output[Model] trained_model       # checkpoint, tokenizer, metrics.json
```

초기 학습 pipeline은 로컬에서 검증·전처리가 끝난 `training.parquet`와 config의 GCS URI를
Importer로 가져와 바로 GPU task에 연결한다. RouterBench 라벨링·전처리를 서버에서 다시
실행하지 않는다. 재학습 pipeline은 feedback manifest/events/config와 기존 학습 artifact를
가져와 `prepare(CPU) → merge(CPU) → train(GPU)`로 실행한다.

GPU task는 `nvidia.com/gpu=1`, CPU request/limit `4/8`, memory request/limit
`24Gi/32Gi`, retry 2회와 cache 활성화를 선언한다. `training_image` parameter에는 GKE가
pull할 수 있는 Artifact Registry의 immutable image URI를 실행 시 전달해야 한다. 기본
로컬 tag는 IR compile test용이며 클러스터에서 그대로 사용할 수 없다.

현재는 GPU 종류와 무관한 `nvidia.com/gpu=1`만 선언한다. T4 node pool label, taint와
toleration은 실제 GKE 구성이 확정된 뒤 platform-specific 설정으로 추가한다. dataset과
config URI는 같은 경로의 내용을 덮어쓰지 않는 immutable version 경로여야 KFP cache와
계보를 안전하게 사용할 수 있다.

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

기본 target은 GKE CPU node와 맞춘 `linux/amd64`다. 데이터는 image layer에 넣지 않고
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
내부 CLI smoke를 통과했다. 초기학습과 feedback 재학습 pipeline은 학습 뒤에
RouterArena 평가 → candidate package → deploy gate를 직접 실행한다. 이미 저장된
checkpoint만 다시 검증할 때는 독립 `model_qualification_pipeline`을 사용한다.

이 gate는 온라인 서비스에 모델을 직접 배포하지 않는다. 통과 여부와 candidate
package를 생성하는 데까지만 ML 책임이며, GCS `approved/` 승격 방식과 FastAPI process
교체·health check·rollback 호출 계약은 온라인 팀과 확정한 뒤 연결한다.
