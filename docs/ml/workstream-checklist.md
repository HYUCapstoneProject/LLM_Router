# ML 작업 총괄 문서

상태: **현행 기준 문서**

기준일: 2026-08-14

브랜치: `feat/data-labeling-training`

최근 기준 커밋: `3866b47`

## 1. 문서 목적과 관리 원칙

이 문서는 다음 세 가지의 현행 구조와 진행 상태를 한 곳에서 관리한다.

1. 현재 `ml/` 코드의 구조, 역할, 입출력과 실행 흐름
2. Spring/API·Gateway·DB 및 인프라 담당자와 맞닿는 책임 경계
3. 로컬 작업부터 GCS·Vertex AI 학습과 모델 승격까지의 단일 체크리스트

ML 작업 상태를 나타내는 체크박스는 **이 문서에만 둔다**. RouterBench, RouterArena,
ModernBERT 세부 명세는 설계 근거와 데이터 규칙을 설명하는 참고 문서이며 진행 상태를
관리하지 않는다. 과거 MVP 문서의 체크리스트도 현행 ML 상태로 해석하지 않는다.

팀 회의에서 값을 확정할 계약과 인프라 항목은
`docs/ml/backend-infrastructure-agreement.md`, 개발 중 질문과 선택 근거는
`docs/ml/development-questions-and-decisions.md`에 별도로 기록한다. 두 문서는 설명과
결정 기록이며 작업 체크박스는 두지 않는다.

체크 표시는 다음 뜻이다.

- `[x]`: 코드·산출물·테스트 또는 container smoke로 확인된 완료
- `[ ]`: 아직 구현·검증·합의가 끝나지 않은 항목
- 로컬 구현만 끝나고 cluster 검증이 남았다면 cluster 항목은 `[ ]`로 유지한다.
- 팀 합의 전 제안값은 코드 기본값일 수는 있어도 운영 계약으로 간주하지 않는다.

## 2. 범위와 책임 경계

### 2-1. ML 담당 범위

- RouterBench 원본 검증, canonicalization, 모델 capability tier와 3단계 라벨 생성
- RouterBench 학습 projection, 중복 그룹, train/validation/test split 생성
- RouterArena 평가 전처리, ModernBERT 추론, 분포·분류 metric과 SVG 생성
- 사용자 에스컬레이션 배치의 검증, 정규화, 중복 제거와 RouterBench 병합
- ModernBERT 학습, checkpoint 계약, 모델 package와 deploy gate
- CPU/GPU Docker image, Kubeflow component·pipeline과 ML artifact 계보
- GCS 데이터·모델·평가 경로, Vertex AI Pipelines CPU 단계와 Custom Training GPU 학습

### 2-2. 온라인 팀과 맞닿는 경계

| 접점 | ML 책임 | 온라인 팀 책임 |
| --- | --- | --- |
| 라우터 추론 | `easy/medium/hard`, confidence, model version 계약 | FastAPI 운영, Spring 호출, timeout/fallback |
| 실제 LLM 선택 | 라벨 순서와 평가용 권장 mapping 제공 | 실제 Claude model ID와 운영 routing 정책 |
| 모델 전달 | 평가, gate, package, checksum, manifest 게시 | 다운로드, health check, process 전환과 rollback |
| 피드백 수집 | export batch 검증과 학습 채택 | 에스컬레이션·최종 만족 결과 수집, DB 저장과 export |
| 개인정보 | 전달된 batch 재검사와 부적합 행 격리 | 수집 동의, PII 제거, retention 적용 |
| 재학습 trigger | idempotent pipeline과 결과 제공 | 누적량 집계·호출 주체 공동 합의 |
| 관측성 | model/data/evaluation version과 ML metric | request, latency, token, cost, cache와 오류 로그 |
| semantic cache | ML/Kubeflow 범위에서 제외 | QQP 평가, bge embedding, pgvector와 threshold 적용 |

### 2-3. ML에서 구현하지 않는 항목

- React 화면, Spring `/api/chat` orchestration과 Anthropic API 호출
- PostgreSQL/pgvector 온라인 CRUD, QQP threshold sweep과 cache 정책
- 실제 FastAPI serving process의 배포·교체·rollback
- 사용자 feedback을 DB에서 직접 조회하는 Kubeflow component

`졸업프로젝트 개요.md`의 QQP 평가는 전체 프로젝트에는 남아 있지만 온라인 서빙·캐시
축의 책임이다. ML은 해당 결과를 학습이나 Kubeflow DAG에 넣지 않고 최종 보고서에서
라우터 결과와 구분해 인용한다.

## 3. 전체 아키텍처와 데이터 흐름

### 3-1. 초기 학습

```text
RouterBench 원본 Pickle
  -> 로컬/Colab 공통 core로 checksum/schema·라벨링·split 검증
  -> 봉인된 training.parquet와 manifest를 Colab/Drive에 게시
  -> Colab T4에서 ModernBERT 초기학습·checkpoint resume
  -> Drive 최종 checkpoint checksum·_SUCCESS 검증
  -> package-v1 assemble/import·RouterArena 평가·deploy gate
  -> GCS candidate/approved 게시와 CURRENT.json 갱신
```

초기 Colab 전처리 core는 `routerbench_labeling_v2.json`만 허용된 production
config로 사용한다. 결과는 v2 라벨 분포와 자연분포 holdout, 클래스별 6,256행 train을
포함하는 directory Dataset artifact이며 GPU 학습기는 그 안의 `training.parquet`만 읽는다.

초기학습의 공식 실행 환경은 Colab이다. `modernbert_full_colab.ipynb`가 공통 Python core와
CLI를 호출하고 Google Drive checkpoint·자동 재개를 담당한다. 기존
`modernbert-router-initial-training` IR은 code/compile regression을 위해 보존하지만 실제
PipelineJob이나 Vertex CustomJob으로 제출하지 않는다. GCP GPU 학습은 3-2의 feedback
재학습부터 사용한다.

### 3-2. 피드백 재학습

```text
온라인 DB escalation event
  -> Spring이 feedback-inline-v1 JSON으로 FastAPI에 전송
  -> FastAPI가 schema·의미 검증 후 Parquet + manifest 생성
  -> FastAPI가 GCS versioned feedback/replay artifact 기록
  -> Vertex Pipeline CPU prepare: schema/checksum/PII/채택 조건 검증
  -> Vertex Pipeline CPU merge: RouterBench train과 병합, holdout 보호
  -> Google Cloud Pipeline Components가 Vertex AI CustomJob 제출·감시
  -> Vertex AI T4 1장 재학습
  -> 초기 학습과 동일한 CPU 평가·package·gate
```

Vertex AI Pipeline이 PostgreSQL을 직접 읽지 않는다. 외부 schema 변경은 feedback adapter에서
흡수하고, 학습기는 항상 동일한 6필드 canonical schema만 받는다.

### 3-3. 모델 자격검증만 재실행

이미 저장된 checkpoint를 다시 평가할 때는 독립 qualification pipeline을 사용한다.

```text
checkpoint + RouterArena + baseline metrics + gate rules
  -> evaluate -> package -> deploy gate
```

## 4. `ml/` 디렉터리 구조

```text
ml/
├── config/                 버전·라벨·학습·평가·feedback 정책 JSON
├── src/llm_router_labeling/
│   ├── training/           RouterBench, feedback, ModernBERT 학습
│   ├── evaluation/         RouterArena 전처리·추론·metric·SVG
│   ├── deployment/         모델 package·checksum·deploy gate
│   └── shared/             파일 I/O, checksum, 공통 텍스트 정규화
├── docker/                 역할별 4개 image 정의
├── kubeflow/
│   ├── components/         KFP container component
│   ├── pipelines/          초기학습·재학습·자격검증 DAG
│   ├── task_config.py      KFP CPU resource·retry·cache 정책
│   └── compile_*.py        KFP IR YAML compiler
├── tests/                  단위·계약·pipeline compile test와 fixture
├── notebooks/              Colab ModernBERT smoke·전체 학습 notebook
├── scripts/                image·Colab archive build helper
├── pyproject.toml           로컬 통합 package와 CLI entrypoint
└── README.md                실행 명령 중심의 빠른 사용 안내
```

## 5. 코드 파일별 역할

### 5-1. `training/`: 학습 데이터와 모델

| 파일 | 핵심 역할 | 주요 입력 | 주요 출력 |
| --- | --- | --- | --- |
| `routerbench_labeling.py` | 모델 성능 컬럼 탐지, 0/5-shot canonicalization, capability tier 산정, tier consensus 라벨링 | 신뢰된 RouterBench DataFrame, labeling config | 라벨 DataFrame, model tier, metric |
| `routerbench_preprocessing.py` | 질문 wrapper 정제, exact/near duplicate grouping, group-aware split, 학습/감사 projection 분리 | `labeled.parquet` | `training.parquet`, `metadata.parquet`, metric |
| `routerbench_cli.py` | download·smoke·label·preprocess 명령을 묶는 RouterBench 진입점 | config, raw Pickle | versioned labeling/preprocessing artifact |
| `feedback_retraining_preprocessing.py` | feedback manifest/row 검증, 채택·격리, 반복 질문 집계, 충돌 제거, base train 병합 | manifest, events, base training | canonical feedback, merged training, metadata |
| `feedback_retraining_cli.py` | feedback `prepare`, `merge`, `build` 파일 기반 명령 | JSON/Parquet, config | Parquet·metric·lineage manifest |
| `modernbert_training.py` | 6필드 학습 schema 검증, tokenizer, sample-weighted loss, epoch 내부 validation과 RNG 격리된 관찰 hook을 제공하는 공통 학습 코어 | training Parquet, TrainingConfig, 선택적 resume checkpoint | Hugging Face checkpoint, `metrics.json` |
| `modernbert_checkpointing.py` | model·optimizer·scheduler·scaler·RNG·epoch/batch cursor를 원자적으로 저장하고 checksum·fingerprint로 검증 | 공통 학습 상태, dataset/config fingerprint | sealed resume checkpoint와 manifest |
| `modernbert_colab_training.py` | Colab 임시 디스크 학습, Google Drive checkpoint 동기화, 9회 RouterBench validation 저장, 최신 정상 checkpoint와 누락 epoch 평가 자동 복구를 조율 | local/persistent run directory, resume 정책 | Drive의 주기 checkpoint, validation/epoch 평가와 최종 model |
| `modernbert_validation_progress.py` | epoch당 33%·66%·100% RouterBench validation metric을 봉인·Drive 동기화하고 9점 진행 SVG를 생성 | validation callback state | observation별 JSON/manifest와 progress JSON·SVG |
| `modernbert_runtime.py` | Python/PyTorch/CUDA/Transformers 등 runtime fail-fast 계약 | runtime config와 실제 library version | 검증 report 또는 실행 중단 |
| `modernbert_training_cli.py` | 로컬·Docker·Kubeflow용 단일 실행 `modernbert-train` 진입점 | dataset/config/runtime path | checkpoint directory |
| `modernbert_colab_training_cli.py` | 무료 Colab 세션 중단을 전제로 checkpoint 저장·Drive 복구를 설정하는 `modernbert-colab-train` 진입점 | dataset/config/runtime, local·Drive run path | 재개 가능한 checkpoint와 최종 model |

RouterBench 라벨의 뜻은 일반적인 교과 난이도가 아니라 **질문을 만족스럽게 해결할 수
있는 최소 모델 capability tier**다. 라벨 계약은 항상 다음 순서를 사용한다.

```text
easy=0, medium=1, hard=2
```

학습 loss에는 `sample_weight`가 반영되며, 모델별 원시 성공률·가격·응답은 feature로
넣지 않는다. 이것들은 라벨 생성 근거로만 사용하고 학습 projection에서 분리한다.

### 5-2. `evaluation/`: RouterArena 평가

| 파일 | 핵심 역할 | 주요 출력 |
| --- | --- | --- |
| `routerarena_preprocessing.py` | Context/Question/Options 조립, Answer 제외, 공식 3라벨 검증, full/robustness join, 오염·중복·RouterBench overlap 제외 | `evaluation.parquet`, `metadata.parquet` |
| `routerarena_cli.py` | RouterArena download와 evaluation artifact 생성 CLI | 전처리 artifact와 metric |
| `routerarena_routing_evaluator.py` | checkpoint label 순서 검사, GPU/CPU 추론, accuracy·macro-F1·confusion·under/over routing과 그룹 분포 | prediction DataFrame과 metric dict |
| `routerarena_epoch_evaluation.py` | Colab epoch 1·2·3 inference-only 평가, 봉인·Drive 동기화·누락 평가 복구와 누수 방지 계약 | epoch별 predictions/metrics/manifest와 progress JSON |
| `routerarena_visualization.py` | 예측 그룹 분포와 epoch별 accuracy·macro-F1·under-routing 변화를 SVG로 변환 | epoch별 SVG 2개와 progress SVG |
| `routerarena_evaluation_cli.py` | checkpoint와 evaluation artifact를 연결하는 `routerarena-evaluate` CLI | 아래 평가 bundle |

평가 bundle은 다음 파일로 구성된다.

```text
predictions.parquet
metrics.json
model_info.json
predicted_group_distribution.svg
predicted_group_heatmap.svg
```

RouterArena의 공식 `easy/medium/hard`와 RouterBench proxy 라벨은 이름이 같아도 의미가
완전히 동일하다고 가정하지 않는다. 주 분석은 모델이 예측한 세 그룹 각각에서
RouterArena 공식 라벨이 어떻게 분포하는지 확인하는 것이다.

### 5-3. `deployment/`: package와 gate

| 파일 | 핵심 역할 | 주의점 |
| --- | --- | --- |
| `model_package.py` | checkpoint 조립, `model-package-v1` manifest 생성, 필수 파일·라벨·SHA-256 검증 | assemble 단계는 사용자가 `approved`를 넣어도 `candidate`로 강제한다. |
| `model_package_cli.py` | `create`, `assemble`, `validate` 명령 | Kubeflow package component가 사용한다. |
| `deploy_gate.py` | 절대 최소/최대 metric과 baseline 대비 delta 규칙 평가 | 수치 임계값은 아직 운영 확정 전이다. |
| `deploy_gate_cli.py` | gate report 저장 후 실패 시 exit code 2 | gate 실패가 Kubeflow task 성공으로 처리되지 않게 한다. |

모델 package 최소 구성은 다음과 같다.

```text
candidate/<model-version>/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors 또는 pytorch_model.bin
├── training_config.json
├── metrics.json
├── routerarena_metrics.json
└── model_manifest.json
```

`model_manifest.json`에는 model/data/preprocessing/label/training/evaluation version,
`id2label`, tokenizer max length, code commit, gate status와 파일별 SHA-256이 들어간다.
현재 gate는 후보를 검사할 뿐 GCS `approved/` 이동이나 온라인 process 교체를 수행하지
않는다.

### 5-4. `shared/`: 공통 기반

| 파일 | 역할 |
| --- | --- |
| `dataset_io.py` | JSON 설정 읽기, SHA-256, 크기 검증, 중단 재개 다운로드, JSON 쓰기 |
| `text_normalization.py` | RouterBench·RouterArena·feedback이 공유하는 중복 탐지용 정규화 |

### 5-5. `config/`: 변경 가능한 정책

| 설정 | 고정하거나 조절하는 값 |
| --- | --- |
| `routerbench_labeling.json` | v1 재현 전용 legacy config |
| `routerbench_labeling_v2.json` | GSM8K partial-credit, unresolved 제외, 자연분포 holdout, 6,256/class train 선택 |
| `routerbench_labeling_v3.json` | RouterBench 내부 probe로 선택한 overall solve-rate 경계와 자연분포 train 실험 |
| `routerarena_preprocessing.json` | 공식 revision/checksum, split, overlap·duplicate 정책 |
| `routerarena_evaluation.json` | label 순서, 평가 의미, batch/max length, 평가용 route mapping |
| `feedback_retraining_preprocessing.json` | export schema version, 채택 조건, sample weight, merge 정책 |
| `modernbert_smoke.json` | 100행 smoke, 2 optimizer step, 512 token 설정 |
| `modernbert_t4_full.json` | v1 checkpoint 재현 전용 legacy config |
| `modernbert_t4_balanced_v2.json` | v2 T4 학습과 optimizer-unit balanced sampler, loss 관찰 |
| `modernbert_runtime_cu124_v1.json` | PyTorch 2.5.1, CUDA 12.4, Transformers 4.48.3 등 runtime 계약 |

`routerarena_evaluation.json`의 `easy→haiku`, `medium→sonnet`, `hard→opus`는 평가용
표시 mapping이다. 온라인 운영 모델 ID와 fallback 정책을 대신하지 않는다.

### 5-6. CLI 진입점

| 명령 | 역할 |
| --- | --- |
| `routerbench-labeling` | RouterBench download·smoke·full label·preprocess·KFP용 `build-full` bundle |
| `routerarena-preprocessing` | RouterArena download와 평가 artifact 생성 |
| `escalation-feedback-preprocessing` | feedback prepare·merge·일괄 build |
| `modernbert-train` | 로컬·Colab·Vertex AI 공통 ModernBERT 학습 |
| `routerarena-evaluate` | checkpoint의 RouterArena 추론과 평가 bundle 생성 |
| `modernbert-package` | candidate package create·assemble·validate |
| `modernbert-deploy-gate` | metric rules 평가와 gate report 생성 |

CLI는 파일 경로와 versioned JSON config를 받는다. GCS SDK나 데이터베이스 접근은 ML
core에 넣지 않고 Kubeflow Importer와 외부 exporter가 담당한다.

## 6. 데이터와 artifact 계약

### 6-1. ML 내부 학습 schema

외부 데이터 출처와 관계없이 학습기는 다음 6필드만 받는다.

| 필드 | 의미 |
| --- | --- |
| `sample_id` | 출처와 version을 포함한 재현 가능 ID |
| `model_input_text` | 응답·정답을 제외한 질문 입력 |
| `difficulty_label` | `easy`, `medium`, `hard` |
| `sample_weight` | 신뢰도와 source 정책을 반영한 양수 |
| `split` | `train`, `validation`, `test` |
| `duplicate_group_id` | 유사 질문의 split 누수를 막는 그룹 |

### 6-2. RouterBench artifact

```text
labeling/
├── labeled.parquet       라벨과 라벨 근거
├── model_tiers.json      모델 capability tier
└── metrics.json

preprocessed/
├── training.parquet      모델 입력용 6필드
├── metadata.parquet      원문, 라벨 근거, 제외 사유와 분석 컬럼
└── metrics.json
```

현재 전체 결과는 라벨링 34,778행, 학습 projection 34,764행이며 split은
train 27,808 / validation 3,478 / test 3,478이다.

### 6-3. RouterArena artifact

```text
preprocessed/
├── evaluation.parquet    모델 입력과 공식 라벨
├── metadata.parquet      원문과 분석·제외 정보
└── metrics.json
```

전체 RouterBench와 exact overlap 22행을 제외한 full 평가 artifact는 8,375행이다.

### 6-4. 온라인 → FastAPI inline feedback 계약

Spring의 HTTP 전달 단위:

```text
batchId, schemaVersion, requestedAt, events[]
```

FastAPI가 내부에서 생성할 manifest 필드:

```text
schema_version, batch_id, created_at
window_start, window_end, event_count
data_format, data_uri, sha256
producer_version, watermark
```

Spring이 보내고 FastAPI가 Parquet으로 변환할 event 필드:

```text
event_id, request_id, question
initial_label, initial_model, final_label, final_model
user_satisfied, escalation_count, created_at
router_model_version, pii_redacted
```

현재 `feedback-export-v0.1` prototype은 다음 조건을 모두 만족한 행만 자동 후보로 둔다.

- SHA-256, schema version과 행 수가 manifest와 일치
- `pii_redacted=true`, `user_satisfied=true`, `escalation_count>=1`
- `easy < medium < hard` 순서에서 final label이 initial label보다 엄격히 높음
- 같은 정규화 질문의 final label이 서로 충돌하지 않음

같은 질문·같은 final label은 한 행으로 집계하고 기본 weight 1.0에 반복 1건당 0.25를
더해 최대 2.0으로 제한한다. 이는 합의된 운영 정책이 아니라 변경 가능한 prototype이다.
validation/test와 겹치는 feedback은 제외하고, 기존 train의 동일 질문과 충돌하면 해당
train 행을 feedback 행으로 교체한다.

### 6-5. ML → 온라인 Router API 제안

최소 요청:

```json
{
  "request_id": "req-123",
  "question": "사용자 질문"
}
```

최소 응답:

```json
{
  "label": "medium",
  "confidence": 0.82,
  "model_version": "router-modernbert-v1",
  "latency_ms": 24
}
```

이 schema와 label 이름, timeout/fallback은 아직 팀 합의 전이다.

## 7. Docker image와 자원 분리

| image | 확정 실행 위치 | 포함 기능 | 변경 상태 |
| --- | --- | --- | --- |
| `llm-router-data-processing:py312-v2` | Vertex Pipeline CPU task | RouterBench·RouterArena 전처리, feedback prepare/merge | 구현됨 |
| `llm-router-modernbert-training:cu124-v1` | Vertex AI T4 CustomJob | `modernbert-train`, runtime 검증 | 공식 CustomJob wrapper 구현됨 |
| `llm-router-modernbert-evaluation:cu124-v1` | Vertex Pipeline CPU task | RouterArena checkpoint 추론·metric·SVG | CPU 실행 전환됨, CPU 전용 image 재빌드 필요 |
| `llm-router-model-promotion:py312-v1` | Vertex Pipeline CPU task | package·checksum·deploy gate | 구현됨 |

네 image는 모두 `linux/amd64`, non-root UID/GID `10001` 기준으로 로컬 build·CLI smoke를
통과했다. 다만 `modernbert-evaluation:cu124-v1`은 CUDA base image이므로 CPU 전용
평가 image와 smoke를 새로 검증해야 한다. GKE resource는 생성하지 않으며 모든 image는
Artifact Registry에서 Vertex AI가 직접 pull한다.

## 8. Vertex AI Pipelines(KFP v2) 구성

### 8-1. component

| component | 확정 실행 위치 | 역할 | 현재 상태 |
| --- | --- | --- | --- |
| RouterBench/RouterArena data component | 로컬/Colab 또는 qualification CPU | 초기 데이터 검증·평가; 초기학습 PipelineJob에는 사용하지 않음 | 구현됨 |
| `feedback_prepare_component` | Vertex managed CPU | export batch 검증과 canonical feedback 생성 | 구현됨 |
| `feedback_merge_component` | Vertex managed CPU | base training과 feedback 병합 | 구현됨 |
| `vertex_modernbert_training_job_component` | Vertex CustomJob launcher | T4 CustomJob 제출·감시, `gcp_resources` 출력 | 구현됨 |
| `routerarena_evaluation_component` | Vertex managed CPU | RouterArena full 평가 bundle 출력 | CPU resource로 전환됨 |
| `model_package_component` | Vertex managed CPU | candidate package와 manifest 생성 | 구현됨 |
| `deploy_gate_component` | Vertex managed CPU | 규칙 평가, 실패 시 task 차단 | 구현됨 |

IR에는 Kubernetes `nvidia.com/gpu` 요청이 없다. 학습 worker pool 안에만
`n1-standard-4`, `NVIDIA_TESLA_T4=1`, replica 1, restart false, 최대 6시간이 있으며
pipeline task retry는 설정하지 않는다. CustomJob resource는 `gcp_resources` 출력으로
Vertex ML Metadata에 연결된다.

### 8-2. pipeline

| pipeline | DAG |
| --- | --- |
| `modernbert_training_pipeline` | **비활성 역사적 IR**; compile regression만 유지하고 제출하지 않음 |
| `modernbert_retraining_pipeline` | Vertex CPU feedback prepare/merge → Vertex T4 CustomJob → Vertex CPU evaluate/package/gate |
| `model_qualification_pipeline` | 기존 checkpoint import → CPU evaluate/package/gate |

`compile_training_workflows.py`는 RouterBench v2 CPU component, feedback CPU component,
학습·평가·승격 component와 pipeline 3개를 KFP
v2 IR YAML로 컴파일한다. compile test뿐 아니라 Vertex job
ID·image digest·입출력 GCS URI를 KFP/MLMD에서 추적할 수 있는지 cloud smoke로 검증해야
한다. production Vertex job을 Vertex PipelineJob 밖에서 직접 실행하는 경로는 두지 않는다.

## 9. 팀 협의 문서

백엔드·온라인 계약과 Vertex AI 인프라 결정은
`docs/ml/backend-infrastructure-agreement.md`에서 관리한다. 해당 문서는 다음 내용을
회의 단위로 분리한다.

- MVP·졸업프로젝트 개요와 현행 ML 설계의 충돌 및 문서 정정 기준
- Router API, 실제 Claude mapping, timeout/fallback과 idempotency
- feedback export 보증 범위, 학습 라벨 채택, trigger N과 실행 주체
- model package, deploy gate, 승인·serving 전환·health check·rollback
- 공통 관측성과 최종 실험 결과 결합 방식
- project, region, GCS, Artifact Registry, Vertex AI Pipelines/Custom Training, IAM과 비용

GKE를 사용하지 않는 Vertex AI Pipelines CPU → Vertex AI Custom Training GPU 구조는
2026-08-14 확정되었으며 팀의 승인 또는 대안 선택을 받는 협의 항목이 아니다. 회의에서는
이 결정을 통보하고 resource 이름,
권한 주체와 온라인 인수인계처럼 팀 경계에 해당하는 값만 확정한다.

회의 결과로 확정된 schema/config/fixture는 코드와 contract test에 반영하고, 완료 여부는
이 문서 11장의 체크리스트에서만 갱신한다.

## 10. GCP·GCS·Vertex AI 착수 전 상태

인증 계정에서 접근 가능한 project는
`project-763753da-fa8f-4afd-b31`(`HYUCapstone`) 하나다. 2026-08-10 API 실측 결과는
다음과 같다.

| 항목 | 상태 |
| --- | --- |
| Google Cloud project | 접근·선택 완료 |
| Cloud Billing | project 연결·billing enabled, 현재 로그인 계정은 billing account 세부 조회 권한 없음 |
| GCS bucket | 0개 |
| Kubernetes Engine API | 비활성 |
| Artifact Registry API/repository | 비활성/없음 |
| kubectl context | 없음 |

따라서 현재는 **계정과 project 연결만 확인된 상태**다. architecture와 기본 region은
확정했지만, billing admin이 Welcome Credit 잔액·원래 만료일을 확인하고 이름·IAM·비용
guardrail을 적용하기 전에는 GPU job을 실행하지 않는다.

권장 GCS 구조 초안:

```text
gs://<bucket>/datasets/routerbench/<version>/
gs://<bucket>/datasets/routerarena/<version>/
gs://<bucket>/datasets/continual-feedback/<candidate-version>/
gs://<bucket>/datasets/retraining/<dataset-version>/
gs://<bucket>/models/router-modernbert/<model-version>/candidate/
gs://<bucket>/models/router-modernbert/<model-version>/approved/
gs://<bucket>/evaluations/<model-version>/<evaluation-version>/
gs://<bucket>/pipeline-root/
```

## 11. 단일 실행 체크리스트

### A. 완료된 로컬 데이터 기준선

- [x] RouterBench source URL·size·SHA-256과 Pickle 신뢰 경계를 고정했다.
- [x] 0/5-shot canonicalization, target question 추출과 핵심 eval 범위를 구현했다.
- [x] 11개 모델 capability tier와 minimum-sufficient-tier 3라벨을 생성했다.
- [x] 100행 smoke와 전체 34,778행 라벨링을 완료했다.
- [x] 중복 그룹·split과 34,764행 학습 projection을 생성했다.
- [x] RouterArena 공식 revision·checksum과 full/sub_10/robustness 전처리를 구현했다.
- [x] 전체 RouterBench overlap 22행을 제외한 RouterArena full 8,375행을 만들었다.
- [x] 합성 feedback validator·adapter·dedup·merge와 holdout 보호를 구현했다.

### B. 완료된 모델·평가·파이프라인 기반

- [x] Colab CUDA 100행 ModernBERT smoke와 checkpoint 재로딩을 통과했다.
- [x] `modernbert-router-balanced-v2` 전체 18,768행을 Colab T4에서 3 epoch 학습하고,
  RouterBench validation 9회·RouterArena 3회 관찰과 run 무결성 보고서를 완료했다.
- [x] Colab 전체 학습용 로컬 50 step·Drive 500 step/epoch 종료 checkpoint, checksum·자동 재개 wrapper와 notebook을 구현했다.
- [x] RouterBench validation을 epoch당 33%·66%·100%, 총 9회 측정해 Drive JSON·SVG로 저장하고 RNG 및 학습 의사결정에서 격리했다.
- [x] 동일 seed의 무중단 4-step과 2-step 중단 후 재개 결과가 정확히 일치하는 통합 테스트를 통과했다.
- [x] `easy=0, medium=1, hard=2`, sample-weighted loss와 runtime 계약을 고정했다.
- [x] 역할별 Docker image 4개를 빌드하고 CLI smoke를 통과했다.
- [x] RouterArena predictor, 분류·routing metric과 예측 그룹 분포 SVG를 구현했다.
- [x] Colab 3 epoch 종료마다 RouterArena inference-only 결과·SVG를 저장하고 중단 시 누락 평가를 복구하도록 구현했다.
- [x] RouterArena 관찰이 RNG·loss·optimizer·early stopping·checkpoint 선택에 영향을 주지 않는 계약을 고정했다.
- [x] `model-package-v1` 생성·필수 파일·라벨·checksum 검증을 구현했다.
- [x] 절대 metric과 baseline delta deploy gate core를 구현했다.
- [x] KFP component 6개, pipeline 3개와 IR compile test를 구현했다.
- [x] 전체 ML 단위·계약·compile test 68개를 통과했다.
- [x] QQP ML 요소를 제거하고 온라인 cache 책임으로 문서화했다.

### C. 백엔드·인프라 협의 전 문서 검토

- [x] ML 코드 구조·역할·artifact·pipeline을 이 문서에 통합했다.
- [x] 흩어진 ML 체크리스트를 이 문서 하나로 통합했다.
- [x] 백엔드·인프라 협의 명세와 필요한 산출물을 별도 문서로 정리했다.
- [x] 팀원 API 초안과 ML 제안을 대조한 회의 진행용 결정 문서를 작성했다.
- [x] GKE를 사용하지 않는 Vertex AI Pipelines CPU → Vertex AI T4 학습 구조를 확정안으로 반영했다.
- [ ] 사용자가 이 문서의 구조·책임 경계·미구현 항목을 검토한다.
- [x] `졸업프로젝트 개요.md`의 GKE GPU/평가/파이프라인 표현을 확정 구조로 정정한다.
- [ ] MVP·졸업프로젝트 개요의 나머지 충돌별 채택 기준을 팀이 확정한다.
- [ ] 백엔드·인프라 담당자에게 협의 명세와 fixture를 전달한다.
- [ ] 팀원에게 확정 인프라 구조와 backend 영향 없음/인수인계 지점을 통보하고 수신을 기록한다.
- [ ] 회의 결과를 이 문서와 contract fixture에 반영한다.

### D. 현재 코드에서 남은 구현·실험

- [ ] RouterBench test split 전용 evaluator와 under/over-routing report를 구현한다.
- [ ] RouterArena robustness 원본/변형 prediction consistency를 구현한다.
- [ ] 실제 full checkpoint로 RouterBench·RouterArena 평가를 실행한다.
- [ ] 전체 학습의 early stopping·최종 checkpoint 선택 정책을 구현한다.
- [ ] Vertex worker 중단·timeout을 위한 checkpoint resume 정책을 구현한다.
- [ ] 실제 기준 모델을 만든 뒤 deploy gate metric 임계값을 확정한다.
- [x] 통과 후보를 `approved/`로 승격하는 코드와 감사 기록을 구현한다. 승인 주체 이름은 팀이 확정한다.
- [ ] package directory만 받는 온라인용 reference predictor/contract fixture를 제공한다.
- [ ] Claude Haiku/Sonnet/Opus 표본으로 RouterBench proxy 라벨을 검증한다.
- [x] RouterBench 내부 고정 split probe로 solve-rate 후보 E/F/G를 비교하고, G(`0.64/0.27`)를 사용하는 v3 라벨·자연분포 artifact를 생성한다.
- [ ] v2/v3 ModernBERT 결과는 역사적 기준선으로 보존하고, 신규 학술적 winner와 Pareto
  권고는 통합 RouterBench+xRouteBench 비교 실험의 사전 고정 규칙으로 결정한다.
- [ ] 통합 train에서 `t_hard={0.18,0.27,0.36}` × `t_easy={0.55,0.64,0.73}` 9개를
  5-fold duplicate-group OOF TF-IDF와 label bootstrap 목적함수로 평가해 한 후보를 동결한다.
- [ ] xRouteBench immutable revision·checksum과 `llmrouter_generic` query pivot을 구현한다.
- [ ] RouterBench RAG와 xRouteBench memory/RAG를 제외하고 두 source의 exact/near duplicate를
  external test > test > validation > train 우선순위로 제거한다.
- [ ] RouterBench+xRouteBench 통합 artifact에 선택된 solve-rate threshold,
  continuous acceptance `0.70`, xRouteBench 최소 유효 모델 `12/18`, natural train과
  source별 manifest 계약을 구현한다.
- [ ] 공통 512-token 비교 helper가 ModernBERT, DeBERTa-v3-base, RoBERTa-base,
  MiniLM-L12-H384와 Qwen2.5-0.5B label-token 학습을 동일 artifact·평가 계약으로 실행하고,
  고정 TF-IDF+LogisticRegression 기준선을 함께 산출하도록 구현한다.
- [ ] `ml/notebooks/router_model_comparison/`에 데이터/threshold `00`, 모델별 `01~06`,
  선택·최종평가 `07` Colab notebook을 구현한다. 다중 모델 비교 KFP pipeline은 만들지 않는다.
- [ ] Colab GPU 200-step pilot으로 모델별 ETA·VRAM을 갱신하고, Drive 원자 checkpoint와
  새 세션 resume를 검증한 뒤 1차 seed 42 sweep을 수행한다.
- [ ] 비교 winner와 Pareto 권고를 보고하되 이번 continual pipeline·자동 승격은
  ModernBERT만 유지한다. 통합 ModernBERT package-v1과 hot-swap/rollback을 검증하고,
  다른 family serving adapter는 구현 범위에서 제외한다.

### E. 확정 인프라 전환 코드 체크리스트

#### E-1. Vertex AI 학습 launcher와 설정

- [x] `modernbert_training_component`를 Google Cloud Pipeline Components의
  `create_custom_training_job_from_component`로 감싼 Vertex AI CustomJob component를 사용한다.
- [ ] launcher가 project, `us-west1`, training image digest, input/config/output GCS URI,
  service account, machine type, accelerator와 제한 시간을 명시적으로 받게 한다.
- [x] 기본 worker pool을 `n1-standard-4`, `NVIDIA_TESLA_T4=1`, replica 1로 제한하고
  `restart_job_on_worker_restart=false`, retry 0, 최대 실행 6시간을 적용한다.
- [ ] 동일 KFP run·candidate version의 재실행이 중복 Vertex job을 만들지 않도록
  deterministic display name/label과 기존 job 조회 규칙을 정의한다.
- [ ] launcher의 `gcp_resources` 출력에 더해 image digest,
  checkpoint/metrics GCS URI와 시작·종료 시각을 포함한다.
- [x] 공식 CustomJob launcher가 Vertex job 실패·취소·timeout을 KFP task 실패로 변환해
  후속 평가·gate를 막는다.
- [x] production용 `gcloud ai custom-jobs create` 직접 실행 스크립트는 만들지 않고,
  운영 run이 항상 Vertex AI PipelineJob에서 시작되도록 한다.

#### E-2. artifact·학습 image 계약

- [ ] 기존 `modernbert-train` CLI와 GPU image가 Vertex custom container에서도 동일하게
  동작하도록 GCS 입력 다운로드·출력 업로드 wrapper 또는 sidecar 경계를 구현한다.
- [ ] KFP artifact의 local `.path`를 Vertex job에 넘기지 않고 immutable `gs://` URI만
  넘기도록 initial/retraining pipeline의 importer·output 계약을 변경한다.
- [ ] checkpoint와 `metrics.json` 업로드 완료를 `_SUCCESS` 또는 manifest/checksum으로
  확인한 뒤에만 launcher가 성공하도록 한다.
- [ ] Artifact Registry image는 mutable tag가 아니라 digest로 pipeline parameter와
  MLMD metadata에 기록한다.
- [ ] Vertex service account가 GCS input read/output write와 Artifact Registry pull만
  수행하도록 최소 권한을 적용한다.

#### E-3. KFP pipeline·CPU 평가 전환

- [x] `ml/kubeflow/task_config.py`의 `configure_gpu_training_task`와
  `configure_gpu_evaluation_task`를 제거하고 일반 component는 CPU 정책만 사용한다.
- [x] `modernbert_training.py`와 `modernbert_retraining.py`가 공식 CustomJob wrapper의
  Vertex job과 artifact 출력을 사용하도록 변경한다.
- [x] `model_qualification.py`와 RouterArena component에서 GPU 요청을 제거하고 CPU
  resource·batch 설정을 둔다.
- [ ] `llm-router-modernbert-evaluation:cu124-v1`을 CPU inference image로 교체하고
  100행 및 RouterArena full 예상 시간·메모리를 측정한다.
- [x] 초기 pipeline에 RouterBench CPU 검증·라벨링·전처리 component를 연결했다. 단, 이
  pipeline은 compile regression용 역사적 IR이며 실제 cloud 실행 대상으로 사용하지 않는다.
- [ ] Colab 초기학습 checkpoint를 package-v1 qualification 입력으로 가져오는 checksum 기반
  importer를 구현한다.
- [ ] feedback-retraining과 qualification pipeline의 Vertex job ID, GCS artifact URI와 gate
  결과를 KFP/MLMD에 연결한다.
- [x] 컴파일된 IR YAML에 `nvidia.com/gpu`, GPU node selector/toleration과 GPU node pool
  전제가 남아 있지 않은지 contract test로 검사한다.

#### E-4. 단위·계약·cloud 검증

- [x] fake Vertex client로 submit→poll→success/failure/cancel 단위 테스트를 만든다.
- [ ] 잘못된 project/region, mutable image tag, `file://`/local path, accelerator count>1,
  retry>0 또는 timeout>6h를 차단하는 config test를 만든다.
- [x] initial/retraining/qualification KFP IR compile test의 기대 DAG를 새 구조로 갱신한다.
  initial IR은 실행 승인이 아니라 회귀 검사용이다.
- [ ] ModernBERT feedback 100행 cloud smoke에서 Vertex PipelineJob → Vertex T4 → GCS
  checkpoint → CPU 평가·gate가 한 번의 lineage로 연결되는지 검증한다. 초기학습과 다른
  비교 모델은 이 경로에 넣지 않는다.
- [ ] 같은 입력 재실행의 KFP cache와 Vertex 중복 제출 방지, 실패 job cleanup을 검증한다.

### F. GCP/GCS/Vertex AI 설정 전 확인

- [x] project를 `project-763753da-fa8f-4afd-b31`(`HYUCapstone`)으로 고정했다.
- [x] 2026-08-15 확정, 2026-08-14 `us-central1` 기준을 대체: 기본 region을 `us-west1`로
  바꿨다. GCS Always Free는 `us-east1`/`us-west1`/`us-central1` 셋 다 동일하게
  적용되고, `us-west1`(오리건)이 한국에서의 인터랙티브 작업(콘솔·`gcloud`·업로드)
  지연이 더 짧다. T4 quota는 `us-west1`에서도 별도로 확인해야 한다(G-0-3).
- [x] GKE cluster와 자체 설치 Kubeflow를 만들지 않기로 확정했다.
- [x] feedback 재학습 GPU 실행만 Vertex AI T4 1장 CustomJob으로 제한했다. 초기학습은
  Colab T4에서 실행한다.
- [ ] billing admin이 Welcome Credit 잔액과 원래 만료일을 확인해 기록한다.
- [ ] 실사용 가능 T4 quota를 확인하고 accelerator quota를 1로 제한한다.
- [ ] globally unique GCS bucket, Artifact Registry repository와 Vertex pipeline root 이름을 정한다.
- [ ] Vertex Spend Cap Preview 사용 가능 여부를 확인하고 가능한 최소 초기 한도를 건다.
- [ ] budget alert와 알림 채널을 설정한다. 이는 hard spending cap이 아님을 runbook에 쓴다.
- [ ] Vertex PipelineJob·CustomJob service account의 최소 IAM을 설계한다.
- [ ] secret을 source/image/pipeline parameter에 넣지 않는 방식을 확인한다.

### G. 인프라 생성과 cloud smoke

- [ ] 필요한 Google API를 활성화한다.
- [ ] `us-west1` GCS bucket과 Artifact Registry repository를 생성한다.
- [ ] CPU data/promotion/evaluation image와 GPU training image를 immutable digest로 push한다.
- [ ] Vertex AI Pipelines와 Custom Training API를 활성화한다.
- [ ] PipelineJob service account의 Vertex submit·GCS·Artifact Registry 권한을 연결한다.
- [ ] pipeline root와 versioned config/dataset artifact를 업로드한다.
- [ ] 컴파일 IR에 Kubernetes GPU resource가 없고 CustomJob worker pool에만 T4가 있음을 확인한다.
- [ ] Colab 초기학습 checkpoint를 GCS qualification 입력으로 import하는 CPU smoke를 수행한다.
- [ ] feedback 100행 재학습 cloud smoke를 Vertex AI PipelineJob으로 제출한다.
- [ ] Vertex 동시 job 1·retry 0·6시간 timeout, Artifact 전달과 MLMD lineage를 확인한다.
- [ ] smoke 종료 직후 Vertex job과 임시 artifact cleanup을
  검증하고 실제 비용을 확인한다.
- [ ] 남은 credit보다 보수적인 feedback 재학습 1회 비용 한도를 계산한 뒤 cloud smoke를 승인한다.

### H. 전체 학습·평가·승격

- [x] Colab에서 RouterBench v2 balanced ModernBERT baseline 3 epoch를 학습한다.
- [ ] RouterBench+xRouteBench 통합 artifact로 `modernbert_full_colab.ipynb` 운영 초기학습을
  실행하고 Drive 최종 checkpoint·checksum·`_SUCCESS`를 동결한다.
- [ ] RouterBench holdout과 RouterArena full/robustness를 평가한다.
- [ ] domain/source/truncation과 실패·저신뢰 표본을 분석한다.
- [ ] baseline과 후보용 gate rules를 versioned JSON으로 확정한다.
- [ ] candidate package가 gate를 통과하고 approved artifact로 승격되는지 검증한다.
- [ ] 이전 approved model rollback을 보존한다.

### I. 온라인 통합과 재학습

- [ ] 합의된 Router API contract test를 Spring stub과 실제 predictor 사이에서 통과한다.
- [ ] model load 실패·timeout·unknown label fallback을 통합 smoke한다.
- [ ] 실제 inline feedback을 Spring fixture로 FastAPI에 보내고, FastAPI가 만든 GCS
      Parquet/manifest를 Vertex 입력으로 전달한다.
- [ ] prepare→merge→retrain→evaluate→package→gate→publish→approve를 실제 batch로 실행한다.
- [x] 같은 batch 재실행이 중복 학습을 만들지 않도록 Vertex `job_id` 멱등성을 구현한다.
- [ ] 같은 batch 재실행이 실제 Vertex에서 중복 학습을 만들지 않는지 검증한다.
- [x] gate 통과 모델을 자동 승인하고 GCS `CURRENT.json`을 원자 갱신하는 경로를 구현한다.
- [x] FastAPI가 `CURRENT.json`을 polling하고 검증된 model을 무중단 교체하는 경로를 구현한다.
- [ ] 자동 승인·모델 전환·health check·rollback을 실제 배포에서 end-to-end로 검증한다.
- [ ] 운영 관측 화면이 필요해지면 승인 UI가 아니라 다음 읽기 전용 범위를 제공한다.
      - candidate 목록과 각각의 RouterBench/RouterArena 지표, baseline 대비 델타를
        한 화면에서 본다.
        - deploy gate 통과 여부와 개별 rule의 pass/fail을 표로 보여준다
          (`gate_result.json`은 현재 pipeline-root에만 남고 `candidate/`에는 올라가지
          않는다 — 대시보드가 읽으려면 게시 경로를 바꾸거나 별도로 노출해야 한다).
      - 클래스별·도메인별 breakdown을 펼쳐볼 수 있게 한다. 자동 게이트가 못 잡는
        판단(집계 지표 뒤 클래스 붕괴, 과적합 방향, 도메인 편중)이 검토의 핵심이므로
        raw JSON이 아니라 이 breakdown이 우선이다.
      - 자동 승인 주체와 전환 시각, 이전·현재 `model_version`을 표시한다.
      - rollback action은 기존 `modernbert-publish rollback`의 generation
        precondition을 반드시 재사용한다.

### J. 최종 결과물

- [ ] always-Opus 대비 비용·품질·모델 분포를 계산한다.
- [ ] 온라인 팀의 cache hit/false-hit/latency 결과를 별도 축으로 받는다.
- [ ] model/data/code/evaluation version을 모든 표·그래프에 표시한다.
- [ ] 반복 실행 seed와 평균·분산 또는 신뢰구간을 기록한다.
- [ ] model card, data card와 cloud/rollback runbook을 작성한다.
- [ ] 보고서·발표자료용 metric artifact를 freeze한다.

## 12. 현재 알려진 미구현·제약

- Colab ModernBERT v2 전체 학습과 RouterBench validation·RouterArena 성능 수치는
  `training-reports/modernbert-v2-balanced-training-report.md`에 있다. RouterBench test
  전용 결과, RouterBench+xRouteBench용 `00~07` Colab 통합 비교와 통합-data Colab
  ModernBERT 운영 초기학습 결과는 아직 없다. 초기학습 Vertex 재현 run은 계획하지 않는다.
- 학습기는 validation metric을 계산하지만 RouterBench test 전용 평가 CLI는 없다.
- RouterArena robustness consistency는 전처리만 있고 metric 계산은 없다.
- deploy gate engine과 GCS candidate 게시·approved 승격 코드는 있으나 운영 threshold는 없다.
- KFP IR은 Vertex AI CustomJob T4 학습과 CPU 평가 구조로 컴파일되지만 initial IR은
  회귀검사용으로만 보존한다. 실제 feedback-retraining PipelineJob, GCS와 Vertex ML
  Metadata cloud smoke는 아직 실행하지 않았다.
- feedback v0.1은 합성 fixture 계약이며 운영 DB schema로 확정되지 않았다.
- FastAPI serving과 Spring 통합은 온라인 팀 협의 후 진행해야 한다.
- FastAPI의 Vertex PipelineJob 제출과 승인 모델 다운로드는 구현·단위 테스트를 마쳤으나
  실제 Vertex·GCS 왕복은 검증하지 않았다.

## 13. 세부 참고 문서

아래 문서는 진행 체크리스트가 아니라 설계 근거와 실행 세부사항이다.

- `docs/졸업프로젝트 개요.md`: 전체 프로젝트 기준과 역할 축
- `docs/ml/routerbench-data-labeling-spec.md`: 라벨링 판정식·전처리 근거
- `docs/ml/routerarena-evaluation-preprocessing-spec.md`: RouterArena 입력·제외 규칙
- `docs/ml/routerarena-routing-evaluation-spec.md`: 평가 metric 의미
- `docs/ml/modernbert-training-spec.md`: 학습·runtime 설계 근거
- `docs/ml/backend-infrastructure-agreement.md`: 백엔드·온라인·인프라 회의 안건과 계약
- `docs/ml/gcp-rollout-and-serving-integration-plan.md`: GCP 설정 runbook, 미구현 코드
  항목과 승인 모델의 FastAPI 전달 경로 구현 명세
- `docs/ml/development-questions-and-decisions.md`: 개발 질문, 선택 근거와 미해결 고민
- `docs/ml/llm-routing-benchmark-dataset-analysis.md`: 벤치마크 점수·라벨 생성 근거
- `docs/ml/llm-routing-model-specs-report.md`: 라우팅 모델·학습 데이터 조사
- `docs/ml/integrated-routing-data-and-model-comparison-plan.md`: 신규 실험의 전체 flow,
  RouterBench+xRouteBench 통합, threshold 후보·목적함수, 최종 모델군, Vertex/GCS,
  FastAPI·API 실행 계약
- `docs/mvp-scope-and-checklist.md`: 2026-07 시점의 프로젝트 범위 초안
- `ml/README.md`: 실제 로컬 CLI와 Docker 실행 명령
