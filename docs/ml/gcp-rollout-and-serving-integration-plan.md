# GCP 배포와 serving 연동 실행 계획

상태: **실행 계획 문서**

기준일: 2026-08-15

> 2026-09-16 통합 주석: 이 문서의 상태 표는 구현 전후의 gap을 추적한 시점 기록이라
> 일부 “미구현” 표기가 현재 코드와 다르다. 신규 RouterBench+xRouteBench 비교,
> threshold 목적함수, ModernBERT-only 재학습/FastAPI 경계와 최신 실행 순서는
> `integrated-routing-data-and-model-comparison-plan.md` 10~15장을 우선하고, 실제 체크 상태는
> `workstream-checklist.md`만 갱신한다. 이 GCP runbook의 KFP/Vertex 경로는 ModernBERT에만
> 적용한다. TF-IDF·MiniLM·RoBERTa·DeBERTa·Qwen을 포함한 모델 비교는 Colab notebook으로
> 실행한다. ModernBERT 초기학습도 Colab에서 실행하며, Vertex T4 CustomJob은 feedback
> 재학습에만 사용한다. 비교용·초기학습용 PipelineJob을 제출하지 않는다.

브랜치: `feat/data-labeling-training`

대상 저장소: `LLM_Router`(ML), `LLM_Router-serving`(FastAPI)

## 1. 문서 목적과 사용법

이 문서는 다음 두 가지만 다룬다.

1. Colab 초기 checkpoint를 GCS package로 인수하고, GCP의 ModernBERT feedback 재학습
   파이프라인 결과를 `LLM_Router-serving` FastAPI에 전달하기까지의 구현 명세
2. 그 코드를 실제로 실행하기 위해 **GCP에서 무엇을 어떤 순서로 설정해야 하는지**

### 1-1. 다른 문서와의 관계

- ML 작업 진행 상태 체크박스는 `docs/ml/workstream-checklist.md` 11장에만 둔다.
  이 문서는 체크박스를 두지 않고 **항목 ID와 상태 열**로 관리한다. 항목이 끝나면
  총괄 체크리스트의 대응 항목을 갱신한다.
- 팀 합의가 필요한 값(bucket 이름, 승인 주체, SA 이름 등)은
  `docs/ml/backend-infrastructure-agreement.md`에 결정자·결정일과 함께 기록한다.
- 실행 구조(Vertex AI Pipelines CPU → Vertex AI Custom Training T4, GKE 미사용)는
  feedback 재학습에만 적용한다. 초기학습은 Colab + Drive checkpoint를 전제로 한다.

### 1-2. 항목 ID 규칙

| 접두사 | 범위 |
| --- | --- |
| `C-` | 계약 불일치 해소 (cross-repo) |
| `M-` | `LLM_Router` ML 코드 |
| `S-` | `LLM_Router-serving` FastAPI 코드 |
| `D-` | 승인 모델 전달·교체 경로 |
| `T-` | 테스트 |
| `G-` | GCP 설정·인프라 작업 |

상태 값은 `미구현`, `부분 구현`, `구현됨(cloud 미검증)`, `완료`다.

## 2. 현재 구현 상태 실측 요약

2026-08-15 기준으로 두 저장소 코드를 직접 확인한 결과다.

### 2-1. `LLM_Router` (ML)

| 영역 | 상태 |
| --- | --- |
| RouterBench 라벨링·v2 학습 전처리 | 구현·로컬 실행 완료 |
| RouterArena 전처리·routing 평가 | 구현 완료 (평가 image는 CUDA base로 남아 있음) |
| feedback prepare/merge | 구현 완료 (합성 fixture 기준) |
| ModernBERT 학습 코어와 `modernbert-train` CLI | 구현 완료 |
| KFP v2 component 7개, pipeline 3개 | 구현·compile test 통과 |
| Vertex CustomJob wrapper (`vertex_training.py`) | 구현됨(cloud 미검증) |
| Docker image 4종 | 로컬 build·CLI smoke 통과, registry push 없음 |
| **GCS 입출력 코드** | **없음.** `ml/src`, `ml/kubeflow` 어디에도 `google-cloud-storage`/`gcsfs` 사용이 없고 모든 CLI가 로컬 경로만 받는다 |
| **approved 승격 코드** | **없음.** `deploy_gate`는 통과/실패 판정과 `gate_result.json`만 만들고 어떤 GCS 경로에도 쓰지 않는다 |
| **package metadata 생성** | **없음.** `package_metadata_uri`가 정적 GCS 파일이라 `model_version`이 run마다 갱신되지 않는다 |

### 2-2. `LLM_Router-serving` (FastAPI)

| 영역 | 상태 |
| --- | --- |
| `POST /router/embed`, `POST /router/classify`, `GET /health` | 구현 완료, 로컬 테스트 통과 |
| `POST /escalate` 검증·멱등성·SQLite lease | 구현 완료 |
| GCS manifest/events 검증 | 구현 완료 |
| **pipeline 제출 backend** | **잘못된 전제.** `retraining/pipeline.py`가 `kfp.Client(host=...)` 기반 Kubeflow Standalone REST를 호출한다 |
| **재학습 결과 조회 API** | **없음.** route는 embed/classify/health/escalate 4개뿐이다 |
| **모델 교체(hot swap)** | **없음.** `ModelManager`에 재적재 진입점이 없고 `router_model_path`는 프로세스 시작 시 고정된 환경변수다 |
| **rollback** | **없음** |

## 3. 최우선 항목: KFP Standalone → Vertex AI Pipelines 계약 불일치

`docs/ml/backend-infrastructure-agreement.md`와 총괄 체크리스트는 GKE와 자체 설치
Kubeflow를 쓰지 않고 관리형 **Vertex AI Pipelines**에서 KFP v2 IR을 실행한다고 확정했다.
그런데 serving의 재학습 제출 코드는 아직 Kubeflow Standalone 전제다. 이 상태로는
`POST /escalate`가 GCP에서 절대 동작하지 않는다. 나머지 모든 연동 작업의 선행 조건이다.

### 3-1. 불일치 지점

| 위치 | 현재 | 필요 |
| --- | --- | --- |
| `pyproject.toml` | `kfp==2.17.0` | `google-cloud-aiplatform` 추가. serving은 IR을 컴파일하지 않으므로 `kfp` 런타임 의존은 제거 가능 |
| `retraining/pipeline.py` | `kfp.Client(host, existing_token, namespace)` | `google.cloud.aiplatform.PipelineJob` |
| 제출 호출 | `create_run_from_pipeline_package(package_path, experiment_name=...)` | `PipelineJob(template_path=gs://..., pipeline_root=..., job_id=..., parameter_values=..., enable_caching=False).submit(service_account=...)` |
| 상태 조회 | `client.get_run(run_id).state` (`SUCCEEDED`/`RUNNING`/...) | `aiplatform.PipelineJob.get(resource_name).state` (`PIPELINE_STATE_SUCCEEDED`/`PIPELINE_STATE_FAILED`/`PIPELINE_STATE_CANCELLED`/`PIPELINE_STATE_RUNNING`/`PIPELINE_STATE_PENDING`/`PIPELINE_STATE_QUEUED`/`PIPELINE_STATE_PAUSED`/`PIPELINE_STATE_CANCELLING`) |
| 설정 | `kfp_host`, `kfp_token`, `kfp_namespace`, `kfp_pipeline_package_path`(로컬 파일) | `vertex_project`, `vertex_location`, `vertex_pipeline_root`, `vertex_pipeline_template_uri`(GCS), `vertex_pipeline_service_account` |
| 인증 | bearer token 환경변수 | 부착 SA / Workload Identity의 ADC. 토큰 환경변수 없음 |

### 3-2. 멱등성을 Vertex `job_id`로 승격

Vertex PipelineJob의 `job_id`는 `[a-z0-9][-a-z0-9]{0,127}` 패턴이고 project·location
안에서 유일하다. 같은 `job_id`로 재제출하면 `AlreadyExists`(409)가 난다. 즉 **Vertex가
서버 측 멱등성 키를 이미 제공한다.**

따라서 `batchId`를 정규화한 값을 `job_id`로 쓰고, `AlreadyExists`를 "이미 제출됨"으로
해석해 기존 `resource_name`을 반환하도록 한다. SQLite lease는 그대로 두되, 프로세스가
죽어 lease를 잃어도 중복 학습이 발생하지 않는 2중 방어가 된다.

- `job_id` = `retrain-` + `batchId`를 소문자화하고 `[^a-z0-9-]`를 `-`로 치환한 값
- 치환 결과가 서로 충돌하지 않도록 `batchId` 원문의 SHA-256 앞 8자리를 접미로 붙인다
- `pipelineRunId`로 Spring에 반환하는 값은 Vertex `resource_name`
  (`projects/<num>/locations/us-west1/pipelineJobs/<job_id>`)으로 고정한다

### 3-3. 파이프라인 파라미터 누락

`EscalationService._pipeline_arguments()`가 넘기는 값은 12개다.
`modernbert_retraining_pipeline`에는 다음 파라미터가 더 있고, 넘기지 않으면 IR에 박힌
기본값(`replace-me`, `replace-project`)이나 빈 문자열이 쓰인다.

| 파라미터 | 미전달 시 결과 |
| --- | --- |
| `data_processing_image` | compile 시점 기본값 사용. digest 고정 IR이면 문제없음 |
| `evaluation_image`, `promotion_image` | 동일. 단 현재 기본값이 `replace-project`다 |
| `training_service_account` | 빈 문자열 → Vertex 기본 compute SA 사용. 최소 권한 설계가 무력화된다 |
| `training_network` | 빈 문자열 → 기본 네트워크 |
| `training_base_output_directory` | 빈 문자열 → CustomJob 출력 경로가 지정되지 않는다 |

### 3-4. 항목

| ID | 내용 | 저장소 | 상태 |
| --- | --- | --- | --- |
| C-1 | `retraining/pipeline.py`를 `VertexPipelineClient`로 교체하고 `PipelineClient` Protocol은 유지한다 | serving | 구현됨(cloud 미검증) |
| C-2 | `Settings`의 `kfp_*`를 `vertex_*`로 교체하고 `.env.example`·README를 갱신한다 | serving | 구현됨(cloud 미검증) |
| C-3 | `google-cloud-aiplatform`을 의존성에 고정하고 `kfp` 런타임 의존을 제거한다 | serving | 구현됨(cloud 미검증) |
| C-4 | `batchId` → `job_id` 정규화와 `AlreadyExists` 처리를 구현한다 | serving | 구현됨(cloud 미검증) |
| C-5 | Vertex `PipelineState` enum 매핑을 `ACTIVE`/`TERMINAL`/`UNKNOWN`으로 다시 작성한다 | serving | 구현됨(cloud 미검증) |
| C-6 | `training_service_account`, `training_base_output_directory`, publish root 파라미터를 설정에서 읽어 전달한다 | serving | 구현됨(cloud 미검증) |
| C-7 | `template_path`를 로컬 파일이 아니라 GCS URI로 받도록 바꾸고 `is_file()` 검증을 GCS URI 검증으로 대체한다 | serving | 구현됨(cloud 미검증) |

2026-08-15 구현 완료. `pyproject.toml`은 `google-cloud-aiplatform==1.164.0`을 고정하고
`kfp` 런타임 의존을 제거했다. serving 42 tests가 통과하며 fake Vertex client로
submit·상태 매핑·`AlreadyExists`·권한 오류를 검증한다. 실제 Vertex 제출은 미검증이다.

### 3-5. 참고 구현 형태

```python
from google.cloud import aiplatform
from google.api_core.exceptions import AlreadyExists

job = aiplatform.PipelineJob(
    display_name=f"retrain-{batch_id}",
    template_path=settings.vertex_pipeline_template_uri,  # gs://.../feedback-retraining-pipeline.yaml
    pipeline_root=settings.vertex_pipeline_root,
    job_id=job_id,                    # batchId에서 결정적으로 파생
    parameter_values=arguments,
    enable_caching=False,
    labels={"batch-id": label_safe_batch_id, "component": "feedback-retraining"},
    project=settings.vertex_project,
    location=settings.vertex_location,
)
try:
    job.submit(service_account=settings.vertex_pipeline_service_account)
except AlreadyExists:
    job = aiplatform.PipelineJob.get(resource_name=..., project=..., location=...)
run_id = job.resource_name
```

`submit()`은 비동기이므로 `run()`이 아니라 `submit()`을 쓴다. `run()`은 블로킹이라
FastAPI 요청 스레드를 최대 6시간 점유한다.

## 4. `LLM_Router` ML 코드 체크리스트

### 4-1. GCS 입출력 경계 (M-1 ~ M-5)

현재 ML 코드에는 GCS 코드가 전혀 없고, Vertex의 GCS FUSE(`/gcs/<bucket>/<path>`)에
전적으로 의존한다. KFP v2 artifact의 `.path`는 Vertex에서 `/gcs/...`로 해석되므로
CPU component는 대체로 동작할 가능성이 높지만, 아래는 cloud smoke 전에 확정해야 한다.

| ID | 내용 | 상태 |
| --- | --- | --- |
| M-1 | CustomJob 컨테이너 안에서 `Input[Dataset].path`가 `/gcs/...`로 마운트되는지 100행 smoke로 실측한다. 안 되면 학습 wrapper에 `gs://` 다운로드/업로드 단계를 추가한다 | 미구현 |
| M-2 | checkpoint(`model.safetensors` 수백 MB)를 GCS FUSE에 직접 쓸 때의 소요 시간·부분 쓰기 위험을 측정하고, 로컬 디스크에 쓴 뒤 일괄 복사하는 방식과 비교한다 | 미구현 |
| M-3 | 학습 산출물 업로드 완료를 `_SUCCESS` 마커 또는 `checksums.json`으로 확인한 뒤에만 성공 처리한다. 현재는 파일 존재만으로 다음 task가 진행된다 | 미구현 |
| M-4 | `training_base_output_directory`를 pipeline 파라미터로 실제 연결한다. 현재 기본값이 빈 문자열이다 | 부분 구현 |
| M-5 | 재실행 시 같은 GCS 경로를 덮어쓰지 않도록 출력 경로에 run ID 또는 candidate version을 포함시킨다 | 미구현 |

### 4-2. package metadata와 승격 (M-6 ~ M-9)

`model_package_component`는 `package_metadata`를 **정적 GCS artifact**로 받는다. 그
JSON 안에는 `model_version`, `dataset_version`, `code_commit` 등 9개 필수 필드가 들어
있고 `create_model_manifest()`가 그대로 manifest에 복사한다. 반면
`candidate_model_version`은 **별도 pipeline 파라미터**로 gate에만 전달된다.
`evaluate_deploy_gate()`는 manifest의 `model_version`과 `candidate_model_version`을
대조하지 않는다.

결과적으로 재학습을 100번 돌려도 모든 package의 `model_manifest.json`이 같은
`model_version`을 갖는다. serving이 model version으로 캐시·라우팅 계보를 구분할 수
없게 되므로 반드시 고쳐야 한다.

| ID | 내용 | 상태 |
| --- | --- | --- |
| M-6 | `modernbert-package assemble`에 `--model-version` override를 추가하고 `model_package_component`가 `candidate_model_version`을 그 값으로 전달한다. `package_metadata` 정적 파일의 `model_version`은 placeholder로 두고 매 run마다 override된다 | 구현됨(cloud 미검증) |
| M-7 | `evaluate_deploy_gate()`에 manifest `model_version` == `candidate_model_version` 검증을 추가한다 | 구현됨 |
| M-8 | gate 통과 시 package를 `models/router-modernbert/<version>/candidate/`에 게시하는 `model_publish_component`를 추가한다. `approved/` 승격은 `modernbert-publish approve`(pipeline 밖) | 구현됨(cloud 미검증) |
| M-9 | 승격 시 이전 approved를 `CURRENT.json`의 `previous_model_version`/`previous_package_uri`로 보존하고 `modernbert-publish rollback`으로 되돌린다 | 구현됨 |

M-6이 없으면 M-7의 gate 검증이 항상 실패한다는 점을 실제 업로드 준비 중 발견해 함께
고쳤다. `package_metadata_uri`는 모든 run이 공유하는 정적 파일이라 그 안의
`model_version`은 run마다 달라지는 `candidate_model_version`을 따라갈 수 없기
때문이다.

승격 주체는 별도 결정 항목이다. 9장 참조.

### 4-3. image와 IR (M-10 ~ M-13)

| ID | 내용 | 상태 |
| --- | --- | --- |
| M-10 | `llm-router-modernbert-evaluation`을 CUDA base에서 CPU inference base로 교체하고 RouterArena full 8,375행의 CPU 소요 시간·메모리를 실측한다. 현재 CPU task에 CUDA image를 붙여 image가 불필요하게 크고 pull이 느리다 | 미구현 |
| M-11 | 4개 image를 `linux/amd64`로 Artifact Registry에 push하고 digest를 기록한다 | 미구현 |
| M-12 | 4개 image 기본값을 모두 환경변수로 주입 가능하게 만든다: `LLM_ROUTER_VERTEX_TRAINING_IMAGE`, `LLM_ROUTER_VERTEX_EVALUATION_IMAGE`, `LLM_ROUTER_VERTEX_PROMOTION_IMAGE`, `LLM_ROUTER_VERTEX_DATA_PROCESSING_IMAGE` | 구현됨 |
| M-13 | 사용된 image digest를 pipeline 파라미터와 MLMD metadata에 남긴다 | 미구현 |

### 4-4. Colab 초기학습 인수 준비 (M-14 ~ M-16)

| ID | 내용 | 상태 |
| --- | --- | --- |
| M-14 | 승인된 통합 `training.parquet`, config와 manifest를 Colab 입력 archive/Drive에 게시하고 SHA-256을 기록한다 | 미구현 |
| M-15 | `modernbert_full_colab.ipynb`가 통합 dataset version을 검증하고 기존 Drive checkpoint resume 계약으로 초기학습하도록 갱신한다 | 미구현 |
| M-16 | Colab 최종 `_SUCCESS` checkpoint와 manifest를 package-v1 qualification 입력으로 가져오는 checksum 기반 importer를 구현한다 | 미구현 |

## 5. `LLM_Router-serving` 코드 체크리스트

3장의 C-1 ~ C-7 외에 추가로 필요한 항목이다.

| ID | 내용 | 상태 |
| --- | --- | --- |
| S-1 | `GET /escalate/{batchId}` 재학습 상태 조회 endpoint를 추가한다. 현재 Spring은 202를 받은 뒤 결과를 알 방법이 없다 | 미구현 |
| S-2 | 응답에 `state`(`running`/`succeeded`/`failed`/`cancelled`), `pipelineRunId`, `candidateModelVersion`, `gateStatus`, `finishedAt`를 포함한다 | 미구현 |
| S-3 | Vertex 상태를 매번 조회하지 말고 SQLite에 캐시하고 TTL을 둔다. `/escalate` 폴링이 Vertex API quota를 소모하지 않게 한다 | 미구현 |
| S-4 | `ObjectStore`에 `write_bytes`/`download_to_dir`를 추가한다. 현재 `read_bytes`만 있어 approved package를 내려받을 수 없다 | 미구현 |
| S-5 | 로그·응답에 `embedModelVersion`과 `routerModelVersion`을 함께 남겨 모델 교체 전후 요청을 구분할 수 있게 한다 | 부분 구현 |
| S-6 | `retraining_db_path`의 SQLite lease가 단일 replica 전제임을 배포 설정에서 강제한다(인스턴스 수 1 고정) | 미구현 |

## 6. 승인 모델을 FastAPI에 전달하는 경로

코드 구현은 완료했고 실제 배포 환경의 end-to-end 검증은 남아 있다. 파이프라인은
deploy gate를 통과한 모델을 candidate로 게시한 뒤 자동 승인하고 `CURRENT.json`을
원자적으로 갱신한다. FastAPI는 이 pointer를 polling해 package를 내려받고 검증한 다음
실행 중인 model 참조를 교체한다.

### 6-1. 확정 방식: GCS 게시 + 자동 승인 + hot reload

2026-09-06 변경 확정. FastAPI로 모델을 전송하는 API는 만들지 않는다. pipeline과
FastAPI 사이에는 **GCS 기반 pull 구조**를 사용한다. 사람의 사전 승인과 배포 설정 수정,
프로세스 재시작은 정상 승격 경로에서 제거했다. 새 model의 다운로드·manifest·checksum·
label 검증이 실패하면 기존 model을 유지하고 다음 polling 주기에 재시도한다.

### 6-2. 전달 흐름

```text
Vertex AI Pipeline
  -> 평가 -> model package -> deploy gate
  -> model_publish_component
  -> gs://<bucket>/models/router-modernbert/<version>/candidate/
  -> model_auto_approve_component가 publish 결과와 package를 재검증
  -> gs://<bucket>/models/router-modernbert/<version>/approved/ + CURRENT.json
  -> FastAPI가 CURRENT.json polling
  -> 새 package 다운로드 -> manifest·checksum·라벨·gate_status 검증
  -> 검증 완료 후 model 참조 원자 교체
  -> 이상 시 기존 model 유지 또는 modernbert-publish rollback으로 pointer 복귀
```

### 6-3. `approved/` 규약

```text
gs://<bucket>/models/router-modernbert/CURRENT.json
gs://<bucket>/models/router-modernbert/<model-version>/candidate/
gs://<bucket>/models/router-modernbert/<model-version>/approved/
gs://<bucket>/models/router-modernbert/<model-version>/approved/model_manifest.json
```

`CURRENT.json` 예시:

```json
{
  "schema_version": "router-current-model-v1",
  "model_version": "router-modernbert-base-feedback-20260901-a1b2c3d",
  "package_uri": "gs://<bucket>/models/router-modernbert/router-modernbert-base-feedback-20260901-a1b2c3d/approved/",
  "previous_model_version": "router-modernbert-base-routerbench-20260731-21c2f1b",
  "approved_at": "2026-09-01T04:12:30Z",
  "approved_by": "<승인 주체>",
  "pipeline_run_id": "projects/<num>/locations/us-west1/pipelineJobs/retrain-batch-0007-9f3ac1e2"
}
```

### 6-4. 항목

| ID | 내용 | 저장소 | 상태 |
| --- | --- | --- | --- |
| D-1 | `model_publish_component`가 gate 통과 package를 `candidate/`에 게시한다. gate 실패 candidate는 업로드 자체를 거부한다 | ML | 구현됨(cloud 미검증) |
| D-2 | gate 다음 자동 component가 `modernbert-publish approve`로 candidate를 재검증해 `approved/`로 복사하고 `gate_status`를 `approved`로 바꾼다 | ML | 구현됨(cloud 미검증) |
| D-3 | `CURRENT.json` 갱신을 GCS generation precondition(`if_generation_match`)으로 보호해 동시 승격을 막는다 | ML | 구현됨(cloud 미검증) |
| D-4 | 이미 current인 version 재승인과, 현재 승인 모델보다 오래된 candidate 승인을 거부한다(`--allow-pointer-rollback`으로만 허용) | ML | 구현됨 |
| D-5 | `modernbert-publish rollback`이 `previous_model_version`으로 포인터를 되돌린다 | ML | 구현됨 |
| D-6 | FastAPI가 시작 시와 실행 중 `CURRENT.json`을 읽고 approved package를 로컬 cache에 둔다 | serving | 구현됨(cloud 미검증) |
| D-7 | 다운로드한 package를 manifest 기준으로 검증한다(schema version, `id2label`, `model_version`, 파일별 SHA-256, `gate_status == "approved"`) | serving | 구현됨 |
| D-8 | 검증 통과 후에만 cache 디렉터리에 게시해, 실패한 다운로드가 사용 가능한 모델을 남기지 않게 한다 | serving | 구현됨 |
| D-9 | 로컬 경로와 명시적 GCS source를 동시에 설정하면 시작 시 거부한다 | serving | 구현됨 |
| D-10 | 이미 받은 version은 재검증만 하고 재다운로드하지 않는다 | serving | 구현됨 |
| D-11 | 새 model을 완전히 검증·적재한 뒤 참조를 원자 교체하고 실패 시 이전 model을 유지한다 | serving | 구현됨(cloud 미검증) |
| D-12 | cache 디렉터리를 `var`와 같은 영속 볼륨에 두어 컨테이너 교체마다 재다운로드하지 않게 한다 | 배포 | 미구현 |

2026-09-06 자동 승인·hot reload까지 구현 완료. ML 쪽은 `ml/src/llm_router_labeling/deployment/model_publish.py`와
`modernbert-publish` CLI, `ml/src/llm_router_labeling/shared/object_store.py`이고, serving
쪽은 `src/llm_router_serving/model_source.py`다. ML 85 tests, serving 42 tests가
통과한다. 실제 GCS 왕복은 미검증이다.

외부에서 호출하는 `POST /admin/models/reload` endpoint는 두지 않는다. FastAPI 내부
watcher만 `CURRENT.json` 변경을 감지해 model을 교체한다.

## 7. 테스트 체크리스트

| ID | 내용 | 상태 |
| --- | --- | --- |
| T-1 | fake Vertex client로 submit과 8개 `PipelineState` 매핑, 권한 오류 단위 테스트 | 구현됨 |
| T-2 | 같은 `batchId` 재제출이 `AlreadyExists`를 만나 기존 `resource_name`을 반환하는 테스트 | 구현됨 |
| T-3 | 잘못된 project/region, mutable image tag, `file://`/로컬 경로, accelerator count > 1, retry > 0, timeout > 6h를 차단하는 config 검증 테스트 | 미구현 |
| T-4 | fake GCS로 publish → approve → `CURRENT.json` → 다운로드 → 검증 전체 경로 테스트 | 구현됨 |
| T-5 | 손상된 package(checksum 불일치, `gate_status != approved`, label 순서 오류, version 불일치)에서 적재가 거부되는지 테스트 | 구현됨 |
| T-6 | gate 실패 candidate가 GCS에 올라가지 않고, 오래된 candidate 승인이 거부되는지 테스트 | 구현됨 |
| T-10 | 컴파일 IR에서 publish task가 gate 뒤에 오고 accelerator를 요청하지 않는지 검사 | 구현됨 |
| T-7 | 컴파일 IR에 `nvidia.com/gpu`, GPU node selector/toleration이 없는지 검사 | 완료 |
| T-8 | 100행 cloud smoke: Vertex PipelineJob → T4 CustomJob → GCS checkpoint → CPU 평가·package·gate가 한 lineage로 연결되는지 | 미구현 |
| T-9 | 실제 batch로 prepare → merge → retrain → evaluate → package → gate → 승격 → reload end-to-end | 미구현 |

## 8. GCP 설정 runbook

아래 명령은 `us-west1`, project `project-763753da-fa8f-4afd-b31`(`HYUCapstone`)
기준이다. `<bucket>`, `<repo>` 등 대괄호 값은 팀에서 확정한 뒤 치환한다.

```bash
export PROJECT_ID=project-763753da-fa8f-4afd-b31
export REGION=us-west1
gcloud config set project "$PROJECT_ID"
```

### G-0. 착수 전 필수 확인 (GPU 실행 금지 구간)

비용 결정에 따라 **credit 잔액과 원래 만료일을 확인하기 전에는 GPU job을 실행하지
않는다.** 아래를 먼저 끝낸다.

| ID | 작업 |
| --- | --- |
| G-0-1 | billing admin이 Welcome Credit 잔액과 **원래 만료일**을 확인해 문서에 기록한다 |
| G-0-2 | billing account가 project에 연결되어 있고 활성인지 확인한다 |
| G-0-3 | `us-west1`의 실사용 가능 T4 quota를 확인한다 |

```bash
gcloud billing projects describe "$PROJECT_ID"
```

quota는 콘솔의 **IAM 및 관리자 → 할당량**에서 서비스 `aiplatform.googleapis.com`,
지표 `custom_model_training_nvidia_t4_gpus`를 확인한다. 신규 project는 0인 경우가 많고
증액 요청에 수 시간~수 일이 걸린다. **여기서 막히면 다른 모든 GPU 작업이 막히므로
가장 먼저 확인한다.** 승인되면 한도를 **1**로 제한한다.

### G-1. API 활성화

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  secretmanager.googleapis.com
```

FastAPI를 Cloud Run에 올리기로 하면 `run.googleapis.com`을 추가한다.
`container.googleapis.com`(GKE)은 **활성화하지 않는다.**

### G-2. GCS bucket

**확정(2026-08-15): `llm-router-mlserver`.**

```bash
export BUCKET=llm-router-mlserver

gcloud storage buckets create "gs://$BUCKET" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  --public-access-prevention
```

객체 버전 관리는 model과 dataset 경로를 실수로 덮어썼을 때 복구 수단이 된다. 다만
pipeline root에 쌓이는 중간 artifact까지 버전이 남으면 용량이 늘어나므로 lifecycle과
함께 쓴다.

```bash
gcloud storage buckets update "gs://$BUCKET" --versioning
```

```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30, "matchesPrefix": ["pipeline-root/"]}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"daysSinceNoncurrentTime": 30}
    }
  ]
}
```

```bash
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file=lifecycle.json
```

경로 규약은 총괄 체크리스트 10장 초안에 승인 모델 포인터를 더한 형태다.

```text
gs://<bucket>/datasets/routerbench/<version>/
gs://<bucket>/datasets/routerarena/<version>/
gs://<bucket>/datasets/continual-feedback/<candidate-version>/ # FastAPI가 쓰기, ML이 읽기
gs://<bucket>/datasets/retraining/<dataset-version>/
gs://<bucket>/config/<config-version>/
gs://<bucket>/pipelines/<ir-version>/              # 컴파일된 IR YAML
gs://<bucket>/models/router-modernbert/CURRENT.json
gs://<bucket>/models/router-modernbert/<model-version>/candidate/
gs://<bucket>/models/router-modernbert/<model-version>/approved/
gs://<bucket>/evaluations/<model-version>/<evaluation-version>/
gs://<bucket>/pipeline-root/
```

`config/`와 `datasets/`는 **내용을 덮어쓰지 않는 immutable version 경로**여야 KFP
캐시와 Vertex ML Metadata 계보가 신뢰할 수 있다. 이건 성능 최적화가 아니라 정확성
요건이다.

### G-3. Artifact Registry와 image

```bash
export REPO=llm-router

gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="LLM Router ML images"

gcloud artifacts repositories update "$REPO" \
  --location="$REGION" \
  --immutable-tags

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

Apple Silicon에서 빌드하면 반드시 `--platform linux/amd64`를 지정한다. Vertex는
amd64만 실행한다.

```bash
export IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

docker buildx build --platform linux/amd64 \
  -f ml/docker/data-processing.Dockerfile \
  -t "${IMAGE_BASE}/llm-router-data-processing:py312-v2" --push .
```

**build context는 반드시 저장소 루트(`.`)다.** 모든 Dockerfile의 `COPY` 경로가
`ml/src/...`, `ml/config/...`처럼 저장소 루트 기준으로 적혀 있다. context를 `ml`로
주면(과거 흔한 실수) `ml/ml/src/...`를 찾다가
`failed to calculate checksum ... not found`로 실패한다. 저장소 루트에서
명령을 실행한다.

`modernbert-training.Dockerfile`, `modernbert-evaluation.Dockerfile`(M-10의 CPU
교체본), `model-promotion.Dockerfile`도 같은 방식으로(context `.`) push한 뒤
digest를 확보한다. 이미지를 나중에 다시 빌드해야 할 때의 절차는
`ml/README.md`의 "Docker image 재빌드·재배포" 절을 참고한다.

```bash
gcloud artifacts docker images list "${IMAGE_BASE}" --include-tags --format=json
```

**tag가 아니라 digest를 IR에 박는다.** mutable tag를 쓰면 같은 IR이 서로 다른 코드로
실행될 수 있고 재현성이 깨진다.

### G-4. 서비스 계정과 IAM

역할을 3개로 나눈다. JSON key는 **어떤 경우에도 만들지 않는다.**

```bash
gcloud iam service-accounts create vertex-pipeline-sa \
  --display-name="Vertex Pipelines runner"
gcloud iam service-accounts create vertex-training-sa \
  --display-name="Vertex CustomJob training"
gcloud iam service-accounts create llm-router-serving-sa \
  --display-name="FastAPI serving runtime"

export PIPE_SA="vertex-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export TRAIN_SA="vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export SERVE_SA="llm-router-serving-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```

| SA | 필요한 권한 | 이유 |
| --- | --- | --- |
| `vertex-pipeline-sa` | `roles/aiplatform.user` (project) | PipelineJob 실행, CustomJob 제출 |
| | `roles/storage.objectAdmin` (bucket) | pipeline root 읽기·쓰기, artifact 저장 |
| | `roles/artifactregistry.reader` (repo) | component image pull |
| | `roles/iam.serviceAccountUser` (on `TRAIN_SA`) | CustomJob에 학습 SA를 지정 |
| | `roles/logging.logWriter` | 로그 |
| `vertex-training-sa` | `roles/storage.objectAdmin` (bucket) | 학습 입력 읽기, checkpoint 쓰기 |
| | `roles/artifactregistry.reader` (repo) | 학습 image pull |
| | `roles/logging.logWriter` | 로그 |
| `llm-router-serving-sa` | `roles/aiplatform.user` (project) | PipelineJob 제출·조회 |
| | `roles/storage.objectViewer` (bucket) | config와 approved package 다운로드 |
| | `roles/storage.objectAdmin` (`datasets/continual-feedback/` 조건부) | inline event의 Parquet/manifest 생성·조회·정리 |
| | `roles/artifactregistry.reader` (repo) | **serving image pull. 없으면 VM이 기동하지 못한다** |
| | `roles/logging.logWriter` (project) | 컨테이너 로그 |
| | `roles/secretmanager.secretAccessor` (secret) | 내부 API 토큰 조회 |
| | `roles/iam.serviceAccountUser` (on `PIPE_SA`) | PipelineJob에 pipeline SA를 지정 |
| Spring 백엔드 SA | 재학습용 GCS 역할 없음 | inline event JSON으로 FastAPI 호출 |

```bash
for ROLE in roles/aiplatform.user roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${PIPE_SA}" --role="$ROLE" --condition=None
done

gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:${PIPE_SA}" --role="roles/storage.objectAdmin"

gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" \
  --member="serviceAccount:${PIPE_SA}" --role="roles/artifactregistry.reader"

gcloud iam service-accounts add-iam-policy-binding "$TRAIN_SA" \
  --member="serviceAccount:${PIPE_SA}" --role="roles/iam.serviceAccountUser"
```

`SERVE_SA`와 Spring SA도 표대로 동일하게 부여한다. Spring SA의 접두사 제한은 IAM
조건식으로 건다.

```text
resource.name.startsWith("projects/_/buckets/<bucket>/objects/datasets/feedback/")
```

### G-5. 비용 guardrail

**budget alert는 지출을 막지 않는다.** 알림일 뿐이라는 사실을 runbook에 명시한다.

| ID | 작업 |
| --- | --- |
| G-5-1 | Billing → 예산 및 알림에서 credit 잔액보다 보수적인 예산과 50/80/100% 알림을 만든다 |
| G-5-2 | 알림 수신 이메일을 팀 전원으로 설정한다 |
| G-5-3 | T4 quota를 1로 제한한다 (G-0-3) |
| G-5-4 | Vertex Spend Cap(Preview) 사용 가능 여부를 확인하고 가능하면 최소 한도를 건다 |
| G-5-5 | pipeline task retry 0, CustomJob `restart_job_on_worker_restart=false`, timeout 21600s가 IR에 실제로 반영됐는지 확인한다 |
| G-5-6 | smoke 종료 후 임시 artifact와 실패 job을 정리하고 실제 청구액을 확인한다 |

대략적인 비용 감각(us-west1, 2026-08 기준 추정이며 반드시 가격 계산기로 재확인):

| 항목 | 개략 단가 | 6시간 |
| --- | --- | --- |
| `n1-standard-4` + T4 1장 CustomJob | 시간당 $0.5~0.6 수준 | 약 $3~4 |
| CPU pipeline task (4 vCPU 급) | 시간당 $0.2 미만 | 무시할 수준 |
| GCS Standard 저장 | GB·월당 $0.02 수준 | 무시할 수준 |
| FastAPI용 `e2-standard-2` 상시 | 시간당 $0.07 수준 | 월 $50 내외 |

**상시 켜두는 serving VM이 GPU 학습보다 총비용이 크다.** 데모 기간 외에는 중지한다.

### G-6. config·dataset 업로드

```bash
export CONFIG_VERSION=v2-20260815

gcloud storage cp ml/config/routerbench_labeling_v2.json \
  ml/config/modernbert_t4_balanced_v2.json \
  ml/config/routerarena_evaluation.json \
  ml/config/feedback_retraining_preprocessing.json \
  ml/config/package_metadata.json \
  ml/config/deploy_gate_rules_smoke.json \
  ml/config/baseline_metrics_empty.json \
  "gs://$BUCKET/config/$CONFIG_VERSION/"
```

`package_metadata.json`, `deploy_gate_rules_smoke.json`, `baseline_metrics_empty.json`은
2026-08-15에 새로 만들었다. `package_metadata.json`의 `model_version`은 placeholder이며
`model_package_component`가 매 run마다 `candidate_model_version`으로 override한다(M-6).

`deploy_gate_rules_smoke.json`은 실제 baseline 성능을 모르는 상태에서 만든 **smoke
전용** gate다. `accuracy >= 0.0`, `under_routing_rate <= 1.0`처럼 항상 통과하는 값만
넣어 package 무결성과 `model_version` 일치만 검증하고, `minimum_delta`/`maximum_delta`는
비워 `baseline_metrics_empty.json`(`{}`)과 짝을 이룬다. 운영 gate로 쓰지 않는다. G-8
전체 학습 이후 실제 수치로 별도의 `deploy_gate_rules.json`(smoke 접미어 없음)을 만들어
확정한다.

**v2 계약 정합성 주의.** 위 4개 config는 모두 v2 기준이다. `routerbench_labeling_v2.json`은
`minimum-sufficient-tier-v2` + `routerbench-training-preprocess-v2-balanced`이고,
`modernbert_t4_balanced_v2.json`은 `train_sampling_strategy: balanced_effective_batch`,
`feedback_retraining_preprocessing.json`의 `base_dataset_contract`도 v2를 가리킨다. 역사적
initial IR과 active feedback-retraining IR의 기본값도 v2다.

다만 FastAPI는 `training_config_uri`, `base_training_dataset_uri` 등을 **pipeline parameter로
항상 전달**하므로 IR 기본값을 덮어쓴다. serving의 `.env`가 v1 config를 가리키면 `/escalate`
재학습이 아무 오류 없이 v1 계약으로 돌아간다. 두 저장소의 값을 항상 같은 version으로 함께
바꾼다.

| 항목 | v2 값 |
| --- | --- |
| 라벨링 config | `routerbench_labeling_v2.json` |
| 학습 config | `modernbert_t4_balanced_v2.json` |
| base training dataset | `...__minimum-sufficient-tier-v2__routerbench-training-preprocess-v2-balanced/training.parquet` |
| feedback 전처리 config | `feedback_retraining_preprocessing.json` (`base_dataset_contract`가 v2) |

업로드 후 SHA-256을 기록해 로컬 파일과 일치하는지 확인한다.

```bash
gcloud storage hash --hex ml/config/modernbert_t4_balanced_v2.json
```

### G-7. IR 컴파일과 cloud smoke

4개 image 모두 digest를 환경변수로 주입해 컴파일한다(M-12 완료로 4개 전부 가능).

```bash
LLM_ROUTER_VERTEX_DATA_PROCESSING_IMAGE="${IMAGE_BASE}/llm-router-data-processing@sha256:<digest>" \
LLM_ROUTER_VERTEX_TRAINING_IMAGE="${IMAGE_BASE}/llm-router-modernbert-training@sha256:<digest>" \
LLM_ROUTER_VERTEX_EVALUATION_IMAGE="${IMAGE_BASE}/llm-router-modernbert-evaluation@sha256:<digest>" \
LLM_ROUTER_VERTEX_PROMOTION_IMAGE="${IMAGE_BASE}/llm-router-model-promotion@sha256:<digest>" \
  .venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-cloud

gcloud storage cp ml/artifacts/kubeflow/training-workflows-cloud/*.yaml \
  "gs://$BUCKET/pipelines/$CONFIG_VERSION/"
```

컴파일 IR에 GPU Kubernetes resource가 없고 CustomJob worker pool에만 T4가 있는지
확인한다.

```bash
grep -c "nvidia.com/gpu" ml/artifacts/kubeflow/training-workflows-cloud/feedback-retraining-pipeline.yaml
grep -c "NVIDIA_TESLA_T4" ml/artifacts/kubeflow/training-workflows-cloud/feedback-retraining-pipeline.yaml
```

앞은 0, 뒤는 1 이상이어야 한다.

feedback 100행 재학습 smoke 제출:

```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)
job = aiplatform.PipelineJob(
    display_name="smoke-feedback-retraining-100",
    template_path=f"gs://{BUCKET}/pipelines/{CONFIG_VERSION}/feedback-retraining-pipeline.yaml",
    pipeline_root=f"gs://{BUCKET}/pipeline-root/",
    parameter_values={...},  # 100행 feedback fixture + approved base dataset/checkpoint
    enable_caching=False,
)
job.submit(service_account=PIPE_SA)
print(job.resource_name)
```

smoke에서 확인할 것:

| ID | 확인 항목 |
| --- | --- |
| G-7-1 | Vertex Pipelines UI에 DAG가 표시되고 CPU task가 image를 pull한다 |
| G-7-2 | CustomJob이 T4 1장·replica 1로 뜨고 `gcp_resources` 출력이 남는다 |
| G-7-3 | checkpoint가 GCS에 기록되고 다음 task가 그 경로를 읽는다 (M-1, M-2 실측) |
| G-7-4 | 평가·package·gate가 CPU에서 끝난다 |
| G-7-5 | Vertex ML Metadata에 lineage가 연결된다 |
| G-7-6 | 같은 입력 재실행이 중복 CustomJob을 만들지 않는다 |
| G-7-7 | 실패 job이 자원을 붙잡고 있지 않다 |
| G-7-8 | 실제 청구액을 Billing에서 확인한다 |

**smoke 비용을 확인하고 남은 credit 대비 feedback 재학습 비용 한도를 계산한다.** 초기
baseline 전체 학습은 이 cloud smoke와 무관하게 Colab에서 수행한다.

### G-8. Colab 초기학습·baseline 확정

| ID | 작업 |
| --- | --- |
| G-8-1 | 승인된 전체 데이터로 Colab에서 ModernBERT baseline을 학습하고 Drive checkpoint·checksum·`_SUCCESS`를 동결한다 |
| G-8-2 | RouterBench holdout과 RouterArena full/robustness를 평가한다 |
| G-8-3 | 실제 수치를 보고 `deploy_gate_rules.json`을 versioned로 확정한다 |
| G-8-4 | Colab checkpoint를 package-v1 importer/qualification으로 검증하고 baseline metrics를 `evaluations/<version>/`에 고정한다 |
| G-8-5 | baseline package를 `approved/`에 올리고 `CURRENT.json`을 최초 생성한다 (M-8 필요) |

### G-9. FastAPI serving 배포

상태: **완료 (2026-08-16)**. GCE VM + 컨테이너로 배포했고 Spring VM에서 실제 호출을
확인했다.

| 항목 | 값 |
| --- | --- |
| VM / 존 | `llm-router-serving` / `us-west1-a` |
| 내부 IP | `10.138.0.2:8000` |
| 공인 IP | `34.182.108.1:8000` (고정 IP `llm-router-serving-ip`) |
| 이미지 | `us-west1-docker.pkg.dev/<project>/llm-router/llm-router-serving@sha256:bc600ee2…62b838` |
| 서빙 모델 | `router-modernbert-base-routerbench-20260814-45dcc6d` |
| 아웃바운드 | Cloud NAT (`llm-router-nat-router` / `llm-router-nat`) |

백엔드 팀 전달 문서는 `LLM_Router-serving/docs/backend-integration-handoff.md`다.

#### 실제 배포에서 걸린 것 — VM을 다시 만들 때 반드시 반영

아래 세 가지는 전부 로컬에서 드러나지 않고 GCP에서만 나타난다. 처음 배포에서 순서대로
전부 밟았다.

**1. `SERVE_SA`에 `roles/artifactregistry.reader`가 필요하다.** 아래 IAM 표에서 이
역할이 `llm-router-serving-sa`에 누락되어 있었다. 없으면 VM이 자기 이미지를 pull하지
못하고 konlet이 `denied: artifactregistry.repositories.downloadArtifacts`로 실패한다.
`roles/logging.logWriter`(컨테이너 로그)와 `PIPE_SA`에 대한
`roles/iam.serviceAccountUser`(`/escalate` 제출용)도 함께 부여한다.

**2. 부팅 디스크를 키운다 — `--boot-disk-size=50GB`.** 기본 10GB에서는 Docker가 쓸 수
있는 공간이 5.7GB뿐이라 이미지가 들어가지 않는다. konlet은 이때 `No such image`라는
오해하기 쉬운 메시지를 남기므로, 진짜 원인(`no space left on device`)은
`journalctl -u docker`에서 확인해야 한다. 이미지가 큰 이유는 `torch`가 리눅스 x86에서
CUDA 휠을 기본으로 받기 때문이며(GPU 없는 VM에는 불필요), CPU 전용 휠로 바꾸면
2~2.5GB 줄어든다.

**3. 마운트한 영속 디스크를 컨테이너 UID로 chown한다.** 컨테이너는 UID 10001로
실행되는데 새로 포맷된 디스크는 root 소유로 올라온다. Dockerfile의 `chown`은 이미지
안의 `/app/var`에 적용된 것이라 마운트에 덮인다. 결과는
`PermissionError: '/app/var/models'`이고 `/health`가 `routerModel: error`로 고정된다.
디스크에 한 번만 적용하면 재부팅해도 유지된다.

```bash
sudo chown -R 10001:10001 /mnt/disks/gce-containers-mounts/gce-persistent-disks/llm-router-serving-var
```

**4. 재학습 환경변수 12개를 빠뜨리면 `/escalate`만 조용히 죽는다.** `VERTEX_*`, 각
데이터셋·설정 `*_URI`, `ALLOWED_GCS_BUCKETS`가 없으면 `EscalationService`가 생성되지
않아 `/escalate`가 `503 PIPELINE_BACKEND_UNAVAILABLE`을 반환한다. 그런데 `/health`는
두 모델만 보므로 계속 `ok`다. 즉 배포는 성공한 것처럼 보이고 재학습 요청 시점에야
드러난다. 배포 후 `/escalate`에 존재하지 않는 batch를 한 번 던져 `422`가 오는지
확인한다. `503`이면 설정이 빠진 것이다.

**5. 임시 외부 IP는 VM 재시작 때 바뀐다.** model hot swap 자체는 VM을 재시작하지
않지만, 외부 노출을 쓴다면 일반 배포·장애 복구를 위해 static IP를 예약해 붙인다.

### GPU 학습 실패의 원인과 해결 (2026-08-17)

Vertex GPU 학습은 오랫동안 한 번도 성공하지 못했다. 원인은 **학습 이미지에 C 컴파일러가
없었던 것**이다.

ModernBERT는 CUDA에서 `torch.compile`을 켜고, TorchInductor는 GPU 커널을 **런타임에 C로
생성해 컴파일**한다. `pytorch/pytorch` 베이스 이미지에는 toolchain이 없어 첫 forward에서
아래로 죽는다.

```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised
```

**이 실패는 로컬에서 재현할 수 없다.** MPS와 CPU eager 경로는 `torch.compile`을 건너뛰기
때문에 실제 CUDA CustomJob에서만 드러난다. 진단이 오래 걸린 이유이기도 하다.

진단 과정에서 배제된 것들: T4 쿼터(정상), 학습·파이프라인 SA의 IAM(정상), 런타임 계약
파일(이미지에 포함됨), 학습 코드(정상). `TORCHDYNAMO_DISABLE=1`만 추가한 동일 job이
성공하면서 확정됐다.

해결은 `ml/docker/modernbert-training.Dockerfile`에 `build-essential`을 설치하고, 빌드
검증 체인에 `cc --version` / `c++ --version`을 추가한 것이다. 컴파일러가 빠지면 이제
**GPU job이 아니라 Docker 빌드가 실패**한다.

| 항목 | 값 |
| --- | --- |
| 새 학습 이미지 | `llm-router-modernbert-training:cu124-v2` |
| digest | `sha256:a3d6d67f9e3ca00017eb182e8775fade51cd1064248449e047403da5c631fe31` |
| 롤백용 | `cu124-v1` 보존 |
| 검증 | `TORCHDYNAMO_DISABLE` 없이 스모크 `JOB_STATE_SUCCEEDED` |
| 새 IR prefix | `gs://llm-router-mlserver/pipelines/v2-20260817/` |

**이미지를 새로 푸시하면 IR을 반드시 재컴파일·재업로드한다.** IR은 이미지를 digest로
고정하므로, 이미지만 바꾸고 IR을 그대로 두면 파이프라인은 계속 옛 이미지를 쓴다. 기존
prefix를 덮어쓰지 말고 새 버전 prefix에 올린 뒤 서빙의
`LLM_ROUTER_VERTEX_PIPELINE_TEMPLATE_URI`를 갱신한다.

<!-- zsh 주의: 태그·URI를 조립할 때 `$VAR:tag`는 zsh의 `:c` 등 수식자로 해석되어 문자가
     사라진다. 반드시 `${VAR}:tag`로 감쌀 것. 실제로 이미지가 잘못된 이름으로 푸시된
     사고가 있었다. -->


#### 리전 간 지연 (실측)

Spring(서울)과 Router(오리건) 사이 RTT는 약 120ms다. 연결을 재사용하면 요청당 120ms,
매번 새로 맺으면 240ms다. Spring은 connection pool을 반드시 켜야 한다.

---

이하 최초 계획 내용이다. GKE를 쓰지 않기로 했으므로 선택지는 둘이었다.

| 방식 | 적합성 | 비고 |
| --- | --- | --- |
| **GCE VM + 컨테이너 (권고)** | MVP에 적합 | SQLite lease와 `var/` 영속 디스크가 그대로 동작한다. 단일 인스턴스가 자연스럽게 보장된다. `e2-standard-2`(2 vCPU/8 GB)면 BGE-large + ModernBERT-base 적재에 충분하다 |
| Cloud Run | 조건부 | min-instances=1, max-instances=1, 4 GiB 이상, CPU always-on 필요. **SQLite를 GCS FUSE에 두면 파일 잠금이 신뢰할 수 없다.** S-6를 지키려면 `RetrainingJobStore`를 Firestore나 Cloud SQL로 교체해야 한다 |

권고안 기준 작업:

```bash
gcloud compute instances create-with-container llm-router-serving \
  --zone="${REGION}-a" \
  --machine-type=e2-standard-2 \
  --service-account="$SERVE_SA" \
  --scopes=cloud-platform \
  --container-image="${IMAGE_BASE}/llm-router-serving@sha256:<digest>" \
  --container-mount-disk=mount-path=/app/var,name=llm-router-serving-var \
  --create-disk=name=llm-router-serving-var,size=20GB,type=pd-balanced,auto-delete=no \
  --no-address
```

| ID | 작업 |
| --- | --- |
| G-9-1 | serving image를 Artifact Registry에 push한다 (`LLM_Router-serving/Dockerfile`) |
| G-9-2 | VM에 `SERVE_SA`를 부착한다. **JSON key를 image나 저장소에 넣지 않는다** |
| G-9-3 | `var/`를 영속 디스크에 마운트한다. 없으면 컨테이너 교체 시 멱등성 기록이 사라진다 |
| G-9-4 | `LLM_ROUTER_INTERNAL_API_TOKEN`과 관리자 토큰을 Secret Manager에 두고 주입한다 |
| G-9-5 | 외부 IP를 붙이지 않고 VPC 내부에서만 Spring이 접근하게 한다. 방화벽은 Spring 인스턴스 태그에서만 8000 포트를 허용한다 |
| G-9-6 | `GET /health`를 readiness로 쓴다. 두 모델 적재 전에는 503이다 |
| G-9-7 | 데모 기간 외에는 VM을 중지해 비용을 줄인다 |

### G-10. 통합 검증

| ID | 작업 |
| --- | --- |
| G-10-1 | Spring stub과 실제 predictor 사이에서 Router API contract test를 통과한다 |
| G-10-2 | Spring이 실제 inline feedback을 전송하고 FastAPI가 `datasets/continual-feedback/`에 artifact를 생성한다 |
| G-10-3 | `POST /escalate` → Vertex PipelineJob 제출 → 완료까지 실행한다 |
| G-10-4 | 같은 `batchId` 재요청이 중복 학습을 만들지 않는지 확인한다 (C-4) |
| G-10-5 | 승격 → `POST /admin/models/reload` → `/health` → 새 version 응답을 확인한다 |
| G-10-6 | rollback을 실제로 한 번 실행해 본다 |

## 9. 실행 순서

앞 단계가 끝나야 뒤 단계가 의미 있는 의존 관계다.

```text
Colab G-8 초기학습·gate rules 확정 ───────────────┐
G-0 credit·quota → G-1 API → G-2/G-3/G-4/G-5 ─┤
                                                └─ M-16 checkpoint import
                                                     → M-6~M-9 승격 → G-9 serving
                                                     → G-6 feedback config·base artifact
                                                     → G-7 feedback-retraining IR·100행 smoke
                                                     → C-1~C-7, S-1~S-6, D-1~D-9
                                                     → G-10 실제 feedback 통합 검증 (T-9)
```

단, **C-1 ~ C-7(Vertex 전환)과 D-2 ~ D-9(모델 교체)는 GCP 자원 없이도 fake client로
먼저 구현·테스트할 수 있다.** G-0의 quota 승인을 기다리는 동안 이 두 묶음을 진행하는
것이 일정상 가장 효율적이다.

## 10. 결정이 필요한 항목

아래는 코드로 정할 수 없고 팀 또는 사용자가 정해야 한다.

| 항목 | 선택지 | 문서 반영 위치 |
| --- | --- | --- |
| GCS bucket·Artifact Registry repo·SA 이름 | 자유 | `backend-infrastructure-agreement.md` |
| 자동 gate 임계값 | 실측 성능을 바탕으로 versioned rules 확정 | `workstream-checklist.md` H |
| FastAPI 배포 대상 | GCE VM(권고) / Cloud Run | 동 문서 5장 |
| Cloud Run 선택 시 job store | Firestore / Cloud SQL | 동 문서 |
| `deploy_gate_rules.json` 임계값 | G-8-2 실측 후 확정 | `workstream-checklist.md` H |
| 재학습 상태 통지 방식 | Spring이 `GET /escalate/{batchId}` 폴링 / FastAPI가 webhook 호출 | `api-contract.md` |

사람의 승인 UI는 만들지 않는다. 운영 화면이 필요해지면 승인 버튼이 아니라 candidate
metric, gate rule별 결과, 자동 전환 이력과 rollback 상태를 보여주는 관측 화면으로 둔다.

## 11. 관련 문서

- `docs/ml/workstream-checklist.md`: ML 진행 상태와 유일한 체크박스
- `docs/ml/backend-infrastructure-agreement.md`: 확정 구조 통보와 팀 합의 계약
- `docs/ml/modernbert-training-spec.md`: 학습·runtime·checkpoint 계약
- `ml/README.md`: 로컬 CLI와 Docker 실행 명령
- `LLM_Router-serving/docs/api-contract.md`: FastAPI 내부 API 계약
- `LLM_Router-serving/docs/serving-and-retraining-architecture.md`: serving 책임 경계
