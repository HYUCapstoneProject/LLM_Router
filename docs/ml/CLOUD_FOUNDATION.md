# R0·R1 공통 클라우드 기반

R0/R1 구현을 분기하기 전에 공통 코드와 GCS/GCP 기반부터 준비한다. 실행 플랫폼은 **관리형 Vertex AI Pipelines CPU component + Vertex AI Custom Training T4**다. 기존 main의 공통 처리와 검증 코드를 먼저 완성하고, 동일한 공통 기준 커밋에서 Spring과 FastAPI의 R0/R1 브랜치를 만든다.

## 현재 자료와 감사 도구

[기반 설정](../../ml/config/cloud-foundation-v1.json)은 기존 serving `.env.example`과 ML `cloudbuild.retraining.yaml`에서 확인한 project `project-763753da-fa8f-4afd-b31`, region `us-west1`, bucket `llm-router-mlserver`, Artifact Registry `llm-router`, Pipeline/Training service account를 기록한다. VM은 `llm-router-serving`/`us-west1-a`다. 2026-09-26 실제 VM 조회로 `RUNNING`, `e2-standard-2`, Serving SA `llm-router-serving-sa@project-763753da-fa8f-4afd-b31.iam.gserviceaccount.com`과 `cloud-platform` scope를 확인하고 SA를 고정했다. 설정에 이름이 있다는 것과 자원이 실제로 올바르게 구성됐다는 것은 별개다.

R0/R1 경로의 공통 준비는 **`namespace_status=provisioned`**다. 실제 VM 계정의 feedback 경로 create/read/delete를 확인했고 각 profile에 SHA로 이름 붙인 namespace 설정 객체를 새로 게시·재읽기했다. [R0 증거](evidence/cloud-namespace-r0-20260926.json), [R1 증거](evidence/cloud-namespace-r1-20260926.json)에 generation과 SHA가 있다. 두 모델 그룹을 게시한 상태는 아니며 기존 CURRENT generation `1786882332374746`은 변하지 않았다. GCS 경로 선언 자체에 별도 디렉터리 생성은 필요하지 않다. 불변 릴리스 artifact를 실제 게시하기 전까지 pipeline·baseline·gate URI와 image digest는 `null`로 둔다. 검증되지 않은 값이나 기존 smoke 설정으로 빈칸을 채우지 않는다.

2026-09-26 00:47 KST의 [실제 클라우드 감사](evidence/cloud-foundation-20260926.json)는 11개 inventory 조회를 완료했고 필수 API·서비스 계정·VM 일치 검사 오류가 0개였다. bucket은 `US-WEST1`, uniform bucket access와 public access prevention이 켜져 있고, 1일 뒤 삭제하는 lifecycle은 `pipeline-root/`만 대상으로 했다. soft delete 보존은 604,800초(7일), Artifact Registry는 `DOCKER`와 immutable tag 설정이었다. 이 artifact는 **IAM 보완 전 스냅샷**이며 개인 IAM principal 2개를 가렸다. 기록된 config SHA는 당시 명세의 SHA다. 이후 namespace 상태 표현이나 설정을 바꾸면 새 감사로 다시 연결한다.

당시 Serving SA는 bucket `objectViewer`와 기존 `datasets/continual-feedback/`만 쓰는 조건부 `objectUser`를 갖고 있었다. 제안한 R0/R1 feedback 및 control prefix에 대한 쓰기 권한은 이 스냅샷에서 확인되지 않았다. 지정 SA 간 `actAs` binding과 Pipeline/Training SA의 bucket `objectAdmin`은 존재했다. 이를 최소 권한 구성 완료나 실제 쓰기 성공으로 확대하지 않는다. 기존 운영 권한을 보존하면서 필요한 새 prefix 권한을 보완한 후 새 inventory와 접근 검증 증거를 남긴다.

```bash
PYTHONPATH=ml/src python ml/scripts/audit-cloud-foundation.py \
  --config ml/config/cloud-foundation-v1.json \
  --output /path/to/new-local-audit.json

PYTHONPATH=ml/src python ml/scripts/audit-cloud-foundation.py \
  --config ml/config/cloud-foundation-v1.json --cloud \
  --output /path/to/new-cloud-audit.json
```

출력 파일은 새 이름으로 생성하며 기존 결과를 덮어쓰지 않는다. config SHA와 관측 시각을 함께 기록한다. 도구는 자원 생성·IAM 변경·secret 읽기·학습 제출·배포를 수행하지 않는다. `--cloud`는 활성 API 목록과 VM/bucket/Artifact Registry/SA 설명 및 IAM 정책만 조회한다. VM metadata 전체나 gcloud 오류 원문은 결과에 포함하지 않는다. 인증 실패나 일부 응답 누락도 오류로 남기며 종료 코드는 1이다.

| 상태 | 의미 |
|---|---|
| `planned` | 로컬 명세 정합성이 깨짐. 클라우드 조회도 실행하지 않음 |
| `local_verified` | namespace와 설정 구조 검사 통과. 클라우드 미조회 또는 일부 조회·일치 검사 실패 |
| `cloud_verified` | 필요한 읽기 전용 inventory 조회와 명시적 정합성 검사 통과 |

`cloud_verified`는 **inventory 확인 상태**다. 실효 IAM 권한이나 비용 한도, 모델 release, 배포 완료를 뜻하지 않는다. `iam_effective_access_verified=false`, `deployment_ready=false`와 남은 점검 목록을 별도로 표시한다. IAM 정책을 읽을 수 있다는 사실만으로 해당 실행 identity의 접근 성공을 주장하지 않는다.

namespace 준비가 실제 완료되면 `namespace_status=provisioned`와 `namespace_evidence: {"path": "상대경로.json", "sha256": "파일 SHA-256"}`를 함께 기록한다. 증거 파일은 `--evidence-root`(기본 현재 작업 디렉터리) 내부에 있어야 하며, SHA가 맞아야 한다. 파일 schema는 `router-namespace-provisioning-v1`, `profile`은 해당 `r0` 또는 `r1`, `namespace_sha256`는 도구의 `namespace_sha256(profile, service_accounts)`로 계산한 경로·identity 계약 해시다. `observed_at`과 실제 검증을 반영한 `iam_configuration_verified`, `prefix_access_probe_verified`, `production_pointer_unchanged`가 모두 필요하다. 경로나 SA가 바뀌면 이전 증거는 무효다. 읽기 전용 inventory는 prefix 접근 probe를 대신하지 않는다. 증거의 SHA는 파일 무결성을 연결하며 그 파일에 기록한 관측의 수행 사실을 자체 증명하지는 않는다.

## 공통 namespace

| 구분 | R0 | R1 |
|---|---|---|
| 설정·IR | `releases/r0/config/`, `releases/r0/pipelines/` | `releases/r1/config/`, `releases/r1/pipelines/` |
| feedback·control | `releases/r0/datasets/feedback/`, `releases/r0/retraining-control/` | `releases/r1/datasets/feedback/`, `releases/r1/retraining-control/` |
| 관측·학습 output | `releases/r0/retraining-runs/`, `releases/r0/vertex-training-output/` | `releases/r1/retraining-runs/`, `releases/r1/vertex-training-output/` |
| registry | `releases/r0/models/router-modernbert/` | `releases/r1/models/router-modernbert/` |
| 임시 pipeline artifact | `pipeline-root/r0/` | `pipeline-root/r1/` |
| SQLite | `/app/var/r0/retraining.sqlite3` | `/app/var/r1/retraining.sqlite3` |

GCS 경로는 모두 `gs://llm-router-mlserver/` 기준이다. 각 registry의 `CURRENT.json`을 사용하며 기존 운영 `models/router-modernbert/CURRENT.json`과 겹치지 않는다. 공통 immutable 입력 dataset은 필요에 따라 읽기 전용으로 공유하되 생성 output·버전 ledger·활성 포인터는 분리한다. 같은 SA를 사용한다면 이 경로 분리는 실행 데이터의 구분이며 별도 보안 identity로 격리됐다는 뜻이 아니다.

## 배포 전에 완료할 작업

| 순서 | 작업 | 완료 증거 |
|---|---|---|
| 1 | 실제 API/VM/SA/GCS/AR 구성 감사 | config SHA와 시각이 있는 inventory, 재인증/권한/부재 오류 해소 |
| 2 | 경로·IAM·보존 정책 확정 | 각 주체의 resource 범위와 접근 검증, 기존 운영 prefix 영향 없음 |
| 3 | 공통 Spring/FastAPI/ML 계약 완성 | 인증·request ID·feedback 멱등성·상태·package/CURRENT·rollback 테스트 |
| 4 | 재현 가능한 build 경로 완성 | 네 ML image 역할, digest 고정, KFP compile, IR/config/data SHA 연결 도구 |
| 5 | R0/R1 분기 후 전용 계약 구현 | objective별 feedback/target/mask/loss/policy/gate; 공통 코드 중복 변경 방지 |
| 6 | 각 모델 릴리스 준비 | 실제 digest로 새 IR 생성, 동결 gate와 baseline, package·policy 버전, 불변 artifact 업로드 및 SHA 확인 |

API/IAM/bucket/Artifact Registry 구성은 서버 배포 전 처리할 수 있으므로 후순위로 미루지 않는다. 모델별 최종 image digest·IR·gate artifact는 해당 구현이 완성된 뒤 확정한다. 공통 build/검증 경로를 먼저 완성한다는 원칙과 모델별 최종 산출물을 나중에 만든다는 순서를 구분한다.

### 권한과 보존

| 주체 | 필요한 경계 |
|---|---|
| Serving SA | 승인 model/config/IR/control/관측 읽기, 지정 feedback prefix 쓰기, Vertex Pipeline 제출·조회, 지정 Pipeline SA `actAs`, VM image pull |
| Pipeline SA | 지정 pipeline artifact·모델 registry·관측 접근, CustomJob 제출·조회, 지정 Training SA `actAs` |
| Training SA | dataset/config/anchor 읽기, 지정 checkpoint/output/관측 쓰기 |
| 관리형 Vertex image pull identity | 실제 서비스 관리 identity가 training container image를 읽을 수 있는지 확인 |
| Build identity | 선택한 Cloud Build 실행 identity의 source 읽기·해당 repository 게시 권한 |
| Spring | 내부 HTTP credential, 재학습 GCS 쓰기 권한 불필요 |

control ledger 쓰기는 Pipeline이 수행하므로 Serving SA에 새 control 쓰기 권한을 추가하지 않는다. 2026-09-26 공통 설정 작업에서 Serving SA의 `objectUser` 조건을 `releases/r0/datasets/feedback/` 또는 `releases/r1/datasets/feedback/`에 한정해 추가했고 기존 운영 prefix binding을 보존했다. 이후 실제 VM identity의 쓰기·재읽기·검증 객체 삭제와 기존 CURRENT 보존을 확인했다. 위의 namespace 증거와 [변경 후 재감사](evidence/cloud-foundation-after-20260926.json)로 연결한다. 위에 링크한 최초 스냅샷은 이 변경 이전 상태를 보존한다.

Custom Training 컨테이너의 custom SA는 Artifact Registry image pull identity와 다르므로 Training SA에 reader 역할만 붙여 완료로 표시하지 않는다. [Google custom service account 문서](https://docs.cloud.google.com/vertex-ai/docs/general/custom-service-account)

프로젝트 IAM과 bucket IAM을 함께 확인한다. bucket `get-iam-policy`는 프로젝트에서 상속되는 binding을 모두 보여주지 않는다. 현재 package 다운로드·retention 구현은 `list_blobs`를 사용하므로 bucket 목록 권한도 필요하다. 일반 object-prefix `resource.name` 조건만으로 `storage.objects.list`를 일부 객체에 제한할 수 없으며 객체 read/write 조건과 목록 권한을 구분해 검토한다. [Cloud Storage IAM](https://docs.cloud.google.com/storage/docs/access-control/iam)

current·previous·anchor package, release/config/평가 증거와 연구 결과는 임시 artifact 정책의 삭제 대상에서 제외한다. 기존 [lifecycle 예시](../../ml/config/gcs_pipeline_artifacts_lifecycle.example.json)는 `pipeline-root/`만 가리키지만 실제 bucket 전체 규칙을 읽어 겹치는 다른 삭제 규칙도 확인해야 한다. soft delete·versioning을 함께 고려한다. [Cloud Storage lifecycle](https://docs.cloud.google.com/storage/docs/lifecycle)

### 릴리스 연결

이미지 역할은 data-processing, training, evaluation, promotion 네 가지다. 각 image를 immutable `@sha256:…`로 고정하고 아래 환경 변수로 IR을 컴파일한다.

```text
LLM_ROUTER_VERTEX_DATA_PROCESSING_IMAGE
LLM_ROUTER_VERTEX_TRAINING_IMAGE
LLM_ROUTER_VERTEX_EVALUATION_IMAGE
LLM_ROUTER_VERTEX_PROMOTION_IMAGE
```

```bash
.venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir /path/to/new-release/compiled-ir
```

새 image 코드가 기존 IR에 자동 반영되지는 않는다. image digest 변경 시 IR도 다시 컴파일하고 새 경로·SHA로 게시한다. 설정 manifest는 baseline/gate/pipeline URI를 해당 R0/R1 경계에 두도록 검사한다. 이 검사는 실제 artifact 내용과 checksum 또는 승격 품질을 대신하지 않는다. 해당 모델의 release 준비기와 package/gate 검증이 이어져야 한다.

기존 serving 예제의 `baseline_metrics_empty.json`, `deploy_gate_rules_smoke.json`, 공유 운영 CURRENT는 새 R0/R1 릴리스에 계승하지 않는다. gate 임계값은 development 자료로 동결하고 candidate와 baseline은 같은 추론 길이와 같은 gate 계약으로 평가한다. 이전 512-token 학습 이력은 보존하고 새 8,192-token 추론 계약은 별도 package/policy로 연결한다.

## 서버에 올린 뒤 검증할 작업

1. 실제 실행 identity·secret 주입·image pull·내부 HTTP 경로와 영속 volume/UID 10001 쓰기 및 SQLite 백업 복구를 확인한다. 가능한 준비는 사전 완료하고 실제 실행 환경의 확인만 여기서 수행한다.
2. 대상 FastAPI VM에서 8,192-token 지연 분포·최대 RSS·동시 요청·timeout을 측정한다. 과거 Colab 프로세스 합벽시계를 요청 지연으로 사용하지 않는다.
3. 유효 batch가 `202 → running → terminal gate → CURRENT → 실제 serving 적재`로 연결되는지 확인한다. 동일 batch 재전송, gate 거부, model load 실패, 동시 pointer 갱신과 이전 승인 package rollback을 검증한다.
4. Spring 응답의 model/package/policy 버전이 실제 FastAPI 활성 상태와 일치하는지 확인하고 R0/R1 서비스 후보를 같은 품질·비용 조건에서 비교한다.

GPU 실행 전 현재 credit/budget·quota·실행 상한을 확인한다. `/health=200`, image build, 로컬 테스트 또는 inventory 감사 성공만으로 재학습 전체 경로 통과를 기록하지 않는다.

## 단일 활성 방식과 기존 API 유지

실제 서비스는 R1을 기본으로 하며 R0로 교체할 수 있다. HTTP endpoint·request·response 계약은 유지한다. 두 profile의 설정 파일 존재와 두 모델 그룹의 동시 GCS 보관을 구분한다. GCS에는 활성 방식의 모델만 두고, 교체되는 방식의 current/previous/anchor와 모든 복구 필수 파일은 `alexyoo0506@gmail.com` Drive에 보존한다. 원격 재읽기·파일별 SHA 검증 전에는 GCS 원본을 제거하지 않는다.

교체는 in-flight 재학습 종료·승격 중지·Drive 검증·조건부 GCS 정리·새 그룹 게시·실제 적재 확인 순서다. 다른 방식의 그룹을 동시에 올리는 rolling 전환은 사용하지 않는다. 현재/직전/anchor 보존은 활성 방식 내부에 적용한다. 모델이 포함된 과거 training archive도 inventory에 포함해 누락을 확인한다. 실제 그룹 교체와 모델 삭제는 이번 공통 경로 준비에서 수행하지 않았다.

전체 IAM 실효 접근 검증은 여전히 미완료다. Serving feedback prefix probe는 통과했고, Pipeline/Training의 실제 job·이미지 pull·출력 접근은 방법별 release 이후 검증한다.
