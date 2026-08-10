# ML-온라인·인프라 협의 명세

상태: **협의 전 제안서**

기준일: 2026-08-10

## 1. 목적

이 문서는 ML 담당자가 단독으로 확정할 수 없는 온라인 서비스 및 GCP/GKE 운영 계약을
회의에서 결정하기 위한 기준이다. 여기 적힌 기본값은 구현 가능한 제안일 뿐 운영 확정값이
아니다. 작업 완료 여부는 `docs/ml/workstream-checklist.md`에서만
관리한다.

회의는 다음 세 종류로 나눈다.

1. 문서 정합성 협의: MVP·졸업프로젝트 개요와 현행 ML 설계 중 어느 쪽을 기준으로 삼을지
2. 온라인·백엔드 협의: Spring, FastAPI serving, Claude 호출, DB와 feedback export
3. 인프라 협의: GCS, Artifact Registry, GKE/Kubeflow, IAM, 자원과 비용

## 2. 이미 결정되어 다시 논의하지 않아도 되는 기준

- ML 분류 라벨과 checkpoint class 순서는 `easy=0`, `medium=1`, `hard=2`다.
- ML은 난이도 라벨을 반환하고 실제 Claude model ID 선택은 온라인 routing 정책이 맡는다.
- RouterBench 초기 검증·라벨링·전처리는 로컬에서 수행하고 검증된 학습 artifact만 GCS에
  올린다.
- feedback 재학습의 검증·정규화·병합은 Kubeflow CPU Pod, 학습은 GPU Pod에서 수행한다.
- RouterArena 전처리는 CPU에서 수행한다. checkpoint 전체 추론 평가는 GPU를 기본으로
  하되 소규모 smoke는 CPU에서도 실행할 수 있다.
- QQP embedding과 cache threshold는 온라인 cache 팀 책임이며 ML/Kubeflow DAG에 넣지
  않는다.
- Kubeflow는 승인 후보를 만드는 곳까지 담당한다. 실제 서빙 모델 전환과 rollback은
  온라인 운영 계약이 필요하다.

이 기준들은 ML 구현의 현행 기준이다. 다만 기존 `mvp-scope-and-checklist.md`와
`졸업프로젝트 개요.md`의 표현을 프로젝트 전체가 이 기준으로 정정하는 것은 3장의
공식 협의를 거쳐 확정한다.

## 3. MVP·졸업프로젝트 개요 정합성 협의

### 3-1. 협의 원칙

- `mvp-scope-and-checklist.md`는 2026-07 시점의 역사적 MVP 범위라는 표시를 유지한다.
- `졸업프로젝트 개요.md`는 최종 목표와 현행 설계가 함께 보이므로, 현재 구현과 다른
  내용을 단순 과거 기록으로 방치하지 않고 `확정`, `계획`, `미구현`으로 구분한다.
- 아래 충돌은 ML 코드의 임의 변경 사유가 아니다. 팀이 기준을 정한 뒤 개요·MVP,
  contract fixture, config와 코드가 같은 결론을 가리키도록 갱신한다.
- Claude 제품명·가격, GCP 비용과 일정처럼 시간에 따라 바뀌는 내용은 기준일과 출처를
  표시하고 인프라 착수 시 다시 확인한다.

### 3-2. 반드시 결론을 내려야 하는 충돌

| 충돌 주제 | MVP·졸업프로젝트 개요의 표현 | 현행 ML 설계·구현 | 회의에서 결정할 사항 | ML 측 제안 |
| --- | --- | --- | --- | --- |
| RouterBench 라벨링 | 졸업프로젝트 개요는 전체 solve rate와 약/중/강 대표 모델의 결과를 주 라벨 신호로 설명 | 11개 모델을 empirical tier로 나누고 tier consensus로 최소 충분 capability를 판단하며 solve rate는 진단값 | 개요의 라벨 알고리즘을 현행 방식으로 교체할지 | 11개 모델 tier-consensus를 공식 기준으로 채택 |
| 초기 데이터 처리 위치 | 개요와 일부 MVP 표현은 Kubeflow CPU component가 초기 라벨링부터 수행하는 흐름으로 읽힘 | 초기 RouterBench 검증·라벨링·전처리는 로컬, 검증된 artifact 업로드 이후 Kubeflow 학습 시작 | 초기 학습 pipeline의 시작점을 원본 데이터와 `training.parquet` 중 어디로 둘지 | 비용과 재현성을 위해 검증된 `training.parquet`에서 시작 |
| RouterArena 평가 자원 | 개요는 RouterArena 평가까지 CPU component/image로 묶어 표현 | 전처리는 CPU, ModernBERT full checkpoint 추론·metric은 별도 GPU 평가 image | CPU-only 평가로 되돌릴지, CPU smoke/GPU full로 분리할지 | CPU smoke와 GPU full 평가 분리 |
| Claude mapping | MVP와 개요는 `easy→Haiku`, `medium→Sonnet`, `hard→Opus`를 확정 mapping처럼 표현 | ML은 3-class까지만 고정하고 실제 model ID는 versioned 온라인 정책으로 관리 | 현재 사용 가능한 실제 model ID, fallback과 변경 주체 | 계열명은 설명용 예시로 남기고 실제 ID는 코드 밖 설정 |
| feedback 라벨 의미 | 최종 만족 tier를 해당 질문의 실제 필요 수준이자 정정 라벨로 곧바로 간주 | 단발 만족 신호는 노이즈가 있으므로 schema·PII·중복·충돌·반복 근거를 검증한 뒤 채택 | 어떤 조건에서 final tier를 학습 정답으로 인정할지 | upstream 보증과 ML 품질 필터를 함께 적용 |
| escalation 재검사 | 개요는 escalation 기록 자체가 ML 정정 라벨 공급원이라고 설명 | 온라인 export가 유효 feedback만 보증한다면 ML이 escalation 비즈니스 규칙을 중복 검사할 필요가 없음 | `escalation_count`를 필수 채택 조건으로 유지할지 감사 metadata로 완화할지 | upstream 계약 확정 시 감사 metadata로 완화 |
| 재학습 trigger | MVP·개요는 정정 라벨 N건 도달 시 Kubeflow 자동 실행으로 표현 | N, class별 최소량, 최대 대기 기간과 실행 주체가 미정이며 초기에는 수동 승인 가능 | N의 정의와 scheduler·수동 승인 책임 | dedup 후 accepted count와 분포·비용을 기준으로 합의 |
| Kubeflow 시작 범위 | 개요는 `데이터→라벨링→학습→평가→배포` 전체 자동화로 표현 | 초기 라벨링은 로컬이며 Kubeflow는 초기 학습 또는 feedback prepare부터 시작 | 개요의 Kubeflow 설명을 초기/재학습 두 흐름으로 나눌지 | 두 pipeline을 구분해 기술 |
| Kubeflow 배포 범위 | MVP·개요는 gate 통과 모델이 바로 재배포되는 것처럼 표현 | 구현은 평가 evidence, package와 gate까지이며 approved 승격·FastAPI 전환·rollback은 미합의 | gate 통과, 승인, serving 전환 각각의 주체 | Kubeflow는 candidate 생성, 온라인·운영은 승인과 전환 |
| API label 계층 | MVP stub 설명은 Spring `LIGHT/MEDIUM/HEAVY`, ML 문서는 `easy/medium/hard` 사용 | checkpoint와 ML API 경계는 `easy/medium/hard`로 고정 | Spring이 같은 이름을 쓸지 내부 adapter를 둘지 | 경계는 `easy/medium/hard`, 필요하면 Spring 내부에서만 변환 |
| Router API schema | MVP 공개 `/api/chat`은 `message`, `conversationId` 중심이고 ML 내부 제안은 `question`, `request_id` 중심 | 서로 다른 계층이지만 명시적 변환 계약이 없음 | 공개 요청에서 내부 Router 요청으로 누가 어떻게 변환할지 | Spring adapter와 양쪽 fixture를 확정 |
| QQP 책임 | MVP·개요는 QQP threshold 평가를 전체 완료 기준으로 포함 | ML/Kubeflow에서는 제거하고 온라인 cache 팀 결과만 최종 보고에 결합 | 전체 목표는 유지하되 담당·산출물을 어디에 둘지 | 온라인 cache 팀 소유로 명시 |
| RAG 의미 | MVP 라벨 명세에는 RouterBench RAG eval-family 포함 여부 실험이 있고 서비스 문맥에서는 RAG 구현 질문도 있었음 | RouterBench 데이터 sensitivity는 남지만 retrieval-augmented generation 서비스는 구현하지 않음 | 두 의미를 문서에서 어떻게 구분할지 | `RouterBench RAG eval-family`와 `service RAG`를 명시적으로 분리 |
| cloud 준비 상태 | 개요는 GCS/GKE/Kubeflow와 비용·일정을 구축 계획 또는 사용 가능 상태처럼 설명 | 확인 당시 project 접근만 가능하고 bucket, registry, GKE/Kubeflow는 미생성 | 계획과 실제 provision 상태를 어떻게 표시할지 | 현재 상태 표와 갱신일을 개요에 추가 |
| 비용·일정·모델 버전 | 개요의 Claude 가격, 모델명, GKE 비용과 6~11월 일정이 고정값처럼 남아 있음 | 변동 가능하며 현재 구현 진척 및 cloud 상태와 일부 다름 | 역사적 가정으로 보존할지 최신 기준으로 다시 산정할지 | 가정·기준일을 표시하고 배포 전 재산정 |

### 3-3. 충돌이 아니라 역할·시점 차이로 유지할 항목

| 항목 | 정합성 판단 | 문서 처리 방식 |
| --- | --- | --- |
| MVP가 Kubeflow·자동 재학습을 제외 | 최소 end-to-end MVP와 최종 목표의 시점 차이 | 역사적 MVP 범위로 유지 |
| RouterArena의 공식 3-class와 RouterBench 3-class | 이름은 같지만 생성 기준이 다른 외부 정렬 평가 | 정확도보다 예측 그룹 분포가 주 분석임을 유지 |
| QQP가 최종 프로젝트 평가에 포함 | 전체 프로젝트 목표와 ML 담당 범위의 차이 | 온라인 팀 산출물을 ML 결과와 분리해 결합 |
| `/api/chat`과 내부 Router API의 필드 차이 | 공개 API와 내부 서비스 API라는 계층 차이 | 충돌로 제거하지 않고 adapter 계약 추가 |

### 3-4. 회의 결과로 문서에 반영할 내용

- 각 충돌 행에 `채택 기준`, `결정자`, `결정일`, `반영 PR`을 기록한다.
- `졸업프로젝트 개요.md`에는 현행 구조와 아직 구현되지 않은 최종 목표를 구분한다.
- `mvp-scope-and-checklist.md`는 역사적 기준을 유지하되 확정 정책처럼 보이는 Claude
  mapping과 자동 재학습 표현에는 현행 협의 문서 링크를 추가한다.
- ML 세부 명세와 config는 합의가 알고리즘·schema·runtime을 바꾸는 경우에만 수정한다.
- 정합성 수정 후 같은 용어가 문서마다 다른 의미로 쓰이는지 다시 검색한다.

## 4. 온라인·백엔드 회의에서 결정할 사항

### 4-1. Router API와 실제 모델 선택

| 주제 | 결정할 질문 | ML 측 제안 | 결정 결과가 반영될 곳 |
| --- | --- | --- | --- |
| 외부 label 표현 | API 경계에서 `easy/medium/hard`를 그대로 쓸 것인가 | ML 경계에서는 그대로 유지하고 Spring 내부 enum 변환은 adapter에서 수행 | OpenAPI/fixture, Spring adapter |
| 요청 schema | 공개 `/api/chat` 요청을 내부 Router 요청으로 누가 변환하는가 | Spring이 `message→question`, `conversationId→request_id/추적 정보`로 변환 | Spring→FastAPI contract test |
| 응답 schema | label 외 confidence, model version, latency를 필수로 할 것인가 | 네 필드를 모두 반환 | FastAPI response fixture |
| Claude mapping | 각 label이 가리킬 실제 Claude model ID는 무엇인가 | Haiku/Sonnet/Opus 계열은 예시이며 model ID는 설정으로 관리 | 온라인 versioned routing config |
| 장애 정책 | timeout, 연결 실패, unknown label, model load 실패 시 어디로 보낼 것인가 | 품질 보호용 상위 tier fallback과 명시적 오류 metric | Spring/FastAPI runbook과 test |
| idempotency | 같은 request ID 재시도를 어떻게 처리할 것인가 | 원래 request ID를 보존하고 중복 호출 여부를 기록 | API와 로그 schema |

Claude 제품명과 가격은 변할 수 있으므로 source code, checkpoint 또는 학습 라벨에 실제
model ID를 고정하지 않는다. mapping 변경은 ModernBERT 재학습 사유가 아니다.

### 4-2. Feedback export와 학습 라벨 채택

ML은 온라인 DB를 직접 읽지 않고 versioned manifest와 Parquet 배치를 받는다. 제안 경로는
다음과 같다.

```text
feedback/<batch-id>/manifest.json
feedback/<batch-id>/events.parquet
```

| 주제 | 결정할 질문 | ML 측 제안 | 결정 결과가 반영될 곳 |
| --- | --- | --- | --- |
| export 주체 | 누가 언제 DB에서 batch를 만들고 GCS에 게시하는가 | 온라인 exporter가 완료된 immutable batch 생성 | exporter, trigger runbook |
| 전달 보증 | export가 이미 “유효한 escalation feedback만 포함”함을 보증하는가 | 보증한다면 `escalation_count`는 ML 채택 조건이 아니라 감사 metadata로 완화 가능 | schema와 validator config |
| 라벨 의미 | 최종 만족 tier를 언제 학습 정답으로 인정하는가 | 만족 신호 하나만으로 즉시 채택하지 않고 반복·충돌·품질 검증 적용 | preprocessing policy |
| 필수 필드 | question, initial/final label, 만족 여부, version 중 무엇이 필수인가 | 재현과 감사에 필요한 필드는 유지하되 ML 판단에 쓰지 않는 필드는 optional로 구분 | `feedback-export` schema |
| 중복·충돌 | 같은 질문에 다른 final label이 있으면 어떻게 할 것인가 | 자동 채택하지 않고 quarantine 후 집계 규칙 적용 | adapter config와 metric |
| 개인정보 | PII 제거, ML 재검사, 원문 보관과 삭제 기간은? | 온라인 1차 제거, ML 2차 검사·격리 | 개인정보·retention 정책 |
| sample weight | 반복 횟수와 RouterBench/feedback 혼합 비율은? | 현재 `1.0 + 0.25×반복`, 최대 2.0은 실험값 | versioned training config |
| trigger N | 유효 feedback 몇 건과 어떤 label 분포에서 재학습할 것인가 | 총 N뿐 아니라 class별 최소량, 최대 대기 기간, 예상 GPU 비용을 함께 사용 | scheduler/trigger contract |
| trigger 주체 | 온라인 scheduler, Kubeflow scheduler, 수동 승인 중 무엇인가 | batch 게시 후 idempotent pipeline 호출, 초기에는 수동 승인 권장 | trigger service와 runbook |

핵심 구분은 다음과 같다.

- 온라인 팀은 **어떤 이벤트를 feedback export에 넣을지** 책임진다.
- ML은 전달된 이벤트가 **학습 예시로 안전하고 일관적인지** 검증한다.
- upstream이 escalation 여부를 보증한다면 ML이 같은 비즈니스 규칙을 중복 구현할 필요는
  없다. 다만 event provenance와 감사 필드는 남겨야 잘못된 batch를 추적할 수 있다.
- N은 단순 행 수가 아니라 dedup과 품질 필터를 통과한 `accepted_count`를 기준으로 한다.

현재 `feedback-export-v0.1` prototype은 `escalation_count>=1`을 자동 채택 조건으로
검사한다. upstream 보증 방식으로 합의하면 이 조건을 optional 감사 필드로 바꾸고,
validator·fixture·schema version을 함께 수정해야 한다.

### 4-3. 모델 package, 승인과 온라인 적용

Kubeflow가 만드는 결과와 온라인 배포를 다음 경계로 나눈다.

```text
train -> evaluate -> package -> deploy gate
                              |
                              +-> rejected
                              +-> candidate/approved 가능 상태
                                      |
                                      +-> 온라인 health check
                                      +-> serving 전환
                                      +-> rollback 보존
```

| 주제 | 결정할 질문 | ML 측 제안 | 결정 결과가 반영될 곳 |
| --- | --- | --- | --- |
| gate 기준 | baseline 대비 어떤 metric과 허용 하락폭을 적용할 것인가 | RouterBench holdout, RouterArena alignment, severe under-routing을 함께 사용 | versioned gate rules JSON |
| 승인 주체 | gate 통과가 자동 승인인지 사람 승인 대기인지 | 첫 운영은 사람 승인, 안정화 후 자동화 검토 | GCS prefix/manifest 상태 |
| package 위치 | candidate/approved/rejected를 어떤 GCS prefix에 둘 것인가 | immutable version 디렉터리와 checksum manifest | GCS layout |
| 변경 감지 | FastAPI가 새 approved version을 poll, event 또는 배포 설정으로 받는가 | 명시적 배포 작업이 version을 전달 | serving deployment spec |
| 적용 방식 | hot reload와 rolling restart 중 무엇을 쓸 것인가 | 첫 버전은 rolling restart와 startup health check | serving runbook |
| rollback | 장애 시 이전 모델로 어떻게 복귀하는가 | 직전 approved package와 image digest를 항상 보존 | rollback runbook |
| 책임 경계 | 누가 candidate를 approved로 바꾸고 누가 실제 트래픽을 전환하는가 | ML은 evidence/package 제공, 온라인·운영이 최종 승인과 전환 | 권한/IAM 정책 |

### 4-4. 관측성과 최종 실험 결과

공통으로 연결해야 할 최소 식별자는 `request_id`, `router_model_version`,
`routing_policy_version`, `data_version`, `evaluation_version`이다. 온라인 팀은 latency,
token, cost, cache hit/miss와 오류를 기록하고 ML은 학습·평가 metric을 기록한다.

최종 보고에서는 다음 결과를 서로 섞지 않고 결합한다.

- ML router: tier 분포, RouterBench holdout, RouterArena 외부 정렬, under/over-routing
- 온라인 cache: QQP threshold, hit/false-hit, cache latency
- 전체 시스템: always-high-tier 대비 품질·비용·응답시간

## 5. 인프라 회의에서 결정할 사항

| 주제 | 결정할 질문 | 현재 상태 또는 제안 |
| --- | --- | --- |
| Google Cloud project | 현재 접근 가능한 project를 그대로 쓸 것인가 | `project-763753da-fa8f-4afd-b31`은 팀 확인 전 |
| region | GCS, Artifact Registry, GKE의 region은 어디인가 | 데이터 이동·GPU 가용성·팀 위치를 함께 고려 |
| GCS | bucket 이름과 datasets/models/evaluations/pipeline-root prefix는? | bucket은 아직 없음 |
| Artifact Registry | repository와 image tag/digest 정책은? | repository/API 아직 없음, 운영은 digest 고정 권장 |
| GKE | cluster, namespace, CPU/GPU node pool은 어떻게 나눌 것인가 | CPU 상시 최소화, GPU scale-to-zero/Spot 가능성 검토 |
| GPU 평가 | full 평가를 학습 GPU에서 이어서 실행할 것인가 | full·반복 평가는 GPU, 작은 smoke는 CPU 허용 |
| Kubeflow | 설치 방식, version, pipeline root, MLMD 보존 기간은? | 로컬 KFP v2 IR compile만 검증됨 |
| IAM | 누가 GCS read/write, image pull, pipeline submit, approved 승격 권한을 갖는가 | Workload Identity와 최소 권한 원칙 |
| secret | Claude/API credential을 어디서 주입하는가 | source, image와 pipeline parameter에 평문 저장 금지 |
| 비용 | 월 예산, alert, T4 quota와 첫 smoke 한도는? | 과금 작업은 팀 승인 후 수행 |
| 장애 | Spot 중단, retry, checkpoint resume와 cleanup 정책은? | retry는 있으나 resume/실클러스터 검증은 남음 |

## 6. 회의 전에 ML이 전달할 자료

- 3-2의 MVP·졸업프로젝트 개요 충돌표와 각 항목의 현행 코드 근거
- `feedback-export-v0.1` 합성 manifest/events fixture와 채택·격리 예시
- Router API 최소 요청·응답 fixture
- `model-package-v1` manifest, checksum과 gate 결과 예시
- 초기 학습·feedback 재학습·자격검증 pipeline DAG
- CPU/GPU image 및 Pod 자원 표
- 현재 cloud project 실측 상태와 새로 생성할 resource 목록

## 7. 회의가 끝났다고 판단할 산출물

- MVP·졸업프로젝트 개요 충돌별 채택 기준, 결정자·결정일과 문서 수정 담당자
- 결정자와 결정일이 적힌 Router API 및 feedback schema version
- 실제 Claude model ID를 code 밖에서 바꿀 수 있는 routing config
- feedback 채택 규칙, trigger N·최대 대기 기간과 실행 주체
- GCS prefix, 승인 주체, serving 적용·health check·rollback 절차
- project, region, bucket, repository, cluster, namespace, service account 이름
- 예산·quota와 첫 cloud smoke 실행 한도
- 변경된 fixture 및 양쪽 contract test 담당자

## 8. 계약 변경 원칙

- 필드 의미·타입·필수 여부가 호환되지 않게 바뀌면 `schema_version`을 올린다.
- 이미 export된 batch의 재현을 위해 기존 version adapter를 즉시 삭제하지 않는다.
- 계약 변경 PR은 schema, fixture와 양쪽 contract test를 함께 갱신한다.
- 합의되지 않은 DB table, bucket, namespace, service account와 실제 model ID를 ML core에
  고정하지 않는다.
- 제안값과 승인값을 구분하고 승인값에는 결정자·결정일·근거 실험을 남긴다.
