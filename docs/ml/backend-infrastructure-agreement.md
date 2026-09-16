# ML-온라인·인프라 협의 명세

상태: **2026-08-14 안 A 확정 — 관리형 Vertex AI Pipelines, GKE 미사용**

기준일: 2026-08-14

> 2026-09-16 범위 보정: production ModernBERT 초기학습과 여러 backbone의 학술적 비교는
> Colab에서 실행한다. 아래 KFP SDK + Vertex AI Pipelines/Custom Training 계약은 ModernBERT
> feedback 재학습에만 적용한다. 초기학습·비교용 GCP training job은 제출하지 않으며,
> 비교 결과가 ModernBERT 외 모델을 자동 승격시키지 않는다.

## 1. 목적

이 문서는 온라인 서비스 계약과 GCP 운영 경계를 팀에 전달하고 남은 값을 회의에서
결정하기 위한 기준이다. 2026-08-14에 5-1의 **안 A — KFP SDK + 관리형 Vertex AI
Pipelines + Vertex AI Custom Training**을 확정했다. GKE와 자체 설치 Kubeflow는
사용하지 않는다. 안 B/C는 선택 과정과 비용 비교를 위한 역사 기록이다. 그 밖의
기본값은 구현 가능한 제안일 뿐 운영 확정값이 아니다. 작업 완료 여부는
`docs/ml/workstream-checklist.md`에서만 관리한다.

실제 회의는 `docs/ml/backend-ml-integration-meeting-guide.md`를 위에서 아래로 진행하고,
확정 결과를 이 문서의 운영 계약과 schema에 반영한다.

회의는 다음 세 종류로 나눈다.

1. 문서 정합성 협의: MVP·졸업프로젝트 개요와 현행 ML 설계 중 어느 쪽을 기준으로 삼을지
2. 온라인·백엔드 협의: Spring, FastAPI serving, Claude 호출, DB와 feedback export
3. 인프라 전달·협의: 확정한 Vertex backend의 resource 이름, IAM 주체와 비용
   확인값 결정

## 2. Colab 초기학습·Vertex feedback 재학습 기준

아래 기준은 안 A 확정 후의 현행 운영 계약이다.

- ML 분류 라벨과 checkpoint class 순서는 `easy=0`, `medium=1`, `hard=2`다.
- ML은 난이도 라벨을 반환하고 실제 Claude model ID 선택은 온라인 routing 정책이 맡는다.
- 초기 데이터 처리와 ModernBERT 초기학습은 공통 core를 호출하는 Colab notebook에서
  수행하고 Drive checkpoint·checksum으로 재현한다. feedback 재학습 데이터 처리는
  Vertex AI Pipelines CPU component에서 수행한다.
- GKE cluster와 자체 설치 Kubeflow를 생성하지 않는다.
- ModernBERT feedback 재학습만 Vertex AI Custom Training의 T4 GPU 1장을 사용한다.
  Google Cloud Pipeline Components가 job을 제출·감시하며 production job을 수동 제출하지 않는다.
- RouterArena 전처리와 checkpoint 평가는 CPU component로 실행한다.
- QQP embedding과 cache threshold는 온라인 cache 팀 책임이며 ML pipeline DAG에 넣지
  않는다.
- Vertex AI Pipeline은 승인 후보를 만드는 곳까지 담당한다. 실제 서빙 모델 전환과 rollback은
  온라인 운영 계약이 필요하다.
- project는 `project-763753da-fa8f-4afd-b31`(`HYUCapstone`), 기본 region은
  `us-west1`이다(2026-08-15 확정, 2026-08-14 `us-central1` 기준을 대체 — GCS
  Always Free는 `us-east1`/`us-west1`/`us-central1` 셋 다 동일 적용되고
  `us-west1`이 한국에서의 인터랙티브 작업 지연이 더 짧다. T4 quota는 `us-west1`
  기준으로 재확인 필요). image는 Artifact Registry, data/checkpoint/metric은
  GCS에 둔다.
- 초기학습 run은 Colab runtime·repo commit·Drive checkpoint checksum을 기록한다. feedback
  재학습 run은 KFP/metadata backend에 image digest, 입력·출력 GCS URI와 Vertex CustomJob
  resource를 기록한다.
- 비용 목표는 Welcome Credit과 Always Free 범위 안의 **실제 카드 청구액 0원**이다.
  GPU와 node VM이 무료라는 뜻이 아니다. credit 잔액·만료일 확인, GPU quota 1,
  동시 실행 1, retry 0, 최대 6시간, budget alert/가능한 Spend Cap과 실행 후 cleanup을
  필수 조건으로 둔다.

팀원에게 요구되는 것은 확정 구조의 영향 인지와 다음 운영값 확정이다.

- backend는 Vertex AI job을 직접 호출하지 않으며 승인된 model package만 인수한다.
- Spring은 합의된 inline JSON event를 FastAPI에 보내고, FastAPI가 ML용 GCS batch를
  생성·게시한 뒤 KFP 실행을 trigger한다.
- bucket, repository, cluster, namespace, service account 이름과 권한 담당자는 회의에서
  채우되 실행 구조는 바꾸지 않는다.
- credit 잔액이 부족하거나 T4 quota가 없으면 실행을 보류하거나 데이터·자원 규모를
  줄인다.

`mvp-scope-and-checklist.md`, `졸업프로젝트 개요.md`, 체크리스트와 KFP 코드는 안 A로
정정한다.

## 3. MVP·졸업프로젝트 개요 정합성 협의

### 3-1. 협의 원칙

- `mvp-scope-and-checklist.md`는 2026-07 시점의 역사적 MVP 범위라는 표시를 유지한다.
- `졸업프로젝트 개요.md`는 최종 목표와 현행 설계가 함께 보이므로, 현재 구현과 다른
  내용을 단순 과거 기록으로 방치하지 않고 `확정`, `계획`, `미구현`으로 구분한다.
- 아래 충돌은 ML 코드의 임의 변경 사유가 아니다. 팀이 기준을 정한 뒤 개요·MVP,
  contract fixture, config와 코드가 같은 결론을 가리키도록 갱신한다.
- Claude 제품명·가격, GCP 비용과 일정처럼 시간에 따라 바뀌는 내용은 기준일과 출처를
  표시하고 인프라 착수 시 다시 확인한다.

### 3-2. 충돌 정리표

`[2026-08-11 기준안]`으로 표시한 행은 당시 통보 사항이었지만, 최초 학습을 Colab으로
옮기고 feedback 재학습 자원만 필요할 때 활성화하는 안이 추가되면서 5-1의 실행 backend
선택에 종속되었다. 5-1 결정 후 선택 결과로 다시 고정한다.

| 충돌 주제 | MVP·졸업프로젝트 개요의 표현 | 현행 ML 설계·구현 | 회의에서 결정할 사항 | ML 측 제안 |
| --- | --- | --- | --- | --- |
| RouterBench 라벨링 | 졸업프로젝트 개요는 전체 solve rate와 약/중/강 대표 모델의 결과를 주 라벨 신호로 설명 | 11개 모델을 empirical tier로 나누고 tier consensus로 최소 충분 capability를 판단하며 solve rate는 진단값 | 개요의 라벨 알고리즘을 현행 방식으로 교체할지 | 11개 모델 tier-consensus를 공식 기준으로 채택 |
| 초기 데이터 처리 위치 | 개요와 일부 구현은 로컬 또는 Kubeflow 시작점을 다르게 표현 | 2026-08-11 기준은 Kubeflow CPU였으나 최초 baseline Colab 안을 재검토 | `[2026-08-11 기준안]`과 5-1 선택을 맞출 것 | 동일 Python core와 importer fast path 유지 |
| RouterArena 평가 자원 | 기존 구현은 full checkpoint 추론을 GPU image로 실행 | 평가는 CPU로 유지하되 최초 baseline은 Colab, feedback 평가는 선택 backend | 평가 위치만 5-1 결과로 정하고 GPU request는 제거 | CPU batch·시간 측정 후 resource 조정 |
| Claude mapping | MVP와 개요는 `easy→Haiku`, `medium→Sonnet`, `hard→Opus`를 확정 mapping처럼 표현 | ML은 3-class까지만 고정하고 실제 model ID는 versioned 온라인 정책으로 관리 | 현재 사용 가능한 실제 model ID, fallback과 변경 주체 | 계열명은 설명용 예시로 남기고 실제 ID는 코드 밖 설정 |
| feedback 라벨 의미 | 최종 만족 tier를 해당 질문의 실제 필요 수준이자 정정 라벨로 곧바로 간주 | 단발 만족 신호는 노이즈가 있으므로 schema·PII·중복·충돌·반복 근거를 검증한 뒤 채택 | 어떤 조건에서 final tier를 학습 정답으로 인정할지 | upstream 보증과 ML 품질 필터를 함께 적용 |
| escalation 재검사 | 개요는 escalation 기록 자체가 ML 정정 라벨 공급원이라고 설명 | 온라인 export가 유효 feedback만 보증한다면 ML이 escalation 비즈니스 규칙을 중복 검사할 필요가 없음 | `escalation_count`를 필수 채택 조건으로 유지할지 감사 metadata로 완화할지 | upstream 계약 확정 시 감사 metadata로 완화 |
| 재학습 trigger | MVP·개요는 정정 라벨 N건 도달 시 Kubeflow 자동 실행으로 표현 | Spring이 threshold를 소유하고 FastAPI는 1건 이상 batch를 허용 | Spring의 N·분포·최대 대기 정책 | dedup 후 accepted count와 분포·비용을 기준으로 조정 |
| Kubeflow 시작 범위 | 개요는 `데이터→라벨링→학습→평가→배포` 전체 자동화로 표현 | 최초 baseline은 Colab, feedback 재학습만 KFP pipeline | 초기/재학습 시작점을 분리 기술 | Colab initial과 KFP feedback-retraining을 구분해 기술 |
| Kubeflow 배포 범위 | MVP·개요는 gate 통과 모델이 바로 재배포되는 것처럼 표현 | gate 통과 후 자동 승인·CURRENT 갱신, FastAPI hot swap까지 구현 | 실배포 end-to-end 검증과 gate 임계값 | 자동 경로를 유지하고 rollback만 운영 수단으로 보존 |
| API label 계층 | MVP stub 설명은 Spring `LIGHT/MEDIUM/HEAVY`, ML 문서는 `easy/medium/hard` 사용 | checkpoint와 ML API 경계는 `easy/medium/hard`로 고정 | Spring이 같은 이름을 쓸지 내부 adapter를 둘지 | 경계는 `easy/medium/hard`, 필요하면 Spring 내부에서만 변환 |
| Router API schema | MVP 공개 `/api/chat`은 `message`, `conversationId` 중심이고 ML 내부 제안은 `question`, `request_id` 중심 | 서로 다른 계층이지만 명시적 변환 계약이 없음 | 공개 요청에서 내부 Router 요청으로 누가 어떻게 변환할지 | Spring adapter와 양쪽 fixture를 확정 |
| QQP 책임 | MVP·개요는 QQP threshold 평가를 전체 완료 기준으로 포함 | ML/Kubeflow에서는 제거하고 온라인 cache 팀 결과만 최종 보고에 결합 | 전체 목표는 유지하되 담당·산출물을 어디에 둘지 | 온라인 cache 팀 소유로 명시 |
| RAG 의미 | MVP 라벨 명세에는 RouterBench RAG eval-family 포함 여부 실험이 있고 서비스 문맥에서는 RAG 구현 질문도 있었음 | RouterBench 데이터 sensitivity는 남지만 retrieval-augmented generation 서비스는 구현하지 않음 | 두 의미를 문서에서 어떻게 구분할지 | `RouterBench RAG eval-family`와 `service RAG`를 명시적으로 분리 |
| cloud 준비 상태 | 개요는 GCS/GKE/Kubeflow와 비용·일정을 구축 계획 또는 사용 가능 상태처럼 설명 | project/billing 연결만 확인, resource는 아직 미생성 | architecture 선택과 실제 provision 상태를 분리 표기 | 현재 상태 표와 갱신일을 개요에 추가 |
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

ML은 온라인 DB나 HTTP JSON을 직접 읽지 않고 FastAPI가 생성한 versioned manifest와
Parquet 배치를 받는다. 내부 경로는 다음 형태다.

```text
<derived-root>/<candidate-version>/feedback/manifest.json
<derived-root>/<candidate-version>/feedback/events.parquet
```

| 주제 | 결정할 질문 | ML 측 제안 | 결정 결과가 반영될 곳 |
| --- | --- | --- | --- |
| export 주체 | 누가 HTTP event를 ML artifact로 만드는가 | Spring은 inline JSON 전송, FastAPI가 Parquet·manifest·GCS 게시 담당 | API, adapter, trigger runbook |
| 전달 보증 | export가 이미 “유효한 escalation feedback만 포함”함을 보증하는가 | 보증한다면 `escalation_count`는 ML 채택 조건이 아니라 감사 metadata로 완화 가능 | schema와 validator config |
| 라벨 의미 | 최종 만족 tier를 언제 학습 정답으로 인정하는가 | 만족 신호 하나만으로 즉시 채택하지 않고 반복·충돌·품질 검증 적용 | preprocessing policy |
| 필수 필드 | question, initial/final label, 만족 여부, version 중 무엇이 필수인가 | 재현과 감사에 필요한 필드는 유지하되 ML 판단에 쓰지 않는 필드는 optional로 구분 | `feedback-export` schema |
| 중복·충돌 | 같은 질문에 다른 final label이 있으면 어떻게 할 것인가 | 자동 채택하지 않고 quarantine 후 집계 규칙 적용 | adapter config와 metric |
| 개인정보 | PII 제거, ML 재검사, 원문 보관과 삭제 기간은? | 온라인 1차 제거, ML 2차 검사·격리 | 개인정보·retention 정책 |
| sample weight | 반복 횟수와 RouterBench/feedback 혼합 비율은? | 현재 `1.0 + 0.25×반복`, 최대 2.0은 실험값 | versioned training config |
| trigger N | 유효 feedback 몇 건과 어떤 label 분포에서 재학습할 것인가 | 총 N뿐 아니라 class별 최소량, 최대 대기 기간, 예상 GPU 비용을 함께 사용 | scheduler/trigger contract |
| trigger 주체 | 유효 feedback 수가 언제 재학습을 시작시키는가 | Spring이 threshold를 판단하고 inline event로 FastAPI `/escalate`를 호출한다. FastAPI는 1건 이상을 허용한다 | trigger service와 runbook |

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
| 승인 주체 | gate 통과가 자동 승인인지 사람 승인 대기인지 | deploy gate 통과 시 pipeline이 자동 승인한다 | GCS prefix/manifest 상태 |
| package 위치 | candidate/approved/rejected를 어떤 GCS prefix에 둘 것인가 | immutable version 디렉터리와 checksum manifest | GCS layout |
| 변경 감지 | FastAPI가 새 approved version을 poll, event 또는 배포 설정으로 받는가 | GCS `CURRENT.json`을 기본 60초 간격으로 polling한다 | serving deployment spec |
| 적용 방식 | hot reload와 rolling restart 중 무엇을 쓸 것인가 | 새 model을 별도 적재·검증한 뒤 참조를 원자 교체한다 | serving runbook |
| rollback | 장애 시 이전 모델로 어떻게 복귀하는가 | 직전 approved package와 image digest를 항상 보존 | rollback runbook |
| 책임 경계 | 누가 candidate를 approved로 바꾸고 누가 실제 트래픽을 전환하는가 | ML pipeline이 gate 기반으로 CURRENT를 갱신하고 FastAPI가 검증 후 자동 전환한다 | 권한/IAM 정책 |

### 4-4. 관측성과 최종 실험 결과

공통으로 연결해야 할 최소 식별자는 `request_id`, `router_model_version`,
`routing_policy_version`, `data_version`, `evaluation_version`이다. 온라인 팀은 latency,
token, cost, cache hit/miss와 오류를 기록하고 ML은 학습·평가 metric을 기록한다.

최종 보고에서는 다음 결과를 서로 섞지 않고 결합한다.

- ML router: tier 분포, RouterBench holdout, RouterArena 외부 정렬, under/over-routing
- 온라인 cache: QQP threshold, hit/false-hit, cache latency
- 전체 시스템: always-high-tier 대비 품질·비용·응답시간

## 5. 인프라 실행 방식과 운영값

아래 표는 2026-08-14 확정한 안 A를 실제 resource로 만들 때 필요한 이름·권한·확인값이다.

| 주제 | 결정할 질문 | 현재 상태 또는 제안 |
| --- | --- | --- |
| Google Cloud project | 누가 project와 billing을 관리하는가 | `project-763753da-fa8f-4afd-b31`(`HYUCapstone`) 확정 |
| region | 각 resource 위치의 세부값은 무엇인가 | `us-west1` 확정(2026-08-15, `us-central1`에서 변경) |
| GCS | bucket 이름과 datasets/models/evaluations/pipeline-root prefix는? | bucket은 아직 없음 |
| Artifact Registry | repository와 image tag/digest 정책은? | repository/API 아직 없음, 운영은 digest 고정 권장 |
| GKE | 생성하는가 | 사용하지 않음 |
| GPU 학습 | Vertex job 제한값과 quota를 누가 확인하는가 | T4 1, replica 1, concurrency 1, retry 0, 6시간 상한 |
| 평가 | CPU resource와 batch/timeout은 얼마인가 | Vertex managed CPU, full 측정 후 값 조정 |
| Pipeline | KFP/GCPC version, pipeline root, metadata 보존 기간은? | KFP 2.17.0, GCPC 2.22.0, 세부값 미정 |
| IAM | PipelineJob, CustomJob, GCS와 approved 승격 권한 주체는? | service account와 keyless 최소 권한 원칙 |
| secret | Claude/API credential을 어디서 주입하는가 | source, image와 pipeline parameter에 평문 저장 금지 |
| 비용 | credit 잔액·만료일, alert/Spend Cap과 첫 smoke 한도는? | 실제 카드 청구 0원 guardrail 적용 후 수행 |
| 장애 | Vertex timeout/cancel, checkpoint resume와 cleanup 정책은? | 6시간 timeout·retry 0 확정, resume는 미구현 |

### 5-1. 실행 방식 비교와 확정 결과

> **확정 결과(2026-08-14): 안 A를 채택한다.** 아래 안 B/C는 선택 근거를 보존하는
> 역사적 비교이며 구현·인프라 생성 대상이 아니다.

이 절은 최초 RouterBench 학습·RouterArena 평가는 Colab 무료 세션에서 완료하고, GCS에
등록한 baseline model 이후의 **feedback 재학습 1회**만 비교한다. 평상시 FastAPI
Router, Gateway, DB와 cache의 온라인 serving 비용은 세 방식에서 동일하므로 제외한다.

#### 비교용 기준 시나리오

아직 trigger `N`과 실제 full-run 시간이 확정되지 않았으므로 아래 값은 견적을 비교하기
위한 가정이지 예산 승인값이 아니다. 아래 표의 지역·단가는 **2026-08-12,
`us-central1` 기준으로 동결된 역사적 비교 수치**다. 2026-08-15 운영 region이
`us-west1`로 바뀌었지만 이 비교표는 안 A/B/C 선택 근거를 보존하기 위한 것이라
재계산하지 않는다. 실제 첫 GPU run 전에는 `us-west1` 단가를 Cloud Billing
calculator에서 다시 확인한다.

| 항목 | 비교 가정 |
| --- | --- |
| 기준일·지역 | 2026-08-12, `us-central1`, USD on-demand list price |
| feedback batch | 비용 비교를 위한 inline event 1,000건 예시; HTTP 계약의 필수 건수는 아님 |
| CPU 작업 | schema 검증·정규화·병합·평가·package·gate 합계 2시간, `e2-standard-4` 약 `$0.134/hour` |
| GPU 작업 | `n1-standard-4` + NVIDIA T4 1장, 6시간, replica/concurrency 1, retry 0 |
| 전체 경과시간 | 준비와 종료를 포함해 약 8시간 |
| Artifact | GCS Standard 10 GiB를 30일 보관 |
| GKE 비교값 | 안 B/C 비교를 위해서만 zonal Standard 1개를 가정; 현행 안 A에는 없음 |
| 할인 | Spot·CUD 제외. 안 B/C의 GKE credit과 저장소 free tier는 비교표에만 별도 표시 |
| 환율 | 표는 USD가 기준이며 원화 환산은 예시로 `1 USD = 1,450원` 사용 |

단가 근거와 계산상 주의점은 다음과 같다.

- Vertex AI Pipelines는 KFP SDK로 만든 pipeline을 serverless backend에서 실행하며,
  pipeline 실행 자체는 약 `$0.03/run`부터 시작한다. CPU·GPU·GCS 등 하위 resource는
  별도 과금된다.
- Compute Engine의 `us-central1` T4 on-demand 단가는 `$0.35/GPU-hour`다. GPU VM은
  GPU 외에도 host VM 비용을 함께 낸다.
- Vertex AI Custom Training에는 Compute Engine 상당 비용과 Vertex 관리 비용이 붙을 수
  있으므로 이 문서에서는 `n1-standard-4 + T4`를 보수적으로 `$0.54~$0.62/hour` 범위로
  잡는다. 실제 실행 전 Cloud Billing SKU/공식 calculator에서 다시 확정한다.
- zonal GKE의 관리비는 `$0.10/cluster-hour`지만 결제 계정당 월 `$74.40` free-tier
  credit은 zonal 또는 Autopilot cluster 한 개의 관리비에만 적용된다. **node VM과 GPU에는
  적용되지 않는다.**
- `us-central1` GCS Standard는 약 `$0.02/GiB-month`이고 Always Free 5 GiB-month가 있다.
  따라서 10 GiB 보관은 총액 `$0.20/month`, 다른 사용량이 없다면 free tier 반영 후 약
  `$0.10/month`다.
- GKE KFP metadata용 50 GiB Standard Persistent Disk는 총액 약 `$2.00/month`다.
  현재 미국 지역 계정 단위 30 GiB free allowance가 남아 있으면 약 `$0.80/month`이지만,
  다른 VM disk가 이미 allowance를 쓰고 있으면 총액에 가까워진다.
- Artifact Registry, Logging API 호출, 소량의 GCS operation, image pull과 동일 지역
  network 비용은 사용량에 따라 달라 비교표에서는 제외한다. 외부 IP나 Load Balancer를
  만들면 별도 상시 비용이 생기므로 KFP UI는 필요할 때만 안전한 접속 경로로 연다.

#### 안 A — KFP SDK + Vertex AI 관리형 Pipelines + Vertex GPU

```text
feedback batch 확정
  -> Spring inline JSON -> FastAPI artifact 생성 및 Vertex AI PipelineJob 제출
  -> 관리형 CPU component: 검증·정규화·병합
  -> Vertex AI Custom Training: T4 1장으로 재학습
  -> 관리형 CPU component: RouterBench/RouterArena 평가·package·gate
  -> approved 후보를 GCS에 기록
  -> 모든 실행 자원 종료
```

이 안에서는 Kubeflow를 GKE에 설치하지 않는다. KFP SDK와 KFP IR YAML은 사용하지만 실행
화면과 metadata backend는 Google Cloud 관리형 서비스를 사용한다.

| 비용 항목 | 계산 | 예상 |
| --- | --- | ---: |
| Pipeline 실행 | 1회 | `$0.03` |
| CPU component | `$0.134 × 2h` | `$0.27` |
| Vertex T4 학습 | `$0.54~0.62 × 6h` | `$3.24~3.72` |
| GCS 10 GiB/월 | 총액 `$0.20`, free tier 잔여 시 약 `$0.10` | `$0.10~0.20` |
| **재학습 1회 합계** | 다른 소액 비용 제외 | **`$3.64~4.22` (약 5,300~6,100원)** |
| **재학습 없는 달의 ML pipeline 비용** | Artifact 10 GiB만 보관 | **`$0.10~0.20/month`** |

장점은 실제 요구대로 재학습 때만 orchestrator와 GPU가 실행되고, KFP server node의 유휴
비용이 없다는 점이다. 단점은 Kubeflow 공식 UI가 아니라 GCP Pipeline UI를 쓰고, 일부
Kubernetes 전용 기능을 사용할 수 없다는 점이다.

#### 안 B — GKE KFP Standalone + Vertex AI GPU

```text
외부 trigger 또는 운영자가 GKE CPU node pool 0 -> 1
  -> KFP backend와 UI Ready 확인
  -> KFP pipeline 제출
  -> GKE CPU component: 검증·정규화·병합
  -> KFP launcher가 Vertex AI T4 CustomJob 제출·감시
  -> GKE CPU component: 평가·package·gate
  -> MLMD/GCS 결과 보존
  -> GKE CPU node pool 1 -> 0
```

2026-08-11에 확정했던 기준안이 이 방식이다. CPU node가 0이면 Kubeflow 자체도 실행되지
않으므로 Kubeflow가 스스로 재학습을 시작할 수 없다. 백엔드 이벤트, Cloud Run/Workflow,
CI 또는 사람이 먼저 node를 깨우는 외부 bootstrap 절차가 필요하다. CPU node는 실제 CPU
연산 2시간뿐 아니라 Vertex job을 제출·감시하는 전체 약 8시간 동안 켜져 있다고 계산한다.

| 비용 항목 | 계산 | 예상 |
| --- | --- | ---: |
| GKE CPU node | `$0.134 × 8h` | `$1.07` |
| Vertex T4 학습 | `$0.54~0.62 × 6h` | `$3.24~3.72` |
| GKE 관리비 | `$0.10 × 8h = $0.80`, 해당 free-tier credit 잔여 가정 | `$0.00` net |
| KFP disk 50 GiB/월 | 총액 약 `$2.00`, 30 GiB allowance 잔여 가정 | 약 `$0.80` |
| GCS 10 GiB/월 | 총액 `$0.20`, free tier 잔여 시 약 `$0.10` | `$0.10~0.20` |
| **node를 재학습 때만 켠 달의 1회 합계** | 다른 소액 비용 제외 | **`$5.21~5.79` (약 7,600~8,400원)** |

CPU node를 0으로 내리지 않고 한 달 내내 유지하면 `e2-standard-4`만 약
`$0.134 × 730h = $97.82/month`가 된다. 재학습 1회를 더하면 대략
**`$101.96~102.54/month`**이므로 “재학습 때만 활성화” 목표에는 맞지 않는다.

이 안은 Kubeflow 공식 UI와 자체 MLMD를 보여줄 수 있고 GPU 운영은 Vertex에 맡긴다. 반면
KFP bootstrap, MLMD/PV 보존, node 0 복귀와 실패 cleanup 자동화가 추가된다.

#### 안 C — GKE KFP Standalone + GKE GPU node pool

```text
외부 trigger 또는 운영자가 GKE CPU node pool 0 -> 1
  -> KFP backend Ready 및 pipeline 제출
  -> GKE CPU component: 검증·정규화·병합
  -> GPU node pool 0 -> 1, driver Ready 확인
  -> KFP가 T4 GPU training Pod 실행
  -> checkpoint를 GCS에 저장하고 GPU node pool 1 -> 0
  -> GKE CPU component: 평가·package·gate
  -> CPU node pool 1 -> 0
```

| 비용 항목 | 계산 | 예상 |
| --- | --- | ---: |
| GKE CPU node | `$0.134 × 8h` | `$1.07` |
| GKE GPU node | `n1-standard-4 약 $0.19 + T4 $0.35 = $0.54 × 6h` | `$3.24` |
| GKE 관리비 | `$0.10 × 8h = $0.80`, 해당 free-tier credit 잔여 가정 | `$0.00` net |
| KFP disk 50 GiB/월 | 총액 약 `$2.00`, 30 GiB allowance 잔여 가정 | 약 `$0.80` |
| GCS 10 GiB/월 | 총액 `$0.20`, free tier 잔여 시 약 `$0.10` | `$0.10~0.20` |
| **node를 재학습 때만 켠 달의 1회 합계** | GPU boot disk 등 소액 제외 | **`$5.21~5.31` (약 7,600~7,700원)** |

Vertex 관리비를 피해서 안 B보다 1회 최대 약 `$0.48` 저렴할 수 있지만, GPU quota·zone
capacity, GPU node autoscaling, NVIDIA driver, taint/toleration, Pod 종료 후 node 0 복귀를
직접 관리해야 한다. GPU node를 실수로 한 달 동안 유지하면 host와 T4만 약
`$0.54 × 730h = $394.20/month`가 추가되므로 자동 cleanup 검증이 필수다.

#### 비용·운영 비교와 ML 측 권고

| 기준 | 안 A 관리형 KFP | 안 B GKE KFP + Vertex GPU | 안 C GKE KFP + GKE GPU |
| --- | --- | --- | --- |
| 재학습 1회 추정 | **`$3.64~4.22`** | `$5.21~5.79` | `$5.21~5.31` |
| 재학습 없는 달 | GCS 약 `$0.10~0.20` | KFP disk 약 `$0.80` + GCS | KFP disk 약 `$0.80` + GCS |
| Kubeflow 공식 UI | 없음, GCP UI 사용 | 있음 | 있음 |
| GPU 운영 | Google 관리 | Google 관리 | 팀이 node/driver 관리 |
| 완전 자동 on-demand | 가장 쉬움 | 외부 wake-up 필요 | 외부 wake-up + GPU readiness 필요 |
| 운영 난이도 | 낮음 | 중간 | 높음 |
| Kubernetes 자유도 | 낮음 | 높음 | 가장 높음 |
| 현재 코드 영향 | **KFP v2/Vertex CustomJob wrapper 구현됨** | GKE용 재설계 필요 | GKE GPU용 재설계 필요 |

ML 측 권고와 확정 결과는 다음과 같다.

1. 평가 기준이 **KFP SDK로 재현 가능한 pipeline 구현**이면 안 A를 채택한다. 비용과
   운영 복잡도가 가장 낮고 “재학습 때만 Kubeflow pipeline과 GPU 활성화” 요구에 가장
   정확히 맞는다.
2. 평가 기준에 **GKE에 직접 설치한 Kubeflow 공식 UI 시연**이 포함되면 안 B를 채택한다.
   안 C의 절감액은 6시간 재학습 기준 최대 약 `$0.48`뿐이라 GPU node 운영 위험을
   정당화하기 어렵다.
3. 안 C는 “GKE GPU node 운영 자체”가 졸업프로젝트 평가 목표일 때만 선택한다.
4. 어느 안이든 첫 cloud run 전 실제 billing account의 credit 잔액·만료일, T4 quota,
   현재 사용 중인 free-tier allowance와 Pricing Calculator 견적을 캡처한다. 표의 금액은
   credit 차감 전 사용액이며 credit이 충분할 때만 실제 카드 청구액 0원이 될 수 있다.

위 권고에 따라 2026-08-14 안 A를 확정했다. 2~3번 조건은 더 이상 열린 선택지가 아니다.

#### 회의에서 순서대로 결정할 질문

| 순서 | 질문 | 선택·기록란 |
| --- | --- | --- |
| 1 | 과제의 “Kubeflow 사용”이 KFP SDK+관리형 backend를 인정하는가, 공식 UI 설치까지 요구하는가? | `미정` |
| 2 | 안 A/B/C 중 공식 feedback 재학습 backend는 무엇인가? | `미정` |
| 3 | 안 B/C라면 누가 node `0↔1` bootstrap과 실패 cleanup을 수행하는가? | `미정` |
| 4 | KFP UI·MLMD를 몇 일 보관하고 PV를 계속 유지할 것인가? | `미정` |
| 5 | trigger `N`, 예상 GPU 시간과 1회 hard budget은 얼마인가? | `미정` |
| 6 | 첫 실행 전 calculator 견적, credit 잔액·만료일과 T4 quota를 누가 확인하는가? | `미정` |
| 7 | 선택 결과에 따라 MVP·개요·KFP 코드·IAM을 수정할 담당자와 PR은 무엇인가? | `미정` |

공식 근거:

- [Vertex AI 관리형 Pipelines와 KFP 실행](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
- [Kubeflow Pipelines backend와 Kubernetes Pod 실행](https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline/)
- [GKE 관리비·free-tier credit·node 별도 과금](https://cloud.google.com/kubernetes-engine/pricing)
- [Compute Engine T4 GPU 가격](https://cloud.google.com/products/compute/gpus-pricing)
- [Cloud Storage 가격과 Always Free](https://cloud.google.com/storage/pricing)
- [Persistent Disk 가격](https://cloud.google.com/compute/disks-image-pricing)

## 6. 회의 전에 ML이 전달할 자료

- 3-2의 MVP·졸업프로젝트 개요 충돌표와 각 항목의 현행 코드 근거
- `feedback-export-v0.1` 합성 manifest/events fixture와 채택·격리 예시
- Router API 최소 요청·응답 fixture
- `model-package-v1` manifest, checksum과 gate 결과 예시
- Colab 초기학습 notebook 흐름, feedback 재학습·자격검증 pipeline DAG
- CPU data/evaluation image·GPU training image와 A/B/C별 실행 위치 표
- 현재 cloud project 실측 상태와 새로 생성할 resource 목록

## 7. 회의가 끝났다고 판단할 산출물

- MVP·졸업프로젝트 개요 충돌별 채택 기준, 결정자·결정일과 문서 수정 담당자
- 결정자와 결정일이 적힌 Router API 및 feedback schema version
- 실제 Claude model ID를 code 밖에서 바꿀 수 있는 routing config
- feedback 채택 규칙, trigger N·최대 대기 기간과 실행 주체
- GCS prefix, 승인 주체, serving 적용·health check·rollback 절차
- bucket, repository, cluster, namespace, service account 이름과 담당자
- credit 잔액·만료일, 예산 guardrail·T4 quota와 첫 cloud smoke 실행 한도
- 변경된 fixture 및 양쪽 contract test 담당자

## 8. 계약 변경 원칙

- 필드 의미·타입·필수 여부가 호환되지 않게 바뀌면 `schema_version`을 올린다.
- 이미 export된 batch의 재현을 위해 기존 version adapter를 즉시 삭제하지 않는다.
- 계약 변경 PR은 schema, fixture와 양쪽 contract test를 함께 갱신한다.
- 합의되지 않은 DB table, bucket, namespace, service account와 실제 model ID를 ML core에
  고정하지 않는다.
- 제안값과 승인값을 구분하고 승인값에는 결정자·결정일·근거 실험을 남긴다.
