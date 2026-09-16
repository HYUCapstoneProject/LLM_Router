# Backend-ML 통합 협의 회의 진행 문서

상태: **회의 진행용 결정 기록지**

작성 기준일: 2026-08-14

> 2026-09-06 후속 결정: Spring이 재학습 threshold를 소유하고 FastAPI는 1건 이상을
> 허용한다. deploy gate 통과 모델은 사람 승인 없이 자동 승격되며 FastAPI가
> `CURRENT.json`을 polling해 hot swap한다. 아래의 수동 승인·rolling restart 선택지는
> 당시 회의용 초안으로만 남긴다.
>
> 2026-09-12 후속 결정: Spring은 `feedback-inline-v1` event 전체를
> `POST /escalate`로 보낸다. FastAPI가 Parquet·manifest·checksum·GCS 게시와 Vertex
> 제출을 담당한다. 정확한 Spring 계약은
> `LLM_Router-serving/docs/backend-integration-handoff.md`가 우선하며, 이 문서의 기존
> GCS exporter 안은 역사 기록이다.

회의 일시: ____________________

참석자: ____________________

진행자 / 기록자: ____________________ / ____________________

## 목차

1. [진행 규칙과 완료 조건](#1-진행-규칙과-완료-조건)
2. [이미 일치하는 방향 확인](#2-이미-일치하는-방향-확인)
3. [Gateway-FastAPI 계약](#3-gateway-fastapi-계약)
4. [Claude mapping과 장애 처리](#4-claude-mapping과-장애-처리)
5. [에스컬레이션과 만족 라벨](#5-에스컬레이션과-만족-라벨)
6. [Feedback export와 ML 입력](#6-feedback-export와-ml-입력)
7. [재학습 trigger와 데이터 정책](#7-재학습-trigger와-데이터-정책)
8. [모델 package, gate와 실제 배포](#8-모델-package-gate와-실제-배포)
9. [GCP-GCS-Vertex AI 인프라](#9-gcp-gcs-gke-kubeflow-인프라)
10. [MVP-졸업프로젝트 개요 정합성](#10-mvp-졸업프로젝트-개요-정합성)
11. [최종 결정표와 후속 작업](#11-최종-결정표와-후속-작업)

## 회의 요약

### 이 회의의 목적

팀원이 작성한 Gateway·Router Serving API 초안과 현재 ML 구현을 하나의 실행 가능한
계약으로 맞춘다. 회의가 끝나면 최소한 다음 연결이 모호하지 않아야 한다.

비교 기준은 팀원이 작성한 Gateway/API 문서와
`docs/ml/backend-infrastructure-agreement.md`의 현행 ML 제안이다. 팀원 문서에서
`/api/chat`과 Gateway 오류 형식은 구현 완료, `/api/escalate`, `/router/embed`,
`/router/classify`는 draft, GCS feedback dump는 확정 필요 상태다.

```text
Frontend -> Spring Gateway -> FastAPI Router
         -> Claude / Cache -> Feedback export
         -> Vertex AI Pipeline retrain -> Model gate
         -> Approved model -> FastAPI serving
```

### 현재 일치하는 핵심 방향

| 상태 | 주제 | 일치한 방향 또는 남은 빈칸 |
| --- | --- | --- |
| `일치` | 난이도 label | 서비스 경계와 checkpoint 모두 `easy/medium/hard` 사용 |
| `일치` | 실제 Claude 선택 | Gateway의 versioned 설정이 최종 결정 |
| `일치` | confidence | 기록·분석용이며 Gateway가 재분기하지 않음 |
| `일치` | DB 소유권 | FastAPI는 DB에 직접 연결하지 않음 |
| `일치` | cache | embedding 추론과 pgvector/threshold 결정을 분리하며 Gateway가 cache 정책 소유 |
| `부분 일치` | 추적성 | request·Claude·router·embedding version 방향은 같고 필드 집합은 미정 |
| `일치` | fallback 실험 | fallback 요청은 정상 라우팅 성능 집계에서 제외 |
| `일치` | feedback 전달 | Spring inline JSON → FastAPI 검증·Parquet/manifest/GCS 게시로 확정 |

### 2026-08-14 확정 인프라

> 관리형 Vertex AI Pipelines를 사용하는 안 A가 확정되었다. GKE와 자체 설치 Kubeflow는
> 생성하지 않는다. 이 문서의 A/B/C 비교는 결정 근거를 위한 역사 기록이다.

```text
Vertex AI PipelineJob
  ├─ 관리형 CPU component: 라벨링·전처리·feedback 병합·평가·package·gate
  └─ Vertex AI Custom Training: T4 GPU 1장, 학습만

Artifact Registry: CPU/GPU container image
GCS: dataset/config/checkpoint/metric
Vertex ML Metadata: CustomJob resource, image digest, GCS URI, 결과 상태
```

- project: `project-763753da-fa8f-4afd-b31`(`HYUCapstone`)
- 기본 region: `us-west1`(2026-08-15 확정, `us-central1`에서 변경. GCS Always Free는
  `us-east1`/`us-west1`/`us-central1` 동일 적용, `us-west1`이 한국에서 인터랙티브
  작업 지연이 더 짧음)
- GKE: 사용하지 않음
- 공식 feedback retraining job: Vertex AI PipelineJob을 통해서만 CustomJob 제출
- 기본 비용 guardrail: T4 1장, replica/concurrency 1, retry 0, 최대 6시간, budget alert,
  가능한 Spend Cap, run 종료 후 cleanup
- 비용 표현: “무료 GPU”가 아니라 Welcome Credit/Always Free 안에서 실제 카드 청구액
  0원을 목표로 함. billing admin의 credit 잔액·원래 만료일 확인 전 GPU run 금지
- backend 영향: Vertex API를 직접 호출하지 않으며 approved model package 인수 계약은
  기존대로 별도 협의

안건 설명 확인자: ____________________  전달 시각: ____________________

선택한 안: **A**  결정일: **2026-08-14**

상세 시나리오·계산식은
`docs/ml/backend-infrastructure-agreement.md`의 5-1을 기준으로 한다.

### 이번 회의에서 반드시 결정해야 하는 핵심 충돌

| 우선순위 | 문제 | 결정하지 않으면 생기는 일 |
| --- | --- | --- |
| 1 | `/router/classify` 필드명과 request 추적 방식 | Spring-FastAPI contract test를 만들 수 없음 |
| 2 | escalation 응답을 언제 `user_satisfied` 학습 라벨로 볼지 | 잘못된 feedback으로 재학습할 수 있음 |
| 3 | feedback inline schema와 trigger | `feedback-inline-v1`, 1건 이상, Spring trigger로 확정 |
| 4 | candidate 승인과 FastAPI 모델 전환 주체 | gate를 통과해도 운영 모델이 바뀌지 않음 |
| 5 | 확정한 관리형 Vertex AI Pipeline의 resource·IAM 값 | cloud smoke를 실행할 수 없음 |
| 6 | 선택 구조에 사용할 resource 이름·IAM 담당자 | cloud smoke를 실행할 수 없음 |
| 7 | MVP·개요와 현행 구현 중 공식 기준 | 보고서와 코드가 서로 다른 시스템을 설명하게 됨 |

### 회의 종료 시 확보해야 할 산출물

- 확정된 Router API 요청·응답·오류 fixture
- 실제 Claude mapping과 fallback을 담은 routing config 책임자
- feedback inline schema, 만족 판정, Spring trigger와 FastAPI artifact 생성 책임
- candidate→approved→serving→rollback 흐름과 담당자
- GCP project·region·bucket·registry, Vertex Pipeline/CustomJob resource·IAM·예산
- MVP·졸업프로젝트 개요의 수정 기준과 담당자
- 미결정 항목마다 실험·확인 담당자와 마감일

## 1. 진행 규칙과 완료 조건

### 상태 표기

| 표기 | 뜻 | 회의 처리 |
| --- | --- | --- |
| `일치` | 양쪽 문서가 같은 방향 | 30초 이내 확인하고 넘어감 |
| `부분 일치` | 방향은 같지만 schema·값이 부족함 | 빈 값만 결정 |
| `불일치` | 서로 다른 계약·정책을 제안함 | 선택지와 영향 비교 후 하나를 채택 |
| `미정` | 어느 쪽에도 결정이 없음 | 결정하거나 담당자·기한 지정 |

### 진행 원칙

1. 이 문서를 위에서 아래 순서로 진행한다.
2. 이미 일치한 방향은 재설계하지 않고 유지 여부만 확인한다.
3. 확정값과 예시값을 구분한다. 특히 Claude 제품명과 가격은 코드에 고정하지 않는다.
4. 결론을 못 내리면 `보류`로 끝내지 않고 확인 담당자, 필요한 근거와 기한을 적는다.
5. DB table, GCS bucket, namespace와 API 필드는 결정 기록 없이 ML core에 고정하지 않는다.
6. 회의 후 결정값을 contract fixture와 양쪽 테스트에 반영해야 완료다.
7. 2026-08-14 확정한 Vertex AI Pipelines CPU → Vertex AI Custom Training GPU 구조와
   GKE 미사용 원칙을 유지한다.

### 권장 시간 배분

| 구간 | 시간 |
| --- | ---: |
| 일치 방향 확인 | 5분 |
| API·장애 정책 | 20분 |
| feedback·재학습 | 30분 |
| 배포·인프라 | 25분 |
| 문서 정합성·담당자 지정 | 15분 |

## 2. 이미 일치하는 방향 확인

아래 항목은 설명을 반복하지 말고 `유지`, `수정`, `추가 검증` 중 하나로 기록한다.

| ID | 확인 항목 | 현재 일치 방향 | 회의 결과 |
| --- | --- | --- | --- |
| A-01 | label | `easy/medium/hard`, checkpoint는 `0/1/2` | ____________________ |
| A-02 | mapping 소유권 | Gateway 설정이 실제 Claude model ID를 결정 | ____________________ |
| A-03 | confidence | 항상 tier를 반환하고 confidence는 분석에만 사용 | ____________________ |
| A-04 | DB 연결 | FastAPI는 PostgreSQL/pgvector에 직접 연결하지 않음 | ____________________ |
| A-05 | cache 정책 | Gateway가 pgvector 조회·threshold·hit/miss 결정 | ____________________ |
| A-06 | 버전 로그 | request, Claude, router, embedding model version 기록 | ____________________ |
| A-07 | fallback 집계 | fallback 요청은 정상 router metric에서 제외 | ____________________ |
| A-08 | 대화 context | `conversationId`는 추적용이며 현재 Claude 입력에 history를 넣지 않음 | ____________________ |
| A-09 | cache 비용 | cache hit의 input/output token과 LLM 비용은 0으로 집계 | ____________________ |

확인 메모: ______________________________________________________________________

## 3. Gateway-FastAPI 계약

### 3-1. 전체 호출 순서

```text
POST /api/chat
  -> POST /router/embed
  -> Gateway pgvector 조회
  -> cache hit: 저장 응답 반환
  -> cache miss: POST /router/classify
  -> Gateway routing config로 실제 Claude model ID 선택
  -> Claude 호출·로그·cache 저장
```

`POST /api/chat`은 구현 완료 상태이므로 공개 API를 다시 설계하지 않는다. 주요 형식은
`message`, `conversationId` 요청과 `requestId`, `tier`, `modelId`, `cacheHit`, token,
latency 응답이다.

### 3-2. `/router/classify` 계약

> 핵심 문제: 팀원 초안은 `message/tier/modelVersion` camelCase이고, 기존 ML 제안은
> `question/label/model_version` snake_case다. 이 상태로는 양쪽 fixture가 다르다.

| 항목 | 팀원 초안 | 기존 ML 제안 | 권장안 |
| --- | --- | --- | --- |
| 요청 본문 | `message` | `question`, `request_id` | `requestId`, `message` |
| 결과 label | `tier` | `label` | `tier` |
| model 선택값 | `model` 참고값 | 없음 | optional hint 또는 제거 |
| confidence | `confidence` | `confidence` | 필수 |
| router version | `modelVersion` | `model_version` | `modelVersion` 필수 |
| 추론 latency | 없음 | `latency_ms` | `inferenceLatencyMs` 권장 |
| naming | camelCase | snake_case | Gateway와 맞춰 camelCase |

권장 fixture:

```json
{
  "requestId": "req-123",
  "message": "사용자 질문"
}
```

```json
{
  "requestId": "req-123",
  "tier": "medium",
  "confidence": 0.82,
  "modelVersion": "router-modernbert-v1",
  "inferenceLatencyMs": 24
}
```

결정할 것:

| ID | 질문 | 선택·기록 |
| --- | --- | --- |
| D-01 | 위 권장 fixture를 채택할 것인가 | ____________________ |
| D-02 | `model` 참고 필드를 유지할 것인가 | ____________________ |
| D-03 | `requestId`를 FastAPI까지 전달할 것인가 | ____________________ |
| D-04 | `inferenceLatencyMs`를 응답 필수값으로 둘 것인가 | ____________________ |

담당자 / 반영 기한: ____________________ / ____________________

### 3-3. `/router/embed` 책임과 규격

현재 방향은 FastAPI가 순수 embedding inference를 제공하고 Gateway가 cache 정책을
소유하는 것이다. 하지만 endpoint 구현·배포 담당자는 명시되지 않았다.

| ID | 결정할 사항 | 현재 제안 | 회의 기록 |
| --- | --- | --- | --- |
| D-05 | `/router/embed` 구현·운영 담당 | router-serving 담당자가 순수 inference 구현 | ____________________ |
| D-06 | vector 차원 | 1024 | ____________________ |
| D-07 | L2 normalization | FastAPI에서 정규화 후 반환 | ____________________ |
| D-08 | pooling | EXP-003 코드와 동일한 값을 확인 후 고정 | ____________________ |
| D-09 | bge prefix | 질문↔질문 대칭 비교이므로 양쪽 모두 붙이지 않음 | ____________________ |
| D-10 | embedding version 변경 | 기존 cache와 혼용하지 않고 version 분리 | ____________________ |

> 주의: pooling, normalization, prefix 중 하나라도 실험과 serving이 다르면 오류 없이
> cache 정확도만 떨어질 수 있다.

### 3-4. 오류와 health 계약

팀원 문서의 공통 오류 형식은 다음과 같다.

```json
{
  "error": "사용자용 메시지",
  "code": "ERROR_CODE",
  "requestId": "req-123"
}
```

| ID | 결정할 사항 | 권장안 | 회의 기록 |
| --- | --- | --- | --- |
| D-11 | FastAPI도 같은 오류 body를 반환할지 | 동일 형식 사용 | ____________________ |
| D-12 | router 전용 오류 code | `ROUTER_NOT_READY`, `ROUTER_INFERENCE_FAILED` 추가 검토 | ____________________ |
| D-13 | `/health` 503 조건 | embedding/router 중 하나라도 미적재 시 503 | ____________________ |
| D-14 | health에 실제 version 포함 | `embedModelVersion`, `routerModelVersion` 추가 권장 | ____________________ |

## 4. Claude mapping과 장애 처리

### 4-1. 실제 mapping

ML은 tier까지만 책임지고 Gateway 설정이 실제 model ID를 선택한다. 아래에는 계열명이
아니라 API 호출에 사용하는 정확한 model ID와 config version을 적는다.

| tier | 실제 model ID | fallback model ID | timeout | config version |
| --- | --- | --- | ---: | --- |
| easy | ____________________ | ____________________ | ______ | ____________________ |
| medium | ____________________ | ____________________ | ______ | ____________________ |
| hard | ____________________ | ____________________ | ______ | ____________________ |

mapping 변경 시 ModernBERT를 재학습하지 않으며, `routingPolicyVersion`을 요청 로그에
남긴다.

### 4-2. 장애 정책

| 실패 | 팀원 제안 | 기존 ML 제안 | 권장 판단 |
| --- | --- | --- | --- |
| embed 실패 | cache miss로 간주 | 온라인 팀 책임 | 팀원 제안 유지 |
| classify 실패 | `medium→sonnet` | 품질 보호 상위 tier | 데모 비용·안정성 기준 medium 권장 |
| unknown tier | 명시 없음 | 상위 fallback | classify 실패 정책과 통일 |
| model load 실패 | health 503 | 이전 approved 보존 | Gateway fallback + 운영 rollback 분리 |
| Claude 실패 | 사용자 오류 반환 | 미정 | 재시도 횟수·대체 model 결정 필요 |

| ID | 결정할 사항 | 회의 기록 |
| --- | --- | --- |
| D-15 | classify/unknown tier 기본값 | ____________________ |
| D-16 | embed/classify timeout 3초 유지 여부 | ____________________ |
| D-17 | Claude timeout·retry·fallback | ____________________ |
| D-18 | fallback log 필드와 metric 제외 규칙 | ____________________ |

## 5. 에스컬레이션과 만족 라벨

### 5-1. 가장 먼저 풀어야 할 의미 문제

```text
사용자가 /api/escalate 호출
  = 이전 답변에 불만족했다는 명시적 신호

상위 모델 답변을 반환
  ≠ 새 답변에 만족했다는 명시적 신호
```

> 현재 초안은 escalation 결과를 만족한 답으로 보고 cache를 갱신한다. 이를 그대로 ML
> 정답 라벨로 사용하면 상위 모델 답변에도 불만족한 사례가 잘못 학습될 수 있다.

### 5-2. 만족 판정 선택지

| 선택지 | 방식 | 장점 | 위험 |
| --- | --- | --- | --- |
| A | escalation API 성공을 곧바로 만족으로 간주 | 구현이 가장 단순 | label noise가 가장 큼 |
| B | 사용자 만족 버튼/이벤트를 별도로 수집 | 가장 명확한 신호 | UI·API 추가 필요 |
| C | 일정 시간 추가 escalation이 없으면 만족 추정 | UI 변경이 적음 | 이탈과 만족을 구분 못 함 |
| D | 반복 질문군의 다수 신호만 학습에 채택 | noise 완화 | 데이터 축적이 느림 |

ML 권장안은 `B`를 우선으로 하고, 데이터가 부족하면 `D`를 함께 적용하는 것이다.

| ID | 결정할 사항 | 회의 기록 |
| --- | --- | --- |
| D-19 | `user_satisfied`를 어떻게 수집·판정할 것인가 | ____________________ |
| D-20 | hard/최상위 모델에도 불만족하면 어떤 label/status로 저장할 것인가 | ____________________ |
| D-21 | cache 갱신 조건과 ML 학습 채택 조건을 분리할 것인가 | ____________________ |
| D-22 | 단발 escalation을 학습 후보로 보관하되 즉시 채택하지 않을 것인가 | ____________________ |

### 5-3. escalation log 최소 필드

모델명만 저장하면 mapping 변경 후 tier를 안전하게 복원할 수 없다. 다음을 직접 보존한다.

| 필드 | 목적 | 필수 여부 결정 |
| --- | --- | --- |
| `event_id`, `request_id`, `conversation_id` | 추적·중복 방지 | ____________________ |
| `question` | 학습 입력 | ____________________ |
| `initial_tier`, `final_tier` | 정정 label 근거 | ____________________ |
| `initial_model_id`, `final_model_id` | 실제 호출 증거 | ____________________ |
| `escalation_path`, `escalation_count` | 경로·반복 분석 | ____________________ |
| `user_satisfied` | 학습 채택 근거 | ____________________ |
| `router_model_version` | 재학습 전후 비교 | ____________________ |
| `routing_policy_version` | 당시 tier→model mapping 재현 | ____________________ |
| `created_at` | window·순서 | ____________________ |
| `pii_redacted` | 개인정보 검증 | ____________________ |

로그 schema 담당자 / 완료 기한: ____________________ / ____________________

## 6. Feedback export와 ML 입력

### 6-1. 전달 경계

```text
Gateway PostgreSQL
  -> Spring: PII 제거·event 선택
  -> FastAPI POST /escalate: feedback-inline-v1 JSON
  -> FastAPI: 검증·Parquet·manifest·checksum
  -> immutable GCS batch
  -> Vertex Pipeline CPU validator/normalizer
```

Vertex AI Pipeline과 FastAPI는 운영 DB를 직접 조회하지 않는다. Spring은 GCS를 다루지
않으며, FastAPI가 GCS batch를 완성한 뒤 Vertex에 제출한다.

### 6-2. manifest 계약

| 필드 | 제안 |
| --- | --- |
| 식별 | `schema_version`, `batch_id`, `producer_version` |
| 기간 | `created_at`, `window_start`, `window_end`, `watermark` |
| 데이터 | `event_count`, `data_format`, `data_uri`, `sha256` |

| ID | 결정할 사항 | 회의 기록 |
| --- | --- | --- |
| D-23 | manifest+Parquet 방식을 채택할 것인가 | ____________________ |
| D-24 | exporter 소유자와 실행 위치 | ____________________ |
| D-25 | GCS dump 주기 또는 batch window | ____________________ |
| D-26 | PII 제거 주체와 원문 retention | ____________________ |
| D-27 | schema version 변경 승인자 | ____________________ |

### 6-3. escalation 보증 범위

현재 ML prototype은 `pii_redacted=true`, `user_satisfied=true`,
`escalation_count>=1`, final tier의 엄격한 상승을 검사한다.

| 선택 | 의미 | 코드 영향 |
| --- | --- | --- |
| 온라인 export가 유효 escalation만 보증 | ML은 schema·품질·중복만 재검사 | `escalation_count`를 감사 metadata로 완화 가능 |
| raw escalation을 모두 export | ML이 비즈니스 채택 조건도 검사 | 현재 strict validator 유지 |

| ID | 결정할 사항 | 회의 기록 |
| --- | --- | --- |
| D-28 | export가 보증하는 최소 이벤트 조건 | ____________________ |
| D-29 | `escalation_count>=1`을 ML 필수 조건으로 유지할지 | ____________________ |
| D-30 | 부적합 행을 삭제할지 quarantine artifact에 보존할지 | ____________________ |

## 7. 재학습 trigger와 데이터 정책

### 7-1. N의 정확한 의미

N은 원본 DB 행 수가 아니라 다음 처리 후의 `accepted_count`로 정의하는 것을 권장한다.

```text
raw events
  -> schema/PII/satisfaction 검사
  -> normalized question dedup
  -> label conflict 제거
  -> holdout overlap 제거
  -> accepted_count
```

| ID | 결정할 사항 | 회의 기록 |
| --- | --- | --- |
| D-31 | 전체 accepted N | ____________________ |
| D-32 | easy/medium/hard class별 최소량 | ____________________ |
| D-33 | N 미달이어도 실행할 최대 대기 기간 | ____________________ |
| D-34 | 수동 승인, scheduler, event trigger 중 실행 방식 | ____________________ |
| D-35 | trigger와 pipeline submit 담당 | ____________________ |
| D-36 | 같은 batch 재실행 idempotency key | ____________________ |

초기 권장안은 수동 승인으로 시작하고, 1~2회 실제 batch의 채택률과 GPU 비용을 확인한 뒤
자동 trigger를 활성화하는 것이다.

### 7-2. 학습 데이터 혼합

| 항목 | 현재 ML prototype | 확정값 |
| --- | --- | --- |
| 반복 질문 weight | `1.0 + 0.25×반복`, 최대 2.0 | ____________________ |
| 동일 질문·동일 label | 한 행으로 집계 | ____________________ |
| 동일 질문·상충 label | 자동 채택 제외 | ____________________ |
| validation/test overlap | feedback 제외 | ____________________ |
| 기존 train 충돌 | versioned 정책으로 교체·집계 | ____________________ |
| RouterBench:feedback 혼합 비율 | 미정 | ____________________ |

## 8. 모델 package, gate와 실제 배포

### 8-1. 책임 경계

```text
Kubeflow
train -> evaluate -> package -> gate
                                |
                                +-> rejected
                                +-> candidate
                                      |
운영                                  +-> approved 승인
                                      +-> FastAPI 적용
                                      +-> health 확인
                                      +-> rollback
```

> deploy gate는 metric을 판정할 뿐 현재 FastAPI 프로세스를 직접 바꾸지 않는다.

### 8-2. 결정할 사항

| ID | 결정할 사항 | 권장안 | 회의 기록 |
| --- | --- | --- | --- |
| D-37 | gate metric과 허용 하락폭 | full baseline 후 versioned JSON | ____________________ |
| D-38 | candidate→approved 승인자 | 초기에는 사람 승인 | ____________________ |
| D-39 | GCS candidate/approved/rejected prefix | immutable model version 디렉터리 | ____________________ |
| D-40 | FastAPI가 approved version을 받는 방식 | 명시적 deployment parameter | ____________________ |
| D-41 | hot reload와 rolling restart | 초기에는 rolling restart | ____________________ |
| D-42 | startup/readiness 검증 | checksum, label order, model version 확인 | ____________________ |
| D-43 | rollback 기준과 실행자 | 직전 approved version 보존 | ____________________ |
| D-44 | 승인·전환 감사 로그 | 승인자·시각·이전/신규 version 기록 | ____________________ |

### 8-3. package 인수인계 계약

온라인 serving이 최소한 다음을 검증할지 결정한다.

| 항목 | 목적 | 채택 여부 |
| --- | --- | --- |
| model/tokenizer 파일 목록 | 누락 방지 | ____________________ |
| SHA-256 checksum | 손상·잘못된 artifact 차단 | ____________________ |
| `easy=0, medium=1, hard=2` | label 뒤바뀜 방지 | ____________________ |
| model/data/code/evaluation version | 재현·감사 | ____________________ |
| gate 결과와 baseline delta | 승인 근거 | ____________________ |

## 9. GCP-GCS-Vertex AI 인프라

최근 확인 상태는 project 접근만 가능하고 GCS bucket, Artifact Registry, Vertex Pipeline
resource는 생성되지 않은 상태였다. 공식 feedback 재학습 backend는 안 A로 확정했으며,
아래 B/C는 결정 근거를 위한 역사적 비교다.

### 9-1. 재학습 실행 방식과 비용 선택

공통 가정은 `us-central1`(2026-08-12 비교 시점 기준, 실제 운영 region은 2026-08-15
`us-west1`로 변경됨. 아래 3안 비교표는 선택 근거 보존용 역사적 수치라 재계산하지
않는다), 비용 비교용 inline feedback 1,000건 예시, CPU 2시간, T4 1장 6시간, 전체
8시간, GCS 10 GiB/30일이다. 1,000건은 HTTP 계약의 필수 건수가 아니다.
Spot·CUD와 온라인 serving 비용은 제외했다. 상세 단가·계산식과 유휴비는
`docs/ml/backend-infrastructure-agreement.md` 5-1에서 확인한다.

| 안 | 재학습 시나리오 | 1회 추정 | 재학습 없는 달 | 핵심 판단 |
| --- | --- | ---: | ---: | --- |
| A | KFP SDK → Vertex 관리형 Pipeline CPU → Vertex T4 → 관리형 CPU 평가 | `$3.64~4.22` | GCS `$0.10~0.20` | 가장 저렴·간단, Kubeflow 공식 UI 없음 |
| B | GKE KFP CPU `0→1` → Vertex T4 → GKE CPU 평가 → `1→0` | `$5.21~5.79` | KFP disk 약 `$0.80` + GCS | 공식 UI 사용, 기존 기준안, 외부 wake-up 필요 |
| C | GKE KFP CPU `0→1` → GKE T4 node `0→1` → 평가 → 모두 `0` | `$5.21~5.31` | KFP disk 약 `$0.80` + GCS | GPU node·driver·cleanup 직접 운영 |

| ID | 결정할 사항 | ML 권고 | 회의 기록 |
| --- | --- | --- | --- |
| D-45 | 과제의 pipeline 요건 | KFP SDK + 관리형 Vertex backend | **안 A 확정(2026-08-14)** |
| D-46 | 공식 feedback 재학습 backend | Vertex AI Pipelines | **안 A 확정(2026-08-14)** |
| D-46A | B/C의 외부 wake-up·node cleanup | 해당 없음 | **GKE 미사용** |
| D-46B | 1회 hard budget과 실행 중단 기준 | GPU 6시간, retry 0, credit 확인 후 실행 | ____________________ |

안 B/C에서 CPU node를 한 달 상시 유지하면 약 `$97.82/month`가 추가된다. 안 C에서 GPU
node까지 한 달 유지하면 약 `$394.20/month`가 추가되므로 두 node pool의 scale-to-zero와
실패 cleanup은 운영 조건이다. 안 B/C는 역사적 비교로만 보존하며 구현하지 않는다.

### 9-2. 선택 후 채울 resource와 운영값

아래에서 project·region·비용 guardrail과 안 A는 확정값이다.

| ID | 항목 | 결정·실측값 | 담당자 | 기한 |
| --- | --- | --- | --- | --- |
| I-01 | project | **확정:** `project-763753da-fa8f-4afd-b31` (`HYUCapstone`) | ML | 완료 |
| I-02 | region | **확정:** `us-west1`(2026-08-15, `us-central1`에서 변경) | ML | 완료 |
| I-03 | 실행 backend | **확정:** Vertex AI Pipelines → Vertex T4 CustomJob | ML | 완료 |
| I-04 | GKE | **사용하지 않음** | ML | 완료 |
| I-05 | Vertex 제한 | **확정:** T4 1, replica/concurrency 1, retry 0, timeout 6h | ML | 완료 |
| I-06 | credit 잔액·원래 만료일 | ____________________ | billing admin | ______ |
| D-47 | GCS bucket와 prefix | ____________________ | ______ | ______ |
| D-48 | Artifact Registry repository | ____________________ | ______ | ______ |
| D-49 | Vertex pipeline root | ____________________ | ______ | ______ |
| D-50 | KFP/GCPC version·Vertex ML Metadata 보존 | `kfp 2.17.0`, `GCPC 2.22.0` | ML | 완료 |
| D-52 | PipelineJob/CustomJob service account | ____________________ | ______ | ______ |
| D-53 | Vertex submit·image pull·GCS·승격 IAM | ____________________ | ______ | ______ |
| D-54 | secret 저장·주입 방식 | ____________________ | ______ | ______ |
| D-55 | budget alert·Spend Cap 가능 여부·T4 quota=1 | ____________________ | ______ | ______ |
| D-56 | 첫 cloud smoke 데이터·비용 한도 | ____________________ | ______ | ______ |

RouterArena full 평가는 CPU에서 실행한다. 컴파일 IR에는 Kubernetes GPU request가 없고
학습 CustomJob worker pool에만 `NVIDIA_TESLA_T4=1`이 있어야 한다.

## 10. MVP-졸업프로젝트 개요 정합성

앞 절의 결정을 기준으로 기존 문서를 정정한다. 아래 표에서 `공식 결론`과 문서 담당자를
기록한다.

| 충돌 | 현행 ML 권장 기준 | 결정 근거 절 | 공식 결론·담당자 |
| --- | --- | --- | --- |
| RouterBench 대표 3개 vs 11개 모델 | 11개 모델 tier-consensus | 2 | ____________________ |
| 초기 라벨링·전처리·학습 위치 | Colab(공통 core + Drive checkpoint), 초기 Vertex job 미제출 | 9 | **2026-09-16 보정** |
| RouterArena 평가 자원 | Vertex AI Pipelines CPU component | 9 | **안 A 확정** |
| Haiku/Sonnet/Opus 고정 mapping | Gateway versioned config | 4 | ____________________ |
| 만족 tier를 곧바로 정답 label로 사용 | 만족·중복·충돌 검증 후 채택 | 5 | ____________________ |
| ML이 escalation을 재검사 | upstream 보증 범위에 따라 조정 | 6 | ____________________ |
| N건이면 자동 재학습 | accepted N·분포·기간·주체 명시 | 7 | ____________________ |
| Pipeline과 GPU 실행 위치 | Vertex managed CPU + Vertex T4 CustomJob | 9 | **안 A 확정** |
| gate 통과 즉시 자동 배포 | candidate·승인·serving 전환 분리 | 8 | ____________________ |
| `LIGHT/MEDIUM/HEAVY` 사용 | 경계는 `easy/medium/hard` | 2, 3 | ____________________ |
| 공개 API와 Router API 필드 | 계층별 schema와 adapter 명시 | 3 | ____________________ |
| QQP를 ML DAG에서 수행 | 온라인 cache 팀 산출물로 결합 | 2, 3 | ____________________ |
| RouterBench RAG와 service RAG 혼용 | eval-family와 서비스 기능을 분리 | 문서 정정 | ____________________ |
| cloud가 준비된 것처럼 기술 | architecture 확정과 provision 상태를 분리 | 통보, 9 | 문서에 반영 |
| 가격·모델명·일정 고정값 | 기준일·가정 표시 후 재산정 | 4, 9 | ____________________ |

문서 수정 순서:

1. `docs/졸업프로젝트 개요.md`에 `확정/계획/미구현` 상태를 표시한다.
2. `docs/mvp-scope-and-checklist.md`는 역사적 범위를 유지하고 현행 계약 링크를 단다.
3. `docs/ml/backend-infrastructure-agreement.md`에 결정자·결정일을 기록한다.
4. API·feedback fixture와 config를 갱신한다.
5. `docs/ml/workstream-checklist.md`에서 합의 항목을 완료 처리한다.

## 11. 최종 결정표와 후속 작업

### 11-1. 회의 종료 확인

| 영역 | 최종 산출물 | 상태·위치 |
| --- | --- | --- |
| Router API | 요청·응답·오류 JSON fixture | ____________________ |
| Routing policy | tier별 model ID, fallback, timeout config | ____________________ |
| Feedback | DB log 및 manifest/events schema | ____________________ |
| Retraining | N, 분포, 기간, trigger·submit 주체 | ____________________ |
| Promotion | gate, 승인, serving 전환, rollback runbook | ____________________ |
| Infrastructure | 안 A resource 이름, IAM, cleanup, credit·guardrail | ____________________ |
| Documents | MVP·개요 충돌별 공식 기준과 수정 담당 | ____________________ |

### 11-2. 미결정 항목

| 결정 ID | 필요한 추가 근거 | 확인 담당자 | 마감일 | 다음 결정일 |
| --- | --- | --- | --- | --- |
| ______ | ____________________ | ______ | ______ | ______ |
| ______ | ____________________ | ______ | ______ | ______ |
| ______ | ____________________ | ______ | ______ | ______ |

### 11-3. 후속 작업

| 순서 | 작업 | 담당자 | 기한 | PR/Issue |
| ---: | --- | --- | --- | --- |
| 1 | 회의 결정문과 schema version 반영 | ______ | ______ | ______ |
| 2 | Gateway-FastAPI contract fixture/test | ______ | ______ | ______ |
| 3 | Spring inline DTO/FastAPI converter/ML validator fixture 정합화 | ______ | ______ | ______ |
| 4 | 안 A에 필요한 GCS·Registry·Vertex Pipeline 설정과 cloud smoke | ______ | ______ | ______ |
| 5 | model 승인·serving 전환·rollback smoke | ______ | ______ | ______ |
| 6 | MVP·졸업프로젝트 개요 정정 | ______ | ______ | ______ |

회의 최종 메모:

________________________________________________________________________________

________________________________________________________________________________

________________________________________________________________________________
