# 온라인 시스템 ↔ ML 파이프라인 전달 계약

상태: **Draft v0.1 — 팀 합의 전 제안안**

기준일: 2026-08-10

대상: Spring/API·Gateway·DB 담당자와 ML/Kubeflow 담당자

## 1. 목적

이 문서는 온라인 서비스와 오프라인 ML 파이프라인 사이에서 서로 전달해야 하는
데이터와 책임을 정의한다. 현재 Kubeflow 서버와 최종 DB schema가 확정되지 않았으므로,
변경 가능성이 큰 DB 컬럼과 변경 가능성이 낮은 ML 내부 schema를 adapter로 분리한다.

이 문서에서 `제안`으로 표시된 항목은 코드에 영구 고정하기 전에 팀 합의가 필요하다.
합의가 끝나면 schema version을 올리고 example fixture와 contract test를 함께 갱신한다.

ML 저장소에는 이 제안안의 실행 가능성을 검증하기 위한 `feedback-export-v0.1` adapter와
합성 fixture가 구현돼 있다. 이는 온라인 팀과 합의된 운영 schema가 아니라 변경 가능한
contract prototype이다.

## 2. 책임 경계

| 영역 | 온라인 팀 책임 | ML 팀 책임 |
| --- | --- | --- |
| 에스컬레이션 수집 | 질문, 최초 route, escalation 경로, 최종 만족 결과를 DB에 저장 | 저장된 event의 의미와 학습 채택 조건 검토 |
| 개인정보 | 수집 동의, 민감정보 제거/차단, retention 적용 | 전달된 데이터 재검사와 부적합 행 격리 |
| 배치 export | 중복 없이 재실행 가능한 batch와 manifest 생성 | checksum/schema 검증 후 canonical dataset 생성 |
| 재학습 trigger | 누적량 집계와 호출 주체 합의 | pipeline parameter와 idempotency 검사, 실행 결과 제공 |
| 모델 전달 | 승인 manifest를 읽고 새 모델 적용·rollback | checkpoint 평가, gate, package와 manifest 게시 |
| 온라인 추론 | timeout/fallback, 실제 Claude model ID와 route 정책 | `easy/medium/hard` 예측과 confidence 제공 |
| cache threshold | QQP 평가, bge/pgvector 조회와 운영 적용 | ML 학습/Kubeflow 범위에서 제외 |

ML 파이프라인은 PostgreSQL table을 직접 조회하지 않는다. 온라인 팀은 파일 batch를
export하고, ML 팀은 해당 version의 adapter를 통해 내부 표준 schema로 변환한다.

## 3. 온라인 → ML: 에스컬레이션 배치

### 3-1. 전달 단위

제안 형식은 다음 두 파일이다.

```text
feedback/<batch-id>/manifest.json
feedback/<batch-id>/events.parquet
```

개발용 fixture는 JSONL을 허용할 수 있지만 운영 기본 포맷은 Parquet을 제안한다. 질문
본문이나 사용자 답변을 manifest에 넣지 않는다.

### 3-2. 배치 manifest schema 제안

| 필드 | 타입 | 필수 | 의미 |
| --- | --- | --- | --- |
| `schema_version` | string | Y | 예: `feedback-export-v0.1` |
| `batch_id` | string | Y | 재실행해도 변하지 않는 배치 식별자 |
| `created_at` | UTC datetime | Y | export 생성 시각 |
| `window_start` | UTC datetime | Y | 포함 event 시간 범위 시작 |
| `window_end` | UTC datetime | Y | 포함 event 시간 범위 끝 |
| `event_count` | integer | Y | 데이터 파일의 행 수 |
| `data_format` | enum | Y | `parquet` 또는 개발용 `jsonl` |
| `data_uri` | string | Y | object storage의 데이터 위치 |
| `sha256` | string | Y | 데이터 파일 무결성 checksum |
| `producer_version` | string | Y | exporter 코드/version |
| `watermark` | string | Y | 다음 증분 export가 이어받을 기준 |

예시:

```json
{
  "schema_version": "feedback-export-v0.1",
  "batch_id": "feedback-20260808-001",
  "created_at": "2026-08-08T12:00:00Z",
  "window_start": "2026-08-01T00:00:00Z",
  "window_end": "2026-08-08T00:00:00Z",
  "event_count": 1000,
  "data_format": "parquet",
  "data_uri": "gs://<bucket>/feedback/feedback-20260808-001/events.parquet",
  "sha256": "<sha256>",
  "producer_version": "feedback-exporter-v0.1",
  "watermark": "<opaque-next-cursor>"
}
```

### 3-3. event row schema 제안

| 필드 | 타입 | 필수 | 검증 규칙 |
| --- | --- | --- | --- |
| `event_id` | string | Y | 전체 export에서 유일해야 한다. |
| `request_id` | string | Y | 온라인 요청 추적 식별자다. |
| `conversation_id` | string | N | 원본 ID 대신 비식별 ID를 권장한다. |
| `question` | string | Y | trim 후 비어 있으면 안 되며 학습 입력 후보가 된다. |
| `initial_label` | enum | Y | `easy`, `medium`, `hard` 중 하나다. |
| `initial_model` | string | Y | 당시 실제 호출한 model identifier다. |
| `final_label` | enum | Y | 최종 만족 모델에 대응하는 난이도다. |
| `final_model` | string | Y | 최종 만족 응답을 만든 model identifier다. |
| `user_satisfied` | boolean | Y | `true`인 event만 자동 학습 후보로 제안한다. |
| `escalation_count` | integer | Y | 0 이상이며 정정 라벨 후보는 1 이상이어야 한다. |
| `created_at` | UTC datetime | Y | event 발생 시각이다. |
| `router_model_version` | string | Y | 최초 label을 예측한 router version이다. |
| `pii_redacted` | boolean | Y | 온라인 단계의 비식별 처리 여부다. |

응답 본문, 벡터 embedding, API key, 사용자 계정 식별자는 ML 학습 event의 필수 필드가
아니다. cache 갱신을 위해 필요한 응답 데이터는 온라인 저장 계약에서 별도로 관리한다.

### 3-4. ML 수신 검증과 채택 규칙 제안

ML adapter는 다음 순서로 처리한다.

1. manifest version, 행 수와 SHA-256을 확인한다.
2. `batch_id`와 `event_id`를 기준으로 중복 실행·중복 행을 제거한다.
3. 필수 필드, enum, UTC 시간, 빈 질문과 `pii_redacted`를 검사한다.
4. `easy < medium < hard` 순서에서 `final_label`이 `initial_label`보다 반드시 높은
   event만 v0.1 자동 정정 라벨 후보로 둔다. 같거나 낮으면 metadata에 격리한다.
5. `user_satisfied=true`, `escalation_count>=1`인 event를 기본 학습 후보로 둔다.
6. 동일·유사 질문에서 final label이 충돌하면 confidence 규칙이 확정될 때까지 검토
   queue로 보낸다.
7. 채택된 행을 ML 내부 학습 표준 schema로 변환하고 RouterBench와 group-aware하게
   병합한다.

최종 만족 tier를 항상 정답으로 간주할지, 반복 질문이 몇 건 이상일 때 채택할지,
feedback sample weight를 얼마로 둘지는 아직 합의·실험이 필요한 항목이다.

현재 prototype은 같은 정규화 질문과 같은 final label이 반복되면 한 행으로 집계하고,
기본 1.0에서 반복 1건당 0.25를 더해 최대 2.0으로 제한한다. 이는
`feedback_retraining_preprocessing.json`의 변경 가능한 실험 기본값이며 운영 합의가
아니다. 동일 질문에 서로 다른 final label이 있으면 모두 자동 학습에서 제외한다.

## 4. ML 내부 canonical schema

외부 export가 바뀌어도 학습기에는 다음 schema만 전달한다.

| 필드 | 타입 | 의미 |
| --- | --- | --- |
| `sample_id` | string | 출처와 version을 포함해 재현 가능한 식별자 |
| `model_input_text` | string | 정답·응답을 제외한 질문 입력 |
| `difficulty_label` | enum | `easy`, `medium`, `hard` |
| `sample_weight` | float | 라벨 신뢰도와 source 정책을 반영한 양수 |
| `split` | enum | `train`, `validation`, `test` |
| `duplicate_group_id` | string | 유사 질문의 split 누수를 막는 group |

feedback adapter는 별도 metadata artifact에 `batch_id`, `event_id`, 원래 label,
채택/제외 사유와 adapter version을 남긴다. 개인정보가 될 수 있는 온라인 식별자는
training projection에 포함하지 않는다.

## 5. ML → 온라인: 승인 모델 전달

ML 팀은 checkpoint 경로만 전달하지 않고 다음 파일을 하나의 승인 package로 제공한다.

```text
approved/<model-version>/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model weights
├── metrics.json
├── training_config.json
└── model_manifest.json
```

manifest 최소 필드 제안:

```text
model_version, base_model, files
dataset_version, preprocessing_version, label_rule_version
id2label, tokenizer_max_length
training_version, evaluation_version, gate_status, created_at, code_commit
```

필수 계약은 `id2label={0: easy, 1: medium, 2: hard}`이며, `gate_status=approved`와
`files`의 파일별 SHA-256 검증에 성공한 package만 온라인에 적용한다. 현재
`model-package-v1` 생성기·검증기는 ML 저장소에 구현돼 있다. 온라인 팀은 새 모델 로딩
실패 시 이전 승인 version으로 되돌릴 수 있어야 한다.

## 6. 온라인 Router API 계약 제안

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

다음 운영 정책은 팀 합의가 필요하다.

- Spring의 `LIGHT/MEDIUM/HEAVY`와 ML의 `easy/medium/hard` 중 공통 명칭
- `easy→Haiku`, `medium→Sonnet`, `hard→Opus`의 실제 Claude model identifier
- router timeout, 연결 실패, unknown label 시 기본 route
- model version 전환과 rollback 방법
- 동일 `request_id` 재시도의 idempotency 정책

## 7. 재학습 trigger 계약 제안

온라인 팀과 다음 두 방식 중 하나를 선택한다.

- 온라인/scheduler가 신규 유효 event 수를 집계하고 `batch_id`와 함께 pipeline을 호출
- ML scheduler가 준비 완료된 manifest를 주기적으로 탐색하고 idempotently 실행

두 방식 모두 다음 값을 실행 기록에 남겨야 한다.

```text
trigger_id, batch_id, schema_version, event_count
base_dataset_version, current_model_version, requested_at
```

누적 임계값 `N`, 최대 재학습 주기, 수동 승인 여부는 실험값이며 현재 확정하지 않는다.
이미 처리한 `batch_id`를 다시 받아도 동일 데이터가 중복 학습되지 않아야 한다.

## 8. cache threshold 역할 경계

온라인 서빙·캐시 팀이 QQP 평가와 versioned threshold artifact를 생성하고
Spring/pgvector에 적용한다. ML 팀은 이 artifact를 학습 데이터나 Kubeflow
component 입력으로 소비하지 않는다. 최종 프로젝트 보고를 위해 공유할 때는
다음 필드를 권장한다.

```text
embedding_model, embedding_dimension, normalization
similarity_metric, threshold, evaluation_dataset_version
precision, recall, false_hit_rate, created_at, code_commit
```

온라인 팀은 bge-large-en-v1.5의 1024차원, normalization과 cosine similarity 설정이
평가 환경과 동일한지 해당 팀의 contract test로 확인한다.

## 9. 팀 회의에서 확정할 체크리스트

- [ ] 대상 Google Cloud project가 `project-763753da-fa8f-4afd-b31`인지 재확인한다.
- [ ] region, GCS bucket, Artifact Registry repository와 GKE cluster 이름을 정한다.
- [ ] 예산·quota, CPU/GPU node pool, namespace와 Workload Identity 계정을 정한다.
- [ ] 운영 export 포맷과 object storage 위치를 정한다.
- [ ] manifest와 event row의 필수/선택 필드를 확정한다.
- [ ] UTC timestamp, ID 생성과 export watermark 규칙을 확정한다.
- [ ] 개인정보 제거 주체, 검증 기준과 retention 기간을 정한다.
- [ ] 최종 만족 tier의 학습 채택 조건과 충돌 처리 규칙을 정한다.
- [ ] feedback sample weight와 RouterBench 혼합 비율의 결정 주체를 정한다.
- [ ] 재학습 trigger 주체, 임계값과 수동 승인 단계를 정한다.
- [ ] 공통 label 명칭과 Claude model mapping을 정한다.
- [ ] Router API timeout/fallback/rollback 계약을 정한다.
- [ ] model·threshold artifact의 위치, 권한과 version 전환 방식을 정한다.
- [ ] QQP threshold는 온라인 서빙·캐시 팀 범위이며 ML/Kubeflow DAG에 넣지 않음을
  재확인한다.
- [ ] 양 팀이 공유할 작은 contract fixture와 통합 smoke 명령을 확정한다.

## 10. 변경 관리

- 호환되지 않는 필드 의미·타입 변경은 `schema_version`을 올린다.
- 기존 version의 adapter는 이미 생성된 batch 재현을 위해 삭제하지 않는다.
- schema 변경 PR에는 manifest/event fixture와 양쪽 contract test 변경을 포함한다.
- 합의되지 않은 DB table 이름, GCS bucket, project, namespace, service account와 실제
  model identifier를 코드 기본값으로 고정하지 않는다.
