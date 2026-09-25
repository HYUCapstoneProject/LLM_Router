# Spring 공통 기반 변경과 담당자 인계

이 변경은 R0/R1 브랜치가 함께 사용할 Spring 기반이다. 기본 분류 계약은 기존 `v1`, 자동 재학습 제출은 `false`를 유지한다. 8192는 FastAPI 분류기의 토큰화 설정이며 Spring에서 문자 수로 대신 자르지 않는다. `ROUTER_READ_TIMEOUT=PT15S`, 제출 timeout `PT60S`는 유지했다. 실제 서버의 8192 지연·메모리·동시 요청 측정 후 timeout과 운영 활성화 여부를 별도로 결정한다.

## 변경 파일과 책임

| 영역 | 파일 | 변경 |
|---|---|---|
| 분류 계약 | `backend/src/main/java/com/llmrouter/client/FastApiRouterClient.java`, `RouterClassification.java` | `/router/classify/v2` 선택, policy SHA·실제 후보 모델 ID·점수 의미·action/tier/model 검증. R0 ordinal과 R1 pointwise 모두 지원 |
| 요청 관측 | `dto/RequestLogEntry.java`, `service/ChatService.java`, `service/RequestLogService.java` | 요청 시점의 정책·후보군·점수 provenance를 nullable `routing_decision`에 기록. v1·cache·fallback은 기존 INSERT 유지 |
| 배치 제출 | `repository/FeedbackBatchRepository.java`, `service/FeedbackBatchService.java`, `dto/FeedbackBatchRequest.java` | row lock으로 이벤트를 고르고 UUID 배치 ID 및 요청 전체 snapshot을 같은 transaction에서 저장. 재시도는 저장된 requestedAt·events 재사용 |
| 원격 계약 | `client/FastApiRetrainingClient.java` | HTTP 202·schema·batchId·statusUrl 확인. 상태 응답의 schema·batchId·enum·terminal 및 활성 모델 일치 검증 |
| 상태 관찰 | `dto/RetrainingStatus.java`, `service/RetrainingScheduler.java` | pipeline terminal과 polling 완료 분리. stale 또는 activation pending은 계속 관찰. loadFailed는 명시적 실패 기록 후 종료 |
| 설정 | `backend/src/main/resources/application.yml` | v2 선택과 정책·후보군 pin 환경 변수 추가. 기본값 v1 유지 |
| DB | `infra/postgres/init/01-schema.sql`, migrations `09-routing-decision.sql`, `10-feedback-batch-snapshot.sql` | 새 DB 및 기존 DB의 additive schema 변경 |

테스트는 위 영역의 client/service/DTO 테스트와 `FeedbackBatchPostgresTest`, 두 저장소 공유 `routing-v2-cross-repo.json` fixture를 사용한다.

## API 호환과 데이터 의미

공개 Spring API와 `/escalate`의 `feedback-inline-v1` 요청 형식은 유지된다. batchId는 `feedback-YYYYMMDD-UUID`로 바뀌지만 기존 영문·숫자·점·밑줄·하이픈, 길이 200 이하 계약을 지킨다. ID의 마지막 부분을 숫자로 파싱하는 외부 도구가 있다면 문자열로 처리해야 한다.

v2를 켜려면 `ROUTER_CLASSIFY_CONTRACT=v2`, `ROUTER_EXPECTED_POLICY_SHA256`, `ROUTER_CANDIDATE_POOL_VERSION`과 실제 `llm.models.easy/medium/hard`가 배포 정책과 일치해야 한다. v2 오류를 v1 분류로 재호출하지 않으며 기존 ChatService의 서비스 fallback 동작을 따른다. 기록된 provenance가 없는 cache·fallback 요청은 R1 학습 관측 근거로 쓰지 않는다.

`userSatisfied=true`는 기존 R0의 체인 종료를 추정하는 약한 신호다. 명시적인 사용자 만족이나 각 후보의 성공 판정으로 바꾸지 않았다. R1 outcome DTO·평가 경계·수집/outbox·FastAPI 전송·masked BCE 학습 연결은 R1 브랜치 범위이며 공통 `/escalate`에 연결하지 않는다.

## DB 적용 순서와 복구

1. 기존 `08-feedback-batch.sql`까지 적용된 DB에서 기존 제출·polling 상태를 확인한다. 배포 중 이전 Spring 인스턴스가 새 제출을 만들지 않도록 제출 scheduler와 수동 제출을 중지한다.
2. `09-routing-decision.sql`, `10-feedback-batch-snapshot.sql`을 순서대로 적용한다. 새 빈 볼륨은 갱신된 `init/01-schema.sql`을 사용한다.
3. 공통 Spring 코드를 배포한다. v1 요청은 routing_decision에 의존하지 않지만 새 batch repository는 migration 10이 필수다.
4. 기존 snapshot 없는 미접수 배치는 `unknown / LEGACY_PAYLOAD_UNAVAILABLE`로 격리된다. 원래 이벤트 소유는 보존된다. FastAPI의 기존 batchId·pipeline/GCS 상태와 최초 payload를 대조하여 담당자가 복구한다. 새 payload 또는 새 batchId로 자동 재제출하지 않는다.
5. 기존 접수된 배치는 새 `polling_complete=false` 기본값으로 다시 조회되어 stale/활성화 상태를 재확인한다. 성공한 학습도 activation pending이면 완료 시각을 기록하지 않는다. loadFailed는 `MODEL_ACTIVATION_FAILED`와 종료 시각을 남긴다.

migration은 반복 실행 가능하다. 코드 rollback 시 additive 컬럼은 남겨도 된다. 다만 예전 코드는 immutable snapshot을 사용하지 않으므로 제출을 중지한 상태에서 rollback하고 배치를 대조해야 한다. snapshot을 삭제하거나 격리 이벤트를 임의로 대기 상태로 풀지 않는다.

동일 batchId 충돌, 인증 오류, 형식이 잘못된 접수 응답은 접수 여부가 불명확하므로 격리한다. 명확한 `422 REQUEST_VALIDATION_FAILED`만 이벤트를 해제한다. 네트워크 timeout과 계약상 retryable 오류는 같은 snapshot으로 재시도한다. PII guard는 최초 제출과 재시도 모두에 적용된다. 저장 snapshot은 기존 DB에 이미 있는 질문의 복사본이므로 DB 접근·보존 정책에 함께 포함해야 한다.

## 검증 방법

`backend`에서 Java 24로 `bash gradlew test`를 실행한다. 실제 PostgreSQL 검증은 운영 DB와 다른 임시 DB의 `TEST_POSTGRES_URL`, `TEST_POSTGRES_USER`, `TEST_POSTGRES_PASSWORD`를 지정하면 활성화된다. 해당 테스트는 무작위 `feedback_test_*` 스키마만 만들고 제거한다. 동시 transaction의 중복 없는 claim, rollback의 원자성, 원본 이벤트 수정 후 snapshot 불변성, stale/pending/loadFailed polling, 기존 snapshot 없는 batch migration을 검증한다. 이 환경 변수가 없으면 DB 통합 테스트가 skip되므로 단위 테스트 통과와 구분해 보고해야 한다.

Spring 담당자는 배포 전에 migration 순서, ID 문자열 취급, v1/v2 환경 설정, 실제 모델 ID와 정책 SHA 일치, 운영 8192 성능 측정 결과, legacy 격리 배치 복구를 확인한다. 이 변경 자체는 서버 배포나 GCP/GCS 변경을 실행하지 않는다.

## 이번 변경의 검증 기록

2026-09-26 Java 24.0.2, Gradle 9.0.0에서 전체 suite 173개 중 단위·계약 테스트 169개 통과, DB 환경 미설정 상태의 통합 테스트 4개 skip을 확인했다. 이어 별도 로컬 PostgreSQL의 임시 스키마에서 `FeedbackBatchPostgresTest` 4개를 실행해 모두 통과했고, 독립 리뷰 후 늦게 도착한 제출 실패·중복 202의 CAS 회귀 검증을 추가해 최종 5개 모두 통과했다(실패·skip 0). 단위·계약 169개와 최종 DB 통합 5개를 합쳐 174개 테스트를 검증했다. 운영 DB migration, 서버 배포 및 8192 운영 latency gate는 실행하지 않았다.
