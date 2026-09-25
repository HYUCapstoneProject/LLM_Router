# R1 Spring 구현과 담당자 인계

이 브랜치는 기존 HTTP 경로와 요청·응답 DTO를 유지한다. `r1` profile에서도 FastAPI `/router/classify` v1을 호출하며 `/router/classify/v2`로 바꾸지 않는다. Spring은 R1 점수 벡터나 정책 해시가 v1 응답에 있었다고 가정하지 않는다. FastAPI의 R1 adapter가 기존 tier/model/confidence/modelVersion 응답으로 투영하며, 실제 호출 모델과 반환 modelVersion은 기존 request_log에 기록된다.

## 변경 범위

| 파일 | 역할 |
|---|---|
| `backend/src/main/resources/application-r1.yml` | v1 classify 유지, legacy CE 제출·자동 제출 비활성화, 내부 관측 수집 기본 false |
| `service/FeedbackBatchService.java`, `service/RetrainingScheduler.java` | R1 profile에서는 수동·재시도·자동 legacy 제출 차단. 기존 접수 배치 상태 조회는 유지 |
| `dto/R1Observation.java` | `r1-observation-v1` 단일 실제 후보 관측과 `r1-observation-batch-v1` artifact envelope. 명시적 Boolean만 허용 |
| `r1/VerifiedR1PolicyBinding.java` | 신뢰된 배포 기록에서 받은 manifest SHA pin → 승인 manifest → policy file SHA 순서 검증. v1 modelVersion을 해당 immutable 정책과 연결 |
| `repository/R1ObservationRepository.java` | 실제 request_log와 원본 호출 조회, immutable 관측 저장, 명시적 event ID 목록의 안정적 export |
| `service/R1ObservationService.java` | 내부 Java evaluator/offline adapter 진입점. cache·fallback·실패·모델 불일치·호출 이전 판정을 거부 |
| `infra/postgres/migrations/11-r1-observation.sql`, `init/01-schema.sql` | 관측·판정 ID 및 request+candidate 중복 제약, provenance와 JSON payload 저장 |

새 controller나 공개 assessment API는 추가하지 않았다. 공개 API의 새 필드도 없다. HTTP 응답 성공, 에스컬레이션 종료, `userSatisfied=true`, 불만 제기가 없다는 사실만으로 `success`를 만들지 않는다. R1 관측이 UI에서 자동 수집되고 있다는 뜻이 아니다.

## 내부 관측 절차

1. 담당자는 승인된 배포 기록에서 model manifest의 SHA-256을 가져온다. 관측 대상 파일에서 방금 계산한 값을 검증 기준으로 쓰면 pin의 의미가 사라진다. `VerifiedR1PolicyBinding.fromPinnedPackage(manifestPath, policyPath, expectedManifestSha)`는 manifest bytes와 그 manifest가 가리킨 policy bytes를 검증한다. canonical policy 내용 검증은 승인된 ML/FastAPI 공통 정책 validator의 책임이며, Java가 별도 부동소수점 canonicalization을 만들지 않는다.
2. 승인된 평가 adapter는 `Assessment`에 실제 호출 requestId, 실제 candidate 모델 ID, 판정 ID·event ID, 명시적 success true/false, source (`explicit_user`, `reference_evaluator`, `human_review`), UTC 시각, target contract version을 전달한다. 현재 UI는 이 adapter를 호출하지 않는다.
3. adapter는 질문을 실제로 비식별화한 `redactedQuestion`과 명시적 `piiRedacted=true`를 제공한다. 서비스는 마스킹을 수행했다고 추정하지 않는다. DB에는 원본 질문의 SHA와 request_log ID를 남기고 export에는 adapter가 제공한 비식별 질문만 넣는다. 신뢰된 adapter가 원문과 다른 무관한 질문을 전달하지 않도록 해당 adapter의 redaction 검증도 필요하다.
4. 서비스는 해당 request_log와 원본 chat 호출을 읽고 성공한 비캐시·비fallback 호출인지, 원본 질문과 routerModelVersion이 맞는지, 실제 model ID가 검증된 정책의 tier 후보에 해당하는지 확인한다. 응답에 없는 전체 action scores를 합성하지 않는다.
5. 동일 event/assessment ID 및 request+candidate는 DB에서 유일하다. 완전히 같은 재입력은 멱등이며 내용 변경·상충 success·새 ID를 사용한 같은 후보의 중복 평가 저장은 거부한다. 다른 후보를 실제로 호출하고 평가했다면 별도 관측으로 저장할 수 있다.
6. `exportJson(eventIds)`는 지정한 관측을 event ID 순서로 정렬해 `r1-observation-batch-v1` JSON 문자열을 만든다. 원격 업로드·HTTP 제출·학습은 실행하지 않는다. 원자적인 파일 작성과 GCS immutable object 업로드는 후속 연결 단계다.

관측 하나는 후보 하나에 대한 판정이다. export에는 관측하지 않은 후보의 success=false나 추측된 mask를 넣지 않는다. ML/FastAPI projection이 `[low, medium, high]` 순서로 실제 관측 위치만 mask=1, 미관측 위치는 mask=0·target=null로 처리해야 한다. 여러 관측을 합칠 때는 정책·후보군·질문 일치 및 상충 판정을 다시 검증한다.

## 기존 API와 재학습 연결

`/api/retraining/submit` 경로·DTO는 남지만 R1 profile에서 legacy FeedbackBatchService 제출 진입점은 명시적으로 차단된다. 기존 예외 응답 형식을 사용하며, 수동 버튼을 눌러도 CE 학습을 시작하지 않는다. 자동 submit과 미접수 retry scheduler도 비활성화한다. `auto-submit=false`만으로 수동 제출까지 막힌다고 가정하지 않는다.

현재 `/escalate`의 `feedback-inline-v1` body를 관측 DTO로 교체하지 않는다. 후속 R1 pipeline은 서버 내부의 immutable 관측 artifact를 검증·join하는 경로가 필요하다. 기존 요청과 artifact를 연결할 batch 식별자, artifact SHA/generation, 정책·target contract pin, 멱등성 규칙과 신뢰된 evaluator adapter가 완성되기 전까지 legacy 제출 차단을 유지한다. 이는 기존 API에 조용히 추가 필드를 넣는 방식으로 해결하지 않는다.

## 배포 순서와 검증

공통 migrations 09·10 이후 migration 11을 적용한다. 빈 DB는 갱신된 init SQL을 사용한다. 기존 행은 추정 backfill하지 않는다. profile `r1`은 `R1_OBSERVATION_ENABLED=false`가 기본이며, 승인된 평가 adapter를 연결할 때만 true 및 `R1_TARGET_CONTRACT_VERSION`을 설정한다. 관측 비활성 상태에서도 기존 chat·escalate·분류 경로는 유지된다. 8192는 FastAPI package 설정이며 추론 timeout 15초를 임의 변경하지 않았다.

검증은 기존 전체 Spring suite와 shared JSON fixture 직렬화, Boolean coercion 거부, 정책 파일 변조 거부, v1 provenance 검증, profile 기반 CE 제출 차단, 임시 PostgreSQL에서 실제 저장·동시 멱등·상충 판정 거부를 포함한다. 운영 DB migration, 승인 모델 생성, 원격 artifact 업로드, 서버 배포는 이 작업에서 실행하지 않는다.

## 이번 검증 기록

2026-09-26 Java 24.0.2에서 Spring 전체 테스트 186개를 별도 로컬 PostgreSQL과 함께 실행해 실패·오류·skip 0을 확인했다. 관측 저장 및 동일 입력 동시 import 테스트는 임시 `r1_test_*` 스키마만 사용했다. 공유 정책 fixture의 실제 canonical SHA는 `072b300cbb93f87dda228ad67adabaa7420d70d0d1bf58fdd686851cd532a54e`이며 Java artifact fixture가 같은 값을 사용한다.
