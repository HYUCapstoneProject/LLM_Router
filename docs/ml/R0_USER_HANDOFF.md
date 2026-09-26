# R0 운영 재학습: 사용자·Spring 담당자 인계

2026-09-27 결정: 기존 큰 A 데이터로 R0 학습한 ModernBERT를 사용한다. R1 도입은 보류하며 R1→R0 전환 학습을 하지 않는다. 외부 API와 finalLabel 피드백은 유지한다. 학습 이력은 512, 새 추론·평가는 8192다. [단일 문서 §5.11–5.12](README.md#common-first-implementation)가 현재 계획의 기준이다.

## 출발 모델

확인 대상은 `modernbert-initial-v1-20260924` recovery final이다. RouterBench+xRouteBench A의 train 21,406 / dev 3,581, optimizer 2,007 step 기록을 가진다.

- 보존 archive: `gs://llm-router-mlserver/training-runs/initial-v1-20260924-2cee27e/final-model.tar`
- archive SHA-256: `d15da837971f5535f1f96dcbe036717d066e339afad923d449c8a74e5a041e4f`
- weights SHA-256: `e1a40fec7d269d8f925b5391e4f5e75df5d0fedbb1e37b4eea96521e27b1a6a7`

2026-09-27 GCS에서 다시 다운로드해 archive·weights 및 artifact manifest의 7개 파일 SHA를 모두 검증했다. 기존 운영 CURRENT의 8월 RouterBench-only R0와 혼동하지 않는다. 보존 archive 자체는 승인된 새 운영 baseline/package가 아니다.

## 사용자가 직접 하면 효율적인 작업

| 담당 | 할 일 | 전달할 최소 결과 |
|---|---|---|
| 사용자 | GCP Billing에서 해당 프로젝트에 적용되는 크레딧 잔액·만료일을 확인 | 잔액, 통화, 만료일, 조회 시각. 결제수단·계정 토큰은 보내지 않음 |
| Spring 담당자 | PR diff, migration 09/10, `prod,r0` 설정, 기존 API·인증·15초 timeout 검토 | 적용할 코드 버전·DB 적용 이력·환경변수 이름. secret 값은 공유하지 않음 |

새 유료 GPU 재학습은 크레딧·만료일 확인 전 실행하지 않는다. 이번 초기 모델 검증은 기존 결과 재사용과 로컬 CPU 평가로 준비하므로 Colab·Drive 추가 확인은 현재 사용자 선행 작업이 아니다. 로컬 코드 검증·패키지 SHA 확인·설정 준비는 결제 확인과 독립적으로 진행한다.

## Spring 배포 담당자 확인

- 운영 profile은 `SPRING_PROFILES_ACTIVE=prod,r0`. `r0`만 켜면 `prod`의 gateway-key 기동 검사가 빠진다. `ROUTER_MODE=fastapi`, 기존 `ROUTER_BASE_URL`, `ROUTER_API_TOKEN`, `ROUTER_MODEL_VERSION`을 실제 승인된 FastAPI 배포와 맞춘다.
- `application-r0.yml`의 `auto-submit: false`를 유지한 채 인수한다. 기존 수동 제출도 배포·migration 중 사용하지 않는다. 자동 활성화는 전체 재학습·실제 적재 검증 후 별도 단계다.
- 기존 DB는 08까지 적용 여부·백업 확인 후 `09-routing-decision.sql`, `10-feedback-batch-snapshot.sql` 순서로 적용한다. R1 관측용 11은 R0에 필요 없다. 상세 절차는 [배포 문서](../../infra/DEPLOY.md#2-6-재배포--스키마-마이그레이션이-있을-때)와 [공통 Spring 인계](SPRING_COMMON_HANDOFF.md)를 따른다.
- migration 10이 격리한 과거 배치는 최초 요청·FastAPI/GCS 상태를 대조한다. 새 ID로 중복 제출하거나 snapshot을 지우지 않는다.
- 기존 자동 배포 도우미는 현재 컨테이너 환경을 이어받는다. `prod,r0`를 자동으로 설정해 준다고 가정하지 않는다.
- `/escalate` 202는 접수일 뿐 완료가 아니다. 승격 증거와 CURRENT, FastAPI 실제 `modelVersion`, Spring polling이 일치해야 완료다.

## 남은 인수

A R0 archive 검증 → 8192 baseline·D_dev/D_gate 평가 및 기준 동결 → 승인 package·이미지·파이프라인 설정 → DB/서버 배포 → 실제 8192 성능과 에스컬레이션 전체 흐름·복구 검증 순서다. 현재 운영 모델 교체나 성공한 실제 재학습을 완료했다고 표시하지 않는다. 기존 피드백의 최종 단계는 대리 라벨이므로 기존 학습 데이터 복습·보호 평가를 유지한다.
