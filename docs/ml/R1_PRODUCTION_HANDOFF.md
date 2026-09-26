# R1 초기 모델과 운영 재학습 인수 계획

2026-09-24 기록과 2026-09-25 인수 갱신. [최종 실험 명세](README.md#retraining-objective-handoff)를 실제 실행 상태와 연결한다.

## 2026-09-26 공통 기반 우선 인수

현재 실행 순서는 [단일 명세 §5.11](README.md#common-first-implementation), 8,192 추론 적용 근거는 [§1.2](README.md#context-length-adoption)를 따른다. 아래 9월 24~25일 내용은 당시 구현 이력이다. 현재 공통 작업 공간은 저장소 안의 `.worktrees/common-ml`과 `.worktrees/common-serving`이며, 소실된 `/private/tmp` 작업 공간을 실행 위치로 사용하지 않는다. 학습 512 기록은 보존하고 새 추론 길이는 8,192로 분리한다.

## 2026-09-25 구현 인수 갱신

G1 네 arm 평가와 [통합 분석](../../experiments/results/routing-method-review-20260925/REVIEW.md)을 마쳤고, 사용자는 R1 방식의 구현 브랜치와 기존 R0 방식의 별도 완성 브랜치를 요청했다. 이는 연구 체크포인트의 즉시 운영 승인을 의미하지 않는다. R1-natural은 구현의 기준이며 실제 서비스 후보의 관측 데이터·비용 정책·승격 gate가 필요하다.

이번 main 통합은 ML 코드·실험 결과·문서에 한정한다. 아래 `c944d37`의 Spring 코드·설정·테스트 6개 파일은 **기존 실험 브랜치에만 보존하고 main 통합에서 제외했다**. main의 Spring은 기존 v1 계약을 유지한다. 통합 main에서 분기한 R1 브랜치에 필요한 Spring 변경을 별도 구현·검증하고 담당자가 검토할 파일별 목록을 제공한다. R0는 별도 브랜치에서 기존 finalLabel CE 흐름을 완성한다. 두 방식의 모델 게시 경로와 활성 포인터를 공유해 경쟁 갱신하지 않는다.

현재 재학습 오케스트레이터는 managed Vertex AI Pipelines이며 GPU 학습은 Vertex AI Custom Training이다. 아래 목록은 구현 체크리스트이며 실제 클라우드 실행·승격 완료 기록이 아니다.

## 확정한 결과

- R0/R1/R2는 같은 ModernBERT 초기 revision, train 21,406행, D_dev 3,581행, 512 tokens, 2,007 optimizer steps, seed 42로 비교했다.
- 개발 검증 source-macro 비용–품질 nAUC는 R1 **0.863179**, R2 **0.833874**, R0 **0.166115**다. **R1을 초기 `1.0`의 채택 후보로 고정했다.**
- 세 방법 모두 `low/medium/high → easy/medium/hard → haiku/sonnet/opus`의 세 단계로 연결한다. R1은 세 후보의 성공 추정치를 독립적으로 출력한다.
- 원본 test 7,202문항의 고정 정책 평가를 완료했다. 기본 상대 비용 0.50에서 R1은 R0보다 품질과 비용이 모두 높았고 xRouteBench 비용은 0.537229로 상한을 넘었다. test로 방법이나 λ를 다시 고르지 않았다.
- [개발 검증 보고서](../../experiments/results/routing-objectives-continual-v1-A-20260924/REPORT.md)와 [봉인 test 보고서](../../experiments/results/routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md)에 원본 해시·한계·보관 증거가 있다.

선정된 R1 가중치와 실행 자료는 `gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/pointwise/final-results.tar`에 있다. 이 파일은 연구 실행 백업이며, 승인된 운영 package나 `CURRENT.json`은 아니다. 개인 Drive에도 flush·재연결 후 파일 검증을 마쳤다. 운영 logical `1.0` 게시·활성화는 아직 완료하지 않았다.

## 준비된 코드

| 영역 | 구현·검증 | 운영 상태 |
|---|---|---|
| ML 학습 진입점 | `9991ca4`: `modernbert-train --objective pointwise`, 이미지의 BCE loss 모듈 포함, CPU 실제 loss 연결 검사 | image 재빌드·원격 IR 연결 전 |
| 추론·패키지 | `2bb4395`, 격리 serving `7a6cb03`: 명시적 정책, 512 정규화, R1 sigmoid와 세 단계 선택, v2 package 검증 | 실제 서비스 정책 미설정 |
| Spring 연결 | `c944d37`: v2 opt-in, canonical policy hash·후보 3개 실제 ID·score 계약 검사; 관련 테스트 70개 통과 | 기본 v1 유지·미배포 |
| 게시·rollback | `2d0e503`, `e565f21`: 동일 물리 버전 덮어쓰기 거부, 올바른 이전 manifest 복원·CAS | 기존 CE gate가 v2를 승인하는 경로 차단 유지 |
| 서버 재학습 상태 | 격리 serving의 Vertex enum·terminal state 관측 수정과 회귀 검증 | 원본 저장소의 미커밋 문서 보존, 배포 전 |

서빙 격리 작업 공간은 `/private/tmp/llm-router-serving-retraining-fix`다. 원본 serving 저장소의 사용자 수정·삭제 파일을 덮어쓰지 않았다. 새 v2 계약은 serving 통합본의 `docs/api-contract.md#routing-policy-v2`에 정리했다.

## 다음 작업의 순서와 완료 조건

1. **서비스 후보와 정책을 동결한다.** 실제 Claude 모델 ID 3개, 가격·고정 decision cost 버전, λ를 기록한다. 공개 benchmark의 역사적 후보 성공률은 Claude 성공률이 아니므로 서비스 평가 자료로 mapping과 정책을 확인한다. 공개 source별 λ의 평균을 서비스 λ로 사용하지 않는다. 서비스 평가의 표본·판정법·신규 호출 범위는 실행 전에 정한다.
2. **관측된 피드백 계약을 정한다.** 현재 Spring의 `userSatisfied=true`는 명시적 만족이 아니라 체인 종료 신호다. 이를 success=1로 사용하지 않는다. 사용자가 답한 만족·불만족을 success proxy로 채택한다면 별도 target version으로 구분하고, 실제 호출 모델·정책·판정 출처·관측 mask를 보존한다. 명시적 만족·선호 입력의 UI 추가 여부는 사용자 답변을 기다리고 있다.
3. **R1 재학습을 연결한다.** prepare/merge/compact replay가 후보별 target·mask·pool 정보를 보존하도록 바꾸고, 관측한 항에만 BCE를 적용한다. 미호출 후보를 실패로 채우거나 CE용 가짜 난이도 정답을 만들지 않는다. canonical digest에는 target·mask·후보·판정 계약을 포함한다. 기존 holdout 보호와 base rehearsal를 유지한다. 현재 운영 trainer의 R1 경로는 완전 관측 target만 지원한다.
4. **R1 정책의 승격 gate를 구현한다.** 동결한 D_gate outcome에 같은 정책을 적용해 기준 모델 대비 품질·비용과 source별 회귀를 검사한다. D_gate에서 λ나 후보를 재선택하지 않는다. 이번 봉인 test의 통계 결과는 운영 승인서가 아니다. dataset/objective/policy/pool/model hash와 baseline이 모두 맞는 새 gate에만 v2 승격을 허용한다.
5. **이미지·IR·설정을 하나의 release로 검증한다.** training/evaluation image digest와 KFP IR에 objective·policy·target 계약을 전달하고, 잘못된 설정·부분 관측·중복·실패 gate가 CURRENT를 바꾸지 않는지 검증한다. 이전 오류 batch의 실제 Vertex terminal state를 SQLite에 재조정하는 경로도 확인한다. 기존 상태 DB는 보존한다.
6. **운영 `1.0`을 게시하고 함께 전환한다.** 후보 package 검증 → 신규 gate → immutable approved package → CURRENT → FastAPI 적재와 Spring v2 policy pin·pool·실제 모델 mapping을 함께 확인한다. 재학습 1회와 실패·rollback까지 검증한 뒤 완료로 기록한다. 단일 policy pin이므로 서버만 교체하면 클라이언트가 거부할 수 있다.

현재 연구 성능만으로 `1.0` 승인이나 자동 재학습 활성화를 표시하지 않는다. R1을 선택한 결정을 유지하면서 위 인수 조건을 완성한다.
