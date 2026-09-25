# R1 서비스 적용과 재학습 인수 계획

2026-09-26 기준. 서비스 구현의 대상은 R1이며 R0는 교체 가능한 대안으로 유지한다. 기존 HTTP 경로와 요청·응답 계약은 유지한다. 공통 기반을 먼저 완성하고 그 기반에서 R1의 관측·학습·서빙 코드를 분리한다. 현재 기준은 [단일 명세 §5.11](README.md#common-first-implementation), 8192 추론의 근거와 한계는 [§1.2](README.md#context-length-adoption)를 따른다.

이 문서는 로컬 브랜치의 구현과 남은 운영 검증을 구분한다. 실제 서비스 후보의 평가, 승인 패키지 게시, 서버 적용, 자동 재학습 연결을 완료했다는 기록은 아니다. 이전 `/private/tmp` 작업 공간은 전원 종료로 소실되었으며 실행 위치로 사용하지 않는다. 현재 R1 작업 공간은 `.worktrees/r1-ml`과 `.worktrees/r1-serving`이다.

## 연구 근거와 서비스 채택의 범위

R0/R1/R2는 같은 ModernBERT 초기 revision, train 21,406행, D_dev 3,581행, 학습 길이 512, 2,007 optimizer steps, seed 42로 비교했다. 개발 검증 source-macro 비용–품질 nAUC는 R1 0.863179, R2 0.833874, R0 0.166115였다. 이후 G1 비교와 [통합 분석](../../experiments/results/routing-method-review-20260925/REVIEW.md)을 거쳐 R1-natural을 후속 구현의 기준으로 삼았다.

이 결과는 실제 Claude 후보의 성공 확률이나 운영 품질을 입증하지 않는다. 원본 test 7,202문항 평가에서는 기본 상대 비용 상한 0.50에 대해 R1의 xRouteBench 비용이 0.537229였으며 test로 λ를 다시 선택하지 않았다. [개발 검증 보고서](../../experiments/results/routing-objectives-continual-v1-A-20260924/REPORT.md)와 [봉인 test 보고서](../../experiments/results/routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md)에 해시·조건·한계가 있다.

추가 학습 없는 512 대 8192 비교는 새 추론 패키지를 구현·검증할 근거다. 기존 `training_max_length=512` 기록은 보존하고 `inference_max_length=8192`를 별도 계약으로 다룬다. 후보·baseline 평가도 같은 추론 길이여야 한다. 이 선택만으로 FastAPI 운영 지연·메모리, 실제 후보의 장문 답변 품질, 서비스 비용 gate가 통과한 것은 아니다.

연구 가중치 백업 `gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/pointwise/final-results.tar`는 승인된 운영 package나 `CURRENT.json`이 아니다. 운영 logical `1.0` 게시·활성화를 완료했다고 표시하지 않는다.

## 현재 구현과 남은 연결

| 영역 | 현재 구현 | 남은 운영 확인 |
|---|---|---|
| 공통 기반 | 정책·패키지·학습/추론 길이 계약, 배치 요청 snapshot·멱등성·CAS, pipeline 종료와 실제 활성화 구분 | 실제 배포 설정과 자격 증명, 서버에 적용한 상태 확인 |
| FastAPI 분류 | R1 정책을 내부에서 적용하고 기존 `/router/classify` v1의 tier/model/confidence/modelVersion으로 투영 | 실제 서비스 모델 pool·정책으로 서버 성능과 fallback 검증 |
| Spring | v1 호출 유지, 실제 호출 provenance 검증, 내부 Java/offline assessment 저장·JSON export, R1 profile에서 legacy CE 수동·자동·재시도 제출 차단 | 신뢰된 평가 adapter·실제 redaction·운영 DB migration 적용 |
| R1 관측 artifact | `r1-observation-v1`, `r1-observation-batch-v1`, 모델·정책·후보군 pin과 실제 관측 후보의 target/mask projection | immutable 업로드와 기존 재학습 요청을 서버 내부 artifact에 연결하는 경로 |
| R1 학습 | 관측한 후보에만 적용하는 masked BCE 경로. 미관측 target을 실패로 채우지 않음 | 실제 서비스 관측을 사용한 prepare/merge/replay, 이미지·IR·실행·결과 연결 검증 |
| 승격·rollback | 공통 무결성·버전·pointer CAS 기반 및 R1 정책 계약 | 서비스 D_gate의 품질–비용·회귀 조건, 같은 길이 baseline 비교, 실제 R1 승인·적재·rollback 검증 |

기존 실험의 v2 opt-in 코드는 공통 확장 기능으로 남아 있으나 R1 서비스 전환의 필수 HTTP 계약이 아니다. `model-package-v2`와 `routing-policy-v2`는 내부 artifact schema이며 `/router/classify/v2` 호출을 의무화한다는 뜻이 아니다. Spring v1 응답에 없는 policy hash나 전체 action scores를 응답에 있었다고 추정하지 않는다. 검증된 immutable 정책과 반환 modelVersion을 연결하고 실제 호출 모델 ID를 확인한다. 구체적인 파일·설정·DB 순서는 [R1 Spring 인계 문서](R1_SPRING_HANDOFF.md)에 있다.

## 관측의 의미와 내부 경계

`userSatisfied=true`는 기존 체인 종료 신호이며 R1 success가 아니다. HTTP 200, 답변 완료, 에스컬레이션 종료, 불만 제기가 없다는 사실도 후보 성공으로 바꾸지 않는다. 내부 adapter가 실제 호출에 대해 명시적 `success=true/false`, 평가 출처·ID·시각을 제공해야 한다. 허용 출처 이름이 `explicit_user`여도 현재 UI에 사용자 평가 수집 기능을 추가했다는 뜻은 아니다.

Spring은 cache·fallback·실패 호출과 모델·정책 불일치를 거부하고, 검증된 관측을 DB에 보존한다. `exportJson(eventIds)`는 명시적으로 선택한 관측의 비식별 JSON을 만들며 업로드나 학습을 시작하지 않는다. 미관측 후보는 target=null·mask=0으로 남기고 관측한 후보만 mask=1로 projection한다. 실제 데이터에서 중복·상충 판정, 동일 정책과 후보군, target 의미와 holdout 보호를 함께 확인해야 한다.

기존 `/escalate`의 `feedback-inline-v1` 요청을 관측 DTO로 교체하거나 필드를 몰래 추가하지 않는다. R1 profile의 legacy CE 제출은 관측 artifact와 서버 내부 R1 경로가 완성될 때까지 차단한다. 자동 ingest, 원격 업로드, 기존 요청과 artifact의 join, 재학습 job 제출을 한 흐름으로 검증한 상태가 아니다.

## 후속 작업의 순서와 완료 조건

로컬 ML 진입점은 [관측 준비 CLI](../../ml/scripts/prepare-r1-observations.py)다. 검증한 policy 파일 SHA, 허용 modelVersion, 보호 문항 hash 목록을 명시적으로 받으며 새 디렉터리에 Parquet와 입력·출력 SHA manifest를 만든다. 같은 요청의 관측은 한 학습 행으로 모으고 미관측 후보는 null/mask0으로 남긴다. [merge 함수](../../ml/src/llm_router_labeling/training/r1_observations.py)는 동일 정책·후보군의 base와 결합하며 group ID와 정규화 문항 hash 양쪽으로 평가 데이터 누수를 거부한다. `--objective pointwise` 학습은 null-only Parquet 및 기존 난이도 metadata가 일부 없는 행도 처리하고 관측한 후보의 BCE만 평균한다. 관측 편향을 자동 보정하거나 미호출 후보의 성공을 알게 되는 방법은 아니다.

[R1 qualification](../../ml/src/llm_router_labeling/evaluation/r1_gate.py)의 `summarize_r1_predictions`와 `evaluate_r1_gate`는 같은8192·정책·데이터·후보군의 candidate/baseline을 비교한다. 질문별 세 후보 outcome과 사전에 동결한 전체·domain 품질 회귀 한도 및 비용 상한이 필요하다. `r1-qualification-result-v1` 결과는 `automatic_promotion_supported=false`이며 승인 package를 생성하거나 CURRENT를 바꾸지 않는다. 기존 publisher에 이 결과만 전달해 자동 승격할 수 없고, 실제 파일 hash와 실행을 결합하는 R1 전용 연결이 남아 있다. Synthetic test fixture는 운영 policy나 성공 판정 데이터가 아니다.

1. **공통 기반과 배포 선행 설정을 확인한다.** 공통 코드·GCS 경로·IAM·상태 저장·패키지 계약을 먼저 확정한다. 실제 서버 주소, 배포된 image digest, DB migration과 같이 배포 이후에만 확정 가능한 항목은 해당 단계에서 적용한다.
2. **실제 서비스 pool과 정책을 동결한다.** 후보 3개의 실제 Claude 모델 ID, 가격·decision cost 버전, λ와 평가 대상·판정법을 기록한다. 공개 benchmark 후보의 성공률을 현재 Claude 후보의 성공률로 해석하지 않는다.
3. **명시적 관측 수집과 immutable artifact 연결을 검증한다.** 승인된 내부 평가 adapter, 실제 비식별화, 호출 provenance, 상충 판정 처리, artifact SHA/generation 및 batch 연결을 검증한다. HTTP 계약은 유지한다. UI나 자동 수집이 없는 부분을 구현 완료로 표시하지 않는다.
4. **관측 artifact부터 masked BCE 학습까지 연결한다.** 실제 prepare/merge/replay가 target·mask·후보·판정 계약을 보존하는지 확인하고 training/evaluation image digest와 KFP IR에 이를 전달한다. 관측하지 않은 후보를 실패로 채우거나 CE용 난이도 정답을 합성하지 않는다. 이 과정에서 base rehearsal와 보호된 holdout을 유지한다.
5. **서비스 품질–비용 gate를 통과한다.** 동결된 D_gate outcome에 같은 정책을 적용해 baseline 대비 품질·비용·source별 회귀를 확인한다. D_gate에서 λ나 후보를 재선택하지 않는다. 후보와 baseline의 평가 길이·데이터·목표·정책·pool·모델 해시를 확인한다. 봉인 test와 외부 장문 라벨 개선은 이 gate를 대체하지 않는다.
6. **게시·적재와 실패 경로를 검증한다.** immutable candidate/approved package, CURRENT, FastAPI 적재, 기존 Spring v1 응답 및 실제 호출 모델 연결을 확인한다. 8192 입력의 운영 지연·메모리·동시 요청·timeout/fallback을 측정하고 재학습 1회와 실패·rollback을 검증한 후 운영 완료로 기록한다.

R1을 서비스 대상에 두는 결정과 운영 검증 완료는 구분한다. R0 교체 가능성은 유지하며, 두 방식이 검증되지 않은 상태에서 같은 활성 pointer를 경쟁 갱신하지 않도록 배포 경로와 전환 절차를 명시한다.
