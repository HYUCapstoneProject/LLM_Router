# R0 교체 프로필과 release 준비 계약

이 문서는 `feat/r0-retraining-8192`의 실행 인터페이스다. 전체 결정·실험 해석은 [최종 실험 명세](README.md#context-length-adoption), 공통 선행 작업과 단일 활성 모델 정책은 [공통 구현 순서](README.md#common-first-implementation)를 따른다. 기본 운영 목표 R1을 바꾸는 문서가 아니며, R0로 교체할 때 사용하는 별도 브랜치다. 이 브랜치의 `cloud-foundation-v1.json`만 `target_profile=r0`다.

## 길이·API·학습 계약

| 항목 | R0 계약 |
|---|---|
| 목적함수·패키지·승격 | ordinal 3-class CE / `model-package-v1` / `deploy-gate-result-v1` |
| 원래 학습·다음 feedback 학습 | `training_max_length=512`, shuffle 유지 |
| 추론·D_dev·D_gate | `inference_max_length=8192`, `tokenizer_max_length=8192`; 평가 batch 1 |
| 길이 명시 | `length_contract_version=train-inference-lengths-v1`; 실제 checkpoint positional limit 검사 |
| 입력·feedback | `structure-preserving-v2`, 기존 `feedback-inline-v1`와 finalLabel proxy |
| 서비스 API | 기존 `/router/classify`, `/escalate`, `/retraining-runs/{batchId}`의 v1 요청·응답 유지 |
| 버전 정책 | 기존 `successfulHttpRequest`, 3회/major 유지; 검증 없이 `validatedUniqueBatch`로 변경하지 않음 |
| 자동 제출 | Spring r0 profile의 `auto-submit: false`; 실제 인수 완료 후 별도 설정 변경 |

체인 종료에서 얻은 finalLabel은 객관적인 정답이나 모든 후보의 성공 관측이 아니다. R0 proxy의 의미를 feedback config에 명시하며 이 입력을 R1 BCE target으로 재해석하지 않는다.

8192 채택은 기존 체크포인트의 길이 평가 결과를 근거로 한 추론 목표다. R0 장문 성능 향상·동등성이나 실제 VM 응답 지연 통과를 뜻하지 않는다. 과거 512 평가 config와 학습 이력은 수정하지 않았다. 새 [D_dev 설정](../../ml/config/r0_dev_evaluation_8192.json)과 [D_gate 설정](../../ml/config/r0_continual_gate_8192.json)은 같은 frozen 데이터와 split을 보존하고 길이·batch·evaluation_version을 새로 기록한다.

## 실제 입력이 있어야 생성되는 release

`ml/scripts/prepare-r0-release.py`는 로컬 파일만 읽고 새 디렉터리를 만든다. GCS 게시·학습·승격·CURRENT 변경을 실행하지 않는다. 기존 출력 디렉터리는 덮어쓰지 않는다. CLI의 `--help`에 모든 필수 인수가 있으며 예시는 실제 파일 경로를 넣어 실행한다.

```bash
PYTHONPATH=ml/src .venv/bin/python ml/scripts/prepare-r0-release.py \
  --output-dir "$R0_RELEASE_DIRECTORY" \
  --gate-rules-path "$R0_FROZEN_RULES" \
  --pilot-report-path "$R0_DDEV_PILOT_REPORT" \
  --pilot-metrics-path "$R0_DDEV_METRICS_8192" \
  --baseline-package-path "$R0_APPROVED_BASELINE_PACKAGE" \
  --baseline-metrics-path "$R0_BASELINE_GATE_METRICS_8192" \
  --baseline-model-version "$R0_APPROVED_BASELINE_VERSION" \
  --development-dataset-path "$R0_DDEV_DATASET" \
  --base-dataset-path "$R0_BASE_DATASET" \
  --base-manifest-path "$R0_BASE_MANIFEST" \
  --gate-dataset-path "$R0_DGATE_DATASET" \
  --code-commit "$R0_FULL_GIT_SHA"
```

입력 검사와 출력의 의미는 다음과 같다.

1. 실제 baseline package의 모든 파일 hash를 검증한다. approved v1/ordinal, 학습512·추론8192, 입력·dataset·split·label·evaluation 계약이 일치해야 한다. 전달한 baseline 지표의 **원본 byte SHA**는 그 manifest의 `gate_metrics.json`과 같아야 한다. 512 지표에 새 metadata만 붙여 통과시킬 수 없다.
2. base dataset·base manifest·D_dev·D_gate 파일을 frozen config의 SHA와 비교한다. pilot metrics는 D_dev, baseline metrics는 D_gate여야 하며 양쪽의 `model_info.max_length`는 정수 8192여야 한다. 필수 pilot 지표는 유한한 숫자여야 한다.
3. `r0-gate-pilot-v1` report는 `objective=ordinal`, `evaluation_role=development_calibration`, D_dev SHA, 실제 baseline version/manifest SHA, `pilot_metrics_sha256`, 길이 marker·8192, 선택된 규칙을 담는다. `selected_rules`는 최종 rules에서 `threshold_selection`만 제외한 값과 정확히 일치한다.
4. rules의 `threshold_selection`은 D_dev SHA, `frozen_before_gate=true`, report SHA를 담는다. required contract는 모든 frozen D_gate 메타데이터와 `model_info.max_length=8192`를 고정한다. macro-F1, hard recall, 두 source와 모든 domain의 baseline 대비 회귀 한도를 요구한다. 허용 한도는 D_dev에서 정한 값이며 준비기가 임의로 채우지 않는다. smoke·placeholder 이름, 비유한 threshold, 무의미한 −1 허용치는 거부한다.
5. baseline 자체가 같은 gate의 계약·절대 하한을 만족하는지 검사한다. 이는 새 후보 승인이나 측정 실행의 증명은 아니다. report의 동결 선언만으로 실행 시각을 암호학적으로 증명할 수 없으므로 실제 pilot 실행 로그와 immutable 업로드 이력을 별도로 보존한다.

출력은 `feedback.json`, `training.json`, `evaluation.json`, `development.json`, `baseline-metrics.json`, `gate-rules.json`, `package-metadata.json`, 그리고 파일 SHA·입력 SHA를 묶은 `release.json`이다. baseline 지표는 manifest와의 SHA 결합을 유지하도록 원본 bytes를 복사한다. package metadata에는 가짜 `model_version`을 넣지 않으며 KFP가 실제 후보 버전을 package CLI의 `--model-version`으로 공급한다. `feedback.json`의 approved는 전처리 config 사용 허용을 뜻한다. release는 항상 `cloud_deployed=false`, `model_approved=false`로 기록한다.

현재 실제 8192 approved baseline·동결 pilot/rules·image digest·IR가 이 코드에 들어 있지 않다. 이 문서나 synthetic fixture로 누락을 대신하지 않는다. 초기 baseline의 qualification·등록은 별도 실제 승인 증거가 필요하며, 이 helper가 새 baseline을 승인해 주지 않는다.

## Spring과 FastAPI 설정

Spring은 `SPRING_PROFILES_ACTIVE=r0`와 기존 `ROUTER_BASE_URL`, `ROUTER_API_TOKEN`, `ROUTER_MODEL_VERSION`을 사용한다. `application-r0.yml`은 FastAPI/v1을 선택하고 자동 제출을 중지한다. `ROUTER_READ_TIMEOUT=PT15S` 등 기존 timeout은 실제 서버 측정 없이 늘리지 않았다. 로컬 stub이 필요하면 `ROUTER_MODE=stub`을 명시한다.

FastAPI 저장소의 `deploy/r0.env.example`은 R0 registry/CURRENT, feedback/control/observability, `/app/var/r0/` cache/SQLite, 실제 공통 프로젝트·서비스 계정을 고정한다. 실제 secret, 승인 baseline version, full code SHA와 생성한 release/IR/data URI를 채운 후 다음 로컬 검사를 실행한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate-r0-profile.py --env-file deploy/r0.env
```

검사기는 기존 Settings의 환경변수 우선순위를 그대로 적용한다. 길이512·legacy 입력·다른 objective 경로·기존 smoke/empty baseline·누락 필수값을 거부하며 secret 값은 출력하지 않는다. 검사 결과의 `profile_valid=true`는 로컬 구성 적합성뿐이다. cloud 객체 내용, 실제 IAM, 지표·승격·서빙 활성화는 이 명령이 검증하지 않는다. 예제 파일 그대로는 exit 1이 정상이다.

## 교체와 실제 인수

GCS에는 활성 objective의 모델 그룹 하나만 둔다. R0/R1의 예약된 namespace metadata와 공통 IAM 존재를 두 모델 그룹 배포로 해석하지 않는다. 교체 전에 제출·승격을 중지하고 진행 중인 작업을 종료한다. 기존 current/previous/anchor와 policy/tokenizer/manifest/gate·승격 증거, 복구용 상태를 개인 Drive로 보존한 뒤 원격으로 다시 읽어 파일별 SHA를 검증한다. 그 후 기존 GCS 그룹을 generation 조건으로 제거하고 새 R0 그룹을 게시한다. Drive Desktop 복사만으로 검증 완료를 표시하지 않는다. 다른 objective의 GCS 그룹을 동시에 남기지 않으므로 교체 구간의 중단 및 Drive 복구 절차가 필요하다.

배포 이후에만 확인 가능한 항목은 실제 UID/볼륨/ADC/secret, Spring 연결, VM의 8192 p50/p95·최대 메모리·동시 요청, 202→Vertex→gate→CURRENT→활성 모델 일치, 실패·재시도·rollback이다. 공통 `router-promotion-evidence-v1`은 candidate/approved manifest·지표·gate·batch/job·CURRENT SHA와 CAS를 검증한다. Vertex `SUCCEEDED`만으로 승격이나 활성화를 완료 처리하지 않는다. R1/v2를 이 CE gate로 승인하는 경로는 열지 않았다.
