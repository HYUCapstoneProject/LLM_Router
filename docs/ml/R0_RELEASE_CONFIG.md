# R0 교체 프로필과 release 준비 계약

이 문서는 `feat/r0-retraining-8192`의 실행 인터페이스다. 전체 결정·실험 해석은 [최종 실험 명세](README.md#context-length-adoption), 공통 선행 작업과 단일 활성 모델 정책은 [공통 구현 순서](README.md#common-first-implementation)를 따른다. 2026-09-27 사용자 결정에 따라 기존 큰 A 데이터로 학습한 R0를 운영 재학습의 출발점으로 사용한다. R1 도입과 R1→R0 전환 학습은 보류한다. 이 브랜치의 `cloud-foundation-v1.json`만 `target_profile=r0`다.

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

2026-09-27 실제 R0 8192 baseline의 초기 qualification·GCS 승인 등록과 D_dev pilot/rules 결합을 완료했다. 실제 파일·SHA는 [등록 검증 기록](../../experiments/results/r0-production-qualification-20260927/REPORT.md)에 있다. 이미지 digest·새 IR·운영 배포 인수는 아직 남아 있다. 이 문서나 synthetic fixture로 누락을 대신하지 않는다. 초기 baseline의 qualification·등록은 별도 실제 승인 증거가 필요하며, 이 helper가 새 baseline을 승인해 주지 않는다.

## Spring과 FastAPI 설정

Spring은 `SPRING_PROFILES_ACTIVE=prod,r0`와 기존 `ROUTER_BASE_URL`, `ROUTER_API_TOKEN`, `ROUTER_MODEL_VERSION`을 사용한다. `application-r0.yml`은 FastAPI/v1을 선택하고 자동 제출을 중지한다. 실제 CPU 장문 측정에 따라 R0의 기본값은 `ROUTER_READ_TIMEOUT=PT240S`, `ROUTER_MAX_ATTEMPTS=1`이며 환경변수로 조정할 수 있다. 입력 8,192토큰 지원과 기존 15초 응답 목표 충족은 구분한다. [단일 문서의 HTTP 대기 예산](README.md#r0-long-context-http-budget)에 실측 및 proxy 설정 근거를 기록했다. 로컬 stub이 필요하면 `ROUTER_MODE=stub`을 명시한다.

FastAPI 저장소의 `deploy/r0.env.example`은 R0 registry/CURRENT, feedback/control/observability, `/app/var/r0/` cache/SQLite, 실제 공통 프로젝트·서비스 계정을 고정한다. 실제 secret, 승인 baseline version, full code SHA와 생성한 release/IR/data URI를 채운 후 다음 로컬 검사를 실행한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate-r0-profile.py --env-file deploy/r0.env
```

검사기는 기존 Settings의 환경변수 우선순위를 그대로 적용한다. 길이512·legacy 입력·다른 objective 경로·기존 smoke/empty baseline·누락 필수값을 거부하며 secret 값은 출력하지 않는다. 검사 결과의 `profile_valid=true`는 로컬 구성 적합성뿐이다. cloud 객체 내용, 실제 IAM, 지표·승격·서빙 활성화는 이 명령이 검증하지 않는다. 예제 파일 그대로는 exit 1이 정상이다.

## 교체와 실제 인수

GCS에는 활성 objective의 모델 그룹 하나만 둔다. R0/R1의 예약된 namespace metadata와 공통 IAM 존재를 두 모델 그룹 배포로 해석하지 않는다. 교체 전에 제출·승격을 중지하고 진행 중인 작업을 종료한다. 기존 current/previous/anchor와 policy/tokenizer/manifest/gate·승격 증거, 복구용 상태를 개인 Drive로 보존한 뒤 원격으로 다시 읽어 파일별 SHA를 검증한다. 그 후 기존 GCS 그룹을 generation 조건으로 제거하고 새 R0 그룹을 게시한다. Drive Desktop 복사만으로 검증 완료를 표시하지 않는다. 다른 objective의 GCS 그룹을 동시에 남기지 않으므로 교체 구간의 중단 및 Drive 복구 절차가 필요하다.

배포 이후에만 확인 가능한 항목은 실제 UID/볼륨/ADC/secret, Spring 연결, VM의 8192 p50/p95·최대 메모리·동시 요청, 202→Vertex→gate→CURRENT→활성 모델 일치, 실패·재시도·rollback이다. 공통 `router-promotion-evidence-v1`은 candidate/approved manifest·지표·gate·batch/job·CURRENT SHA와 CAS를 검증한다. Vertex `SUCCEEDED`만으로 승격이나 활성화를 완료 처리하지 않는다. R1/v2를 이 CE gate로 승인하는 경로는 열지 않았다.

## 첫 R0 8192 baseline의 별도 qualification

이미 approved baseline이 있어야 하는 일반 release 준비기와 처음 모델을 등록하는 절차를 분리한다. `ml/scripts/qualify-r0-initial.py`는 보호된 D_gate의 **실제 8192 평가 결과**를 candidate package의 파일 SHA와 대조하고, D_dev에서 미리 동결한 **절대 품질 하한**을 적용한다. 자기 자신을 baseline으로 넣거나 512 평가 metadata만 바꾸는 경로가 아니다. 이 CLI는 로컬 gate 증거만 기록하며 GCS 게시·CURRENT 변경·GPU 실행은 하지 않는다.

필수 입력은 다음과 같다.

- `--candidate-package`, `--gate-metrics`, `--development-metrics`: 실제 초기 R0 패키지와 해당 모델의 8192 D_gate/D_dev 지표.
- `--gate-dataset`, `--development-dataset`, `--config-dir`: frozen 데이터 SHA와 평가 계약. D_dev와 D_gate는 서로 다른 데이터다.
- `--rules-path`, `--pilot-report`: `r0-initial-rules-v1`와 `r0-initial-pilot-v1`. macro-F1, hard recall, 각 source/domain macro-F1 절대 하한을 외부에서 명시하고, baseline 대비 delta는 사용하지 않는다. 도구에 기본 하한은 없다.
- `--root-uri`, `--expected-current-generation`: 게시 대상 registry와 확인한 CURRENT generation. CURRENT가 없을 때만 `absent`를 명시한다.
- `--output`: 아직 존재하지 않는 새 결과 파일. 불합격도 증거를 기록하고 exit 2로 종료한다.

`rules.required_evaluation_contract`는 D_gate의 role/split/version/data SHA/label semantics/order/model_info.max_length/dataset contract 전체를 고정한다. `threshold_selection`은 D_dev SHA, `frozen_before_gate=true`, pilot report 원본 SHA를 고정한다. pilot은 초기 candidate version/weights SHA, D_dev metrics SHA, 길이8192, `selected_rules`(threshold_selection을 제외한 rules)를 묶는다. 이 동결 선언 자체가 실행 시각을 증명하지는 않으므로 D_gate 실행 전 파일·로그 보존이 필요하다. 초기 gate 통과는 내부 난이도 라벨에 대한 qualification이며 실제 Claude 응답 품질이나 서버 성능 보장은 아니다.

통과 결과는 기존 `deploy-gate-result-v1`에 `qualification_mode=initial_baseline`을 명시한다. 기존 publisher의 `publish-candidate --gate-result`와 `approve --publish-result --training-mode initial_training`을 사용하며, 승인 시 모델·패키지·registry·qualification 당시 CURRENT generation이 일치해야 한다. 중간에 CURRENT가 바뀌면 거부하고 마지막 pointer 쓰기도 기존 generation CAS로 보호한다. 이 증거로 일반 feedback 재학습을 승인할 수 없다. 최초 승인 이후에는 기존 `prepare-r0-release.py`가 요구하는 approved baseline과 회귀 한도를 이용한다.

초기 qualification의 소표본 domain 예외는 `initial-small-domain-report-only-v1`로 명시한다. **D_dev domain 문항 수가 30 미만**이면 그 domain은 숫자 하한을 넣지 않고 `insufficient_support`, `quality_unverified=true`, 실제 문항 수를 기록한다. qualification 도구는 SHA가 고정된 D_dev parquet의 category→domain 매핑으로 문항 수를 다시 계산하여 예외 목록과 대조한다. 전체 macro-F1·hard recall·두 source와 충분한 표본의 domain 하한은 계속 필수다. D_gate에도 제외 domain 지표를 출력하며, 결과의 `quality_unverified_domains`와 `automatic_feedback_submission_authorized=false`를 유지한다. 이 예외는 초기 등록만 대상으로 하며 일반 재학습 회귀 gate를 완화하지 않는다.

2026-09-27 D_dev에서 duplicate_group_id 단위 bootstrap 1,000회(seed 42)의 pointwise 양측 95% 구간 하단(2.5 percentile)을 초기 라벨 일관성 하한으로 선택했다. conversation은 8문항이고 하한이 0이므로 위 명시 정책의 보고 전용 대상이다. 0을 통과용 threshold로 쓰거나 임의 양수로 바꾸지 않았다. 동결은 D_gate 실행 전에 이루어졌으며, 서비스 대화 품질의 보증이나 독립 검증에서의 개선 증거로 해석하지 않는다. 전체 실제 파일과 SHA는 `ml/artifacts/r0-release-20260927/qualification/development-provenance.json`에 있다.

### Vertex 학습 입력 경로 결합 (2026-09-27)

첫 합성 E2E에서 준비·병합 단계는 성공했으나, 실제 Vertex CustomJob의 `--training-path`에 KFP `ConcatPlaceholder`가 경로 대신 JSON 문자열로 전달되는 문제가 발견됐다. 학습 CLI는 이 값을 파일 경로로 읽으므로 GPU가 시작되기 전에 해당 실행을 중단하고 입력 전달을 수정한다.

초기·재학습 GPU 컴포넌트는 데이터 artifact 경로와 suffix를 별도 positional argument로 전달하고, worker의 짧은 Python entrypoint에서 문자열로 결합한 뒤 `os.execvp`로 기존 `modernbert-train`을 실행한다. 셸 해석이 없으며 이미지·학습 CLI·학습량·평가·승격 기준은 바뀌지 않는다. 컴파일된 CustomJob worker spec을 실제 subprocess로 실행하는 회귀검증으로 직접 Parquet와 directory `/training.parquet` 입력, 공백·따옴표·셸 문자가 포함된 경로의 전달을 확인한다. 새 IR 게시·서버 연결·실제 재실행 결과는 별도 증거로 남긴다.

### GCSFuse 패키지 복사 (2026-09-27)

수정된 입력 경로로 GPU 학습(97 train / 48 validation, 3 epochs / 12 steps)과 동결 평가가 성공한 뒤, 패키지 생성이 GCSFuse의 `utime` 미지원으로 실패했다. `shutil.copy2`가 바이트 복사 이후 POSIX timestamp 복제를 시도한 것이 원인이다. 체크포인트·정책·평가 파일의 패키지 복사를 `copyfile`로 바꾸고 기존 SHA 검증을 유지한다. `os.utime`이 EPERM을 내는 조건에서도 v1/v2 패키지가 생성되고, 이후 가중치 변조는 여전히 검증에서 거절되는지 회귀검증한다. 이 수정은 모델 바이트·라벨·학습·평가·승격 기준을 변경하지 않는다.
