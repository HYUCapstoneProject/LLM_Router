# 실행·산출물·검증 상태

기준일: 2026-09-20. 이 문서는 [최종 ML 명세](README.md)의 **실행 순서, 재현 계약, 구현 차이와 결과 보존**을 관리한다. 설계가 확정되었다는 것과 구현·실험이 완료되었다는 것을 구분한다. 이번 통합에서 유료 학습, Claude/GPT API 호출, GCP 배포는 실행하지 않았다.

## 1. 실행 경계와 순서

| 작업 | 실행 환경 | 산출물 |
|---|---|---|
| 공개 데이터 검증·정규화·split·threshold audit | 로컬 CPU/Colab | 봉인한 dataset, metadata, manifest |
| 초기 ModernBERT, 표현/모델·목표함수·논문 라우터 비교 | Colab 중심 | checkpoint, validation lock, test prediction·metric |
| 최소 feedback 성능 실험 | 로컬/Colab 공통 core | F0/F1/F2 stream·update·gate·비용 ledger |
| 운영 feedback 자동화 | 관리형 Vertex AI Pipelines CPU + Custom Training T4 | GCS package, gate, CURRENT, 활성화 상태 |
| 캐시 실험 | 동일한 로컬/serving 측정 환경 | trace, 재사용 판정, 실제 처리 지연·비용 |

Colab 초기학습이 공식 경로다. 기존 `initial-training-pipeline.yaml`은 compile regression용이며 신규 초기학습을 Vertex에 제출하지 않는다. 여러 비교 모델은 운영 자동 승격 대상이 아니다. GKE나 자체 설치 Kubeflow를 추가하지 않는다.

실행 순서는 다음과 같이 고정한다.

1. [데이터](data-and-preprocessing.md)와 [라우팅 비교](routing-experiments.md)의 미구현 차이를 해소하고 code/config를 commit한다.
2. 데이터·누수 audit, 공통 weight·입력 제한·후보 pool, run budget을 manifest로 동결한다.
3. CPU baseline과 작은 GPU smoke/pilot으로 runtime·실제 step·예상 비용을 확인한다.
4. validation에서 설정과 finalist를 고르고 lock을 만든다. test를 본 뒤 다시 선택하지 않는다.
5. 봉인 test와 RouterArena 평가, 독립 인지 수준 분석을 [평가 계약](evaluation.md)에 따라 수행한다.
6. 새 continual partition의 initial anchor를 학습하고 [재학습 실험](retraining-pipeline.md)을 실행한다.
7. 같은 고정 라우터로 [캐시 실험](semantic-cache.md)을 수행하고 시스템 총비용을 결합한다.

번호는 데이터 의존 순서다. 캐시 문항 작성·코드 구현처럼 독립적인 준비는 병행할 수 있다. 아직 실행하지 않은 실험의 예상 결과를 결과표에 채우지 않는다.

## 2. 로컬 진입점

설치·CLI 목록과 도움말 호출은 [ML 실행 진입점](../../ml/README.md)에서 관리한다. 실제 등록은 [pyproject.toml](../../ml/pyproject.toml), 명령 옵션은 해당 CLI parser를 따른다. 아래 문서화는 명령을 실행했다는 기록이 아니다.

운영 feedback을 재현할 때는 `escalation-feedback-preprocessing`의 `--training-data-strategy compact_rehearsal`을 명시한다. objective CLI의 `--method ordinal`은 일반 3-class CE인 R0이며 `regret`은 확장용이다. CLI에 존재하는 모든 method를 자동 실행하지 않는다.

실제 통합 build 시 `--routerarena-path`를 존재하는 봉인 artifact로 명시한다. 파일이 없는 채로 실행해 `overlap=0`이 출력되었다고 누수 검사가 완료된 것으로 처리하지 않는다. 원본 Pickle은 출처·checksum을 검증한 파일에 한하여 기존 `--trust-pickle` 계약을 따른다.

현재 source tree에는 최소 실험 전체를 실행하는 단일 명령이 없다. 아래 미구현 표의 후보·weight·평가 adapter가 정렬되기 전에는 기존 notebook을 그대로 실행하고 “최종 사양을 재현했다”고 표시하지 않는다.

## 3. Colab, runtime, checkpoint

학습 environment는 data-only environment와 분리한다. ModernBERT 운영 runtime은 [runtime contract](../../ml/config/modernbert_runtime_cu124_v1.json)와 [training lock](../../ml/requirements-training.lock)의 Python 3.11/3.12, PyTorch 2.5.1, Transformers 4.48.3, CUDA 12.4 기준이다. Colab의 당일 preinstalled 버전이 맞는지 확인하고, 실제 runtime fingerprint를 기록한다. 새 모델에 별도 dependency가 필요하면 environment를 version으로 분리한다.

| Archive 생성기 | 용도 |
|---|---|
| [build-router-model-comparison-colab-archive.sh](../../ml/scripts/build-router-model-comparison-colab-archive.sh) | 표현/모델 비교 source·config bundle |
| [build-routing-objective-colab-archive.sh](../../ml/scripts/build-routing-objective-colab-archive.sh) | 목표함수 비교 source·config bundle |
| [build-modernbert-colab-balanced-v2-archive.sh](../../ml/scripts/build-modernbert-colab-balanced-v2-archive.sh) | 기존 운영 v2 재현·호환 bundle |
| [build-modernbert-colab-smoke-archive.sh](../../ml/scripts/build-modernbert-colab-smoke-archive.sh) | 작은 운영 trainer smoke |

비교 code archive는 dataset 자체를 포함하지 않으므로 데이터 manifest·checksum과 별도로 전달한다. archive를 만들 때 들어간 실제 code commit과 tar SHA-256을 함께 기록한다. 노트북은 [표현/모델 폴더](../../ml/notebooks/router_model_comparison/)와 [목표함수 폴더](../../ml/notebooks/routing_objective_comparison/)를 사용하되 최종 정책 반영 여부를 먼저 확인한다.

운영 Colab driver는 다음 내구성 정책을 구현한다.

- 학습 I/O는 Colab 로컬 disk, durable 사본은 Drive에 둔다.
- 기본 50 optimizer step마다 로컬 checkpoint, 500 step마다 Drive 동기화; epoch 끝·최종 step도 동기화한다.
- model·optimizer·scheduler·scaler·epoch/batch cursor·RNG를 저장한다. `.partial-*` 복사 후 파일 checksum이 일치해야 완결 checkpoint로 승격한다.
- `_SUCCESS`와 manifest가 있는 checkpoint만 resume한다. data/config/runtime/step fingerprint가 다르면 재개를 거부한다.
- 기본 최근 checkpoint 2개 보존은 모든 epoch weight 보존과 다르다. 최종 비교·feedback 분석에 필요한 snapshot은 별도 경로로 복사해 보호한다.

재학습의 approved-anchor warm start는 위 interrupted-run resume와 다르다. 재학습은 weight만 재사용하고 optimizer를 새로 만든다. 기존 baseline notebook의 epoch별 RouterArena 관측을 최종 연구의 checkpoint 선택에 사용하지 않는다. 새로운 run directory와 version을 만들고 과거 artifact를 덮어쓰지 않는다.

## 4. GCP image·IR·resource 계약

현행 component source: [vertex_training.py](../../ml/kubeflow/components/vertex_training.py), [task_config.py](../../ml/kubeflow/task_config.py), [compiler](../../ml/kubeflow/compile_training_workflows.py).

| 항목 | 코드에서 확인한 값 |
|---|---|
| Region 기본값 | `us-west1`; 실제 project·bucket·service account는 배포 manifest로 관리 |
| KFP / Google components | 2.17.0 / 2.22.0 |
| Training worker | `n1-standard-4`, T4 1장, replica 1, pd-ssd 100GB |
| GPU timeout / 재시도 | `21600s`(6시간), 학습 task retry 없음, worker restart=false |
| CPU task retry | **2회**, 30초부터 exponential backoff; 전체 DAG가 retry 0인 것은 아님 |
| 동시 feedback | SQLite lease로 하나의 active run; cloud quota·다른 제출 경로도 별도 관리 |
| Pipeline caching | serving의 `PipelineJob(enable_caching=False)`; publish/approve는 task에서도 cache 비활성 |
| GPU 위치 | CustomJob worker pool에만 T4; KFP task에 `nvidia.com/gpu` 요청 없음 |

네 image는 data-processing, modernbert-training, modernbert-evaluation, model-promotion이다. [Dockerfile 폴더](../../ml/docker/)에서 `COPY ml/...`를 사용하므로 **build context는 저장소 root `.`**다. Mac에서도 target은 `linux/amd64`로 맞춘다. 학습 image는 TorchInductor를 위한 `build-essential`, `cc`, `c++`를 포함한다. CPU 추론 task라고 evaluation image까지 CPU-only base인 것은 아니다. 현재 evaluation Dockerfile은 CUDA 포함 PyTorch base이며 경량화는 별도 작업이다.

로컬 build 예:

```bash
./ml/scripts/build-data-processing-image.sh
./ml/scripts/build-modernbert-training-image.sh
```

cloud용은 네 image를 registry에 게시하고 각각 immutable `@sha256:…`를 확보한다. 실제 digest를 다음 네 환경변수에 넣어 IR을 컴파일한다.

```text
LLM_ROUTER_VERTEX_DATA_PROCESSING_IMAGE
LLM_ROUTER_VERTEX_TRAINING_IMAGE
LLM_ROUTER_VERTEX_EVALUATION_IMAGE
LLM_ROUTER_VERTEX_PROMOTION_IMAGE
```

```bash
.venv-kubeflow/bin/python -m ml.kubeflow.compile_training_workflows \
  --output-dir ml/artifacts/kubeflow/training-workflows-final
```

위 environment에는 먼저 [requirements-kubeflow.txt](../../ml/requirements-kubeflow.txt)를 설치한다. compiler 기본 URI의 `replace-project`/`replace-me`가 남아 있거나 mutable tag인 IR은 게시하지 않는다. 새 image를 push하면 IR도 **재컴파일·새 prefix로 업로드**하고 serving의 template URI를 갱신해야 새 코드가 실행된다. image 교체만으로 이미 컴파일한 IR은 바뀌지 않는다.

운영 active 학습 template은 `feedback-retraining-pipeline.yaml`, 기존 checkpoint 자격검증은 `model-qualification-pipeline.yaml`이다. 최초 학습 IR은 회귀검사용으로 보존한다. 연구 gate로 전환한 IR은 운영 prefix와 분리하고 config·image·IR SHA를 연결한다.

## 5. 배포 전후 확인과 비용

GPU 실행 전 billing credit 잔액·만료일, T4 quota, 실제 region 가용성, 실행당 상한과 총 제출 횟수를 확인한다. 과거 문서의 시간당 가격이나 무료 저장 한도를 현재 보장으로 재사용하지 않는다. billing budget alert는 지출 차단기가 아니다. 파일·image 수명, soft delete·object versioning, pipeline CPU 비용도 총비용에 포함한다.

| 주체 | 필요한 권한 경계 |
|---|---|
| Serving SA | model/config/IR 읽기, derived feedback·observability 경계 접근, Vertex 제출, Pipeline SA actAs, image pull, logging |
| Pipeline SA | pipeline artifact·model registry 작업, CustomJob 제출, Training SA actAs, image pull |
| Training SA | 입력 dataset/config/model 읽기, checkpoint/관측 output 쓰기, image pull |
| Spring | 내부 HTTP 호출 credential; 재학습 GCS 권한 불필요 |

Role 이름만 맞는지 보지 않고 resource 범위를 확인한다. Serving의 `Settings.retraining_configuration_errors()`로 dataset·config·pipeline template·root·allowed bucket·model provenance가 모두 설정되었는지 검사한다. [설정 소스](../../../LLM_Router-serving/src/llm_router_serving/config.py)를 기준으로 오래된 “환경변수 12개” 목록에 의존하지 않는다.

운영 VM 재구성 시 image pull 권한, image 크기에 맞는 boot disk, UID/GID 10001의 model/SQLite persistent volume 쓰기 권한을 확인한다. SQLite를 새로 만들면 version history를 잃으므로 backup과 migration을 먼저 검증한다. IP/hostname은 설정으로 전달한다.

E2E 완료는 유효 batch의 `202 → running → terminal gate 결과 → CURRENT → serving activation`과 동일 batch retry·load 실패·rollback 증거가 있을 때 표시한다. `/health=200`, Docker build, IR compile만으로 이 전체 경로를 통과했다고 기록하지 않는다. 작은 유효 batch라도 실제 `/escalate`는 GPU 비용을 발생시킬 수 있다.

GCS는 다음 논리 영역을 versioned prefix로 분리한다.

```text
datasets/       원본·전처리·split·feedback/replay
config/         label/training/evaluation/gate 계약
pipelines/      image digest가 고정된 IR
pipeline-root/  임시 component artifact
models/router-modernbert/<version>/candidate|approved/
models/router-modernbert/CURRENT.json
retraining-runs/<batch>/observability-v1/
research/<experiment-version>/  독립 dataset·prediction·snapshot·비용 ledger
```

임시 pipeline artifact에는 별도 lifecycle을 적용하되 durable model/research 결과를 함께 만료시키지 않는다. [lifecycle 예시](../../ml/config/gcs_pipeline_artifacts_lifecycle.example.json)는 실제 prefix와 복구 요구에 맞춰 검토한다. 모델은 current/previous/anchor, 연구 증거는 연구 retention 정책으로 보존한다.

## 6. 실험별 필수 산출물

| 묶음 | 반드시 보존할 내용 |
|---|---|
| Source/data | 원본 URL·revision·license 기록, 파일 크기·SHA-256, sample/group IDs, partition manifest, 제외 사유, source/domain/label 분포, 누수 audit |
| Config | 데이터/라벨/전처리/후보 pool 버전, tokenizer·model revision, seed, input 길이·truncation, sample weight·sampler, objective·selection·gate 규칙 |
| Code/runtime | 실제 code commit, dirty 여부와 재현 snapshot, archive·image·IR digest, dependency lock, CPU/GPU·메모리·runtime fingerprint |
| Training | 실제 optimizer step·token 수·시간, loss/validation, 선택 근거, checkpoint manifest·`_SUCCESS` |
| Routing 평가 | validation lock, 질문별 선택·확률·품질·비용, seed별 지표와 신뢰구간, baseline·oracle 정의 |
| Cognitive audit | judge model/prompt/version, blind 입력·출력, human 독립 주석, 판정 불가·불일치, 통제 문항쌍 |
| Feedback | arrival trace, 공개한 outcome, event/batch/canonical digest, accepted/excluded, anchor/version, gate/활성화, 실패 비용 |
| Cache | 후보·hit/miss·answer correctness, namespace/version, calibration freeze, 전체 요청 분모·미측정 범위, 실제 지연·순비용 |

최종 보고서는 실제 model prediction과 evaluator 입력을 별도로 보존한다. 누락 outcome이나 미판정 cache 응답을 방법별로 유리하게 분모에서 빼지 않는다. 새 API 생성은 공개 replay 결과와 구분하고 시행했다면 호출량·가격 시점·실제 청구를 기록한다.

## 7. 현재 구현과 최종 명세의 차이

2026-09-20 로컬 코드 확인 기준이다. 이 표가 유일한 실행 차이 목록이며 다른 문서의 설계 설명을 구현 완료 표시로 읽지 않는다.

| 범위 | 확인한 상태 | 최종 실행 전에 필요한 것 |
|---|---|---|
| RB+xRB 공통 데이터 | build·manifest·train-only threshold 선택 코드 있음 | 코드 들여쓰기·수식·필수 system 조건 보존, word-only 중복 key의 기호 충돌·task-ID 감사, 실제 결측 분모/solve-rate 재계산, metric registry 검증, 봉인 파일로 overlap 검사 실행 증거; 최종 provenance·manifest 재생성 |
| 공통 학습 weight | objective trainer는 sample weight 사용, TF-IDF는 동일 적용 안 됨 | 최종 초기 비교의 effective weight 1을 모든 방법에 구현·manifest 기록; 기존 metadata weight는 보존하고 운영 feedback 가중치와 구분 |
| 3개 실제 후보 비교 | objective 데이터·훈련·lock CLI 있음 | B의 source-only T1 probe와 실제 3-model outcome/비용 adapter·공통 후보 계약 검증; 집단 평균을 실제 호출로 해석하지 않음 |
| R0/R1/R2 | ModernBERT CE/pointwise/pairwise core 존재 | selection의 `METHODS`가 R3 포함 4개를 요구하므로 최소 3-arm lock 지원으로 조정; 최종 데이터·후보·가중치·선택 규칙을 맞춘 smoke와 실제 결과; R0 명칭 정정 |
| 표현/모델 | TF-IDF 코드·비교 notebook 있음 | BGE+kNN, 동일 예산·입력 길이·sample weight audit, 최종 최소 모델군 결과 |
| 공식 라우터 | 논문·adapter 설계 | RouterDC 공통 pool adapter, test를 validation으로 사용하는 코드 경로 방지, 고정 budget 재학습; GraphRouter 등은 확장 |
| Held-out 평가 | RouterArena evaluator와 objective test CLI 있음 | label/선택품질/인지 정렬 구분, global nAUC를 source별 curve/macro와 정렬, 퇴화 정규화 N/A, row bootstrap을 group 단위로 보완, robustness paired 지표·batch-1 latency·최종512 config |
| Calibration·선정 | raw ECE·Q_model와 기존 finalist lock 코드 | 공통 raw-confidence ECE, neural 보정 진단 분리, B 우선 전역 2개 finalist 반복 예산을 명시적으로 구현; 표현 축의 별도 top2 sweep 방지 |
| 기준선의 선택 정책 | 현재 random은 균등 1 seed, best-single은 validation 기반 | 최종 matched-random 100 seed와 train-only best-single로 정렬; 진단용 test action shuffle과 배포 가능한 val-ratio random 구분 |
| 인지 수준 보조 실험 | blind annotation·통제 문항쌍 설계 | 고정 judge prompt/model, 사람 검증, 평가 데이터 봉인; API 호출은 미실행 |
| 운영 feedback core | inline→GCS, prepare/merge, version/replay, CustomJob, gate/publish/watcher 코드 있음 | 최신 유효 batch E2E·실제 serving 활성화 증거 확보; code 존재와 cloud 실측 분리 |
| 연구 feedback | F0/F1/F2·6기회 사양 | fresh continual split, frozen logger·shared batch harness, F1/F2 공통 shuffle·실제 step 일치·공통 validation48 |
| 연구 승격 gate | 운영 DAG는 RouterArena gate | D_gate schema·metric adapter와 별도 root; D_dev로 고른 임계값 freeze; 통합 base contract 새 version |
| Canonical uniqueness | whole-batch digest 예약/commit 구현 | merge 후 nonempty·novel 정보 검증을 연구 harness에 추가; 중복 mode가 이를 보장한다고 가정하지 않음 |
| 캐시 | BGE/선택 rerank serving 경계, 별도 실험 사양 | exact replay·semantic reuse trace, human/정답 검증, 비용·지연 ledger; reranker default off 유지 |
| Colab→운영 인수 | checksum checkpoint와 package CLI 있음 | 통합 데이터 초기 모델의 검증된 인수 manifest·gate·baseline; 자동 importer의 검증 범위 확인 |
| 운영 복구·효율 | SQLite lease, atomic CURRENT, load-before-swap, rollback 코드 | state 유실 복구 migration, 상태 polling 부하/TTL, evaluation CPU image 경량화는 별도 후속 |

실행 중 각 행을 갱신할 때 증거 artifact URI·commit·검증일을 남긴다. 과거 체크리스트의 `[x]`를 새 실험 완료로 승계하지 않는다.

## 8. 과거 결과에서 보존할 사실

아래는 **당시 기록된 결과 요약**이며 현재 최종 데이터/실험의 결과가 아니다. 원문은 통합 전 보존 commit `fa09cb8`에 있다. 원래 수치를 다시 계산한 것이 아니며, live GCP 상태도 이번 통합에서 재조회하지 않았다.

| 시점·run | 남길 사실 |
|---|---|
| 2026-08 v1 baseline | epoch 1, 3,476 step에서 중단; final-model 없음. 마지막 RB validation macro-F1 0.4456, medium F1 0; RouterArena macro-F1 0.3011 |
| 2026-08-15 v2 balanced 기록 | 18,768 train, validation/test 각 3,218; T4 FP16·8,192 token·3 epoch·7,038 step, 약 5시간48분. RB val macro-F1 epoch2 0.4505 → epoch3 0.4352; RouterArena 0.3293 → 0.3212 |
| 당시 TF-IDF 진단 | 같은 v2 validation macro-F1 0.4198. ModernBERT와의 간격은 관측이며 “라벨이 성능 상한”, “출처만 학습”이라는 인과 증명은 아님 |
| 2026-08-16 최초 package | `router-modernbert-base-routerbench-20260814-45dcc6d`, v2 epoch3. 당시 baseline·정식 gate 규칙 부재로 **gate를 생략**한 최초 형식 검증 모델 |
| 최초 package의 재현 한계 | epoch2 weight는 최근 checkpoint 2개 retention으로 삭제됨. `code_commit=45dcc6d` 당시 실제 학습 코드 일부가 untracked/미커밋이어서 해당 SHA만으로 완전 재현 불가. config/data checksum 보존과 별개인 provenance 공백 |
| 2026-08-16 serving 기록 | GCE `llm-router-serving`, `us-west1-a`에서 container 및 Spring 측 호출 확인 기록. 현재 health·image·CURRENT 상태로 단정하지 않음 |
| 2026-08-17 GPU smoke 기록 | C compiler 누락을 training Dockerfile의 build-essential로 수정한 뒤 CUDA smoke 성공 기록. 최신 feedback 전체 E2E 성능 검증과 다름 |
| 2026-09-12~16 serving 기록 | inline/API/version-policy/관측 계약과 GCP 교체 배포·OpenAPI·모델 동기화 확인 기록. 최신 유효 feedback의 GPU E2E 검증은 별도 항목 |

과거 RouterArena 결과를 보고 연구 방향을 논의한 이력이 있으므로 완전히 한 번도 본 적 없는 벤치마크라고 표현하지 않는다. 새 최종 run에서는 이를 tuning/gate에 사용하지 않고, 추가 외부 데이터와 blind 인지 audit로 보강한다. 과거 v2의 tier label과 현재 solve-rate label도 서로 다른 계약이다.

## 9. Git 보존과 원본 artifact 복구

**Git이 과거 실험의 모든 원본 로그·가중치를 보관하는 것은 아니다.** 통합 전 `docs/ml`의 문서·리포트 22개는 commit `fa09cb8`로 보존했다. `ml/data`, `ml/artifacts`의 원본 데이터·Parquet·대형 로그·checkpoint는 ignore 대상이며 Drive/GCS/로컬 보관 상태를 따로 확인해야 한다. 문서 삭제는 원본 artifact나 소스를 삭제하는 작업이 아니다.

과거 문서의 조회·전체 추출 방법은 [보존 안내](README.md#5-이전-문서실험-기록의-보존)에만 둔다.

실험 재현에 필요한 원본 artifact가 없다면 문서 수치만으로 재생성된 것처럼 표시하지 않는다. 다음 run부터는 실행 전에 code/config commit과 artifact index를 함께 봉인한다. 모델·데이터·실험의 최신 상태와 결정은 memorize에도 기록하되, 대형 결과물 자체는 versioned artifact storage가 보관한다.
