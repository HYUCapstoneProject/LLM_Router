# ML 학습·평가·배포 작업 총괄 체크리스트

기준일: 2026-08-10
담당 영역: 아키텍처의 오른쪽 **오프라인 학습·평가 경로**와 왼쪽 온라인 경로의 ML 연동 경계
작업 브랜치: `feat/data-labeling-training`

## 1. 이 문서의 목적

이 문서는 RouterBench 원본 데이터에서 시작해 승인된 ModernBERT 모델을 온라인
라우터에 전달하고, 운영 중 수집된 에스컬레이션 정정 라벨을 다음 재학습에 다시 넣는
과정을 한 곳에서 관리한다.

체크 표시는 다음 의미로 사용한다.

- `[x]`: 코드·산출물·테스트로 확인된 완료 항목
- `[ ]`: 아직 완료되지 않은 항목
- 완료 여부는 구현 존재만이 아니라 재현 가능한 실행 명령과 검증 결과까지 포함한다.

### 1-1. 진행 상황 점검 규칙

앞으로 ML 작업을 시작하거나 마칠 때 이 문서를 기준으로 진행 상황을 점검한다.

- 작업 시작 전 `7. 바로 이어서 할 순서`와 해당 세부 체크리스트를 확인한다.
- 구현만 존재하는 상태는 완료로 표시하지 않는다. 로컬 테스트, container smoke 또는
  pipeline compile 중 해당 단계에 맞는 검증 증거가 있어야 `[x]`로 바꾼다.
- 체크 상태를 바꿀 때 관련 명령, artifact version, commit 또는 실행 결과를 가까운
  명세 문서나 commit에 함께 남긴다.
- 클러스터에서만 검증할 수 있는 항목은 로컬 구현이 끝나도 `[ ]`로 유지하고,
  `서버 확보 후` 항목에서 별도로 확인한다.
- 온라인 팀과 합의가 필요한 계약은
  `docs/online-ml-handoff-contract.md`를 먼저 갱신한 뒤 코드에 반영한다.

## 2. 담당 범위와 경계

### 2-1. 직접 담당

- RouterBench 다운로드, canonicalization, capability tier 산정과 3-class 재라벨링
- RouterBench 학습 projection, 중복 제거, group-aware split과 데이터 품질 검증
- ModernBERT 로컬 smoke 학습, 전체 학습, checkpoint 패키징
- RouterArena 일반화 평가와 RouterBench holdout 평가
- Kubeflow CPU/GPU component와 pipeline 구성
- GCS 데이터·모델·평가 artifact 저장 규칙
- MLMD 실행·데이터·모델 계보
- 후보 모델 배포 게이트와 승인 모델 manifest 생성
- 에스컬레이션 정정 라벨의 검증·학습 데이터 변환·재학습 입력 결합

### 2-2. 왼쪽 온라인 경로와 공동 담당하는 접점

| 접점 | 이 작업에서 책임질 것 | 온라인 경로에서 책임질 것 |
| --- | --- | --- |
| 모델 전달 | checkpoint 패키지, checksum, `id2label`, model manifest | 승인 버전 다운로드, 프로세스 교체와 rollback |
| Router API | 요청/응답 schema, Python reference predictor, contract test | FastAPI 배포, Spring 호출, timeout/fallback |
| tier 매핑 | `easy/medium/hard` 출력과 권장 route mapping | 실제 Claude 모델 ID와 운영 정책 적용 |
| cache threshold | ML 학습/Kubeflow 범위에서 제외 | QQP 실험, bge 임베딩, pgvector 조회·저장과 threshold 적용 |
| 정정 라벨 | event schema 검증, dedup, dataset version 생성 | 사용자 에스컬레이션 수집과 DB 저장 |
| 관측성 | model/data version과 평가 metric 제공 | request id, latency, token, cost와 오류 로그 |

### 2-3. 직접 구현하지 않는 온라인 기능

- React 채팅 화면과 에스컬레이션 UI
- Spring `/api/chat`의 전체 비즈니스 흐름
- Anthropic API 호출, retry와 fallback 구현
- PostgreSQL/pgvector 온라인 CRUD와 인덱스 운영

단, 위 기능이 ML 산출물을 올바르게 소비하는지 확인할 contract fixture와 통합 smoke에는
참여한다.

## 3. 전체 흐름

```text
RouterBench 원본
  -> canonicalize / capability tier / relabel
  -> training projection / duplicate group split
  -> local ModernBERT smoke train
  -> GKE T4 full train
  -> RouterBench holdout + RouterArena evaluation
  -> deploy gate
  -> GCS approved model manifest
  -> online FastAPI Router가 승인 모델 로드

온라인 escalation event
  -> PostgreSQL export
  -> validation / dedup / corrected-label dataset
  -> 다음 Kubeflow 재학습 입력
```

Kubeflow는 데이터 의미론을 다시 구현하지 않는다. 저장소의 같은 Python 함수를
component에서 호출하고, Kubeflow는 artifact 전달, cache, retry, resource, MLMD 계보와
단계 실행을 담당한다.

### 3-1. 현재 제약과 구현 원칙

에스컬레이션 원본 schema와 온라인 배포 계약은 팀 합의 전이다. 이 제약은
로컬 component 작성과 pipeline 컴파일을 막지 않는다. 다음 세 경계를 유지한다.

1. Kubeflow component는 DB 컬럼을 직접 입력받지 않고 `Dataset`/`Model` 같은 파일
   artifact와 scalar parameter만 주고받는다.
2. 온라인 팀의 원본 export는 schema adapter에서 ML 내부 표준 schema로 변환한다.
   원본 컬럼이 바뀌면 adapter만 바꾸고 전처리·학습 DAG는 유지한다.
3. 클러스터 의존 값인 project, region, bucket, namespace, service account, image URI와
   GPU node selector는 코드에 고정하지 않고 parameter나 배포 설정으로 둔다.

ML 내부 학습 표준 schema는 다음 필드로 유지한다.

```text
sample_id, model_input_text, difficulty_label
sample_weight, split, duplicate_group_id
```

현재 작업 가능 범위는 다음과 같이 구분한다.

| 구분 | 지금 진행할 작업 | 완료 판단 |
| --- | --- | --- |
| 서버 없이 완성 가능 | CPU 데이터 image, component wrapper, 초기 pipeline, artifact/manifest 계약 | 로컬 container smoke, 단위 테스트, IR YAML compile |
| schema adapter 골격 | 합성 feedback fixture, validation, dedup, merge, 재학습 pipeline 골격 | fixture 기반 계약 테스트와 IR YAML compile |
| 팀 합의 후 확정 | 실제 feedback row mapping, PII/retention, trigger owner와 누적 임계값 | 양 팀 contract fixture 통과 |
| 서버 확보 후 검증 | Registry/GCS/IAM, CPU·GPU Pod, cache, MLMD, cloud smoke와 전체 학습 | Kubeflow run과 저장된 artifact 확인 |

### 3-2. 2026-08-10 클라우드 실측과 회의 전 작업 경계

인증 계정 `myday2day@hanyang.ac.kr`에서 접근 가능한 Google Cloud project는
`project-763753da-fa8f-4afd-b31`(`HYUCapstone`) 하나다. 로컬 전용 Google Cloud CLI와
config는 Git에서 제외한 `.tools/` 아래에 두었다.

사용 가능하다는 구두 상태와 별개로 API로 직접 확인한 현재 상태는 다음과 같다.

| 항목 | 실측 상태 | 다음 조치 |
| --- | --- | --- |
| GCS bucket | project 내 bucket 0개 | bucket 이름·region 확정 후 생성 |
| GKE | Kubernetes Engine API 비활성 | 비용·region·cluster 이름 승인 후 API 활성화와 생성 |
| Artifact Registry | API 비활성, repository 없음 | repository 이름·region 확정 후 생성 |
| kubectl context | 없음 | cluster 생성과 credential 발급 후 설정 |

따라서 현재 project는 **계정과 project만 준비된 상태**이며, GCS bucket과 GKE cluster가
실제로 provision된 상태는 아니다. API 활성화, cluster·bucket·registry 생성은 과금과
외부 상태 변경을 수반하므로 회의 전 임의로 수행하지 않는다.

회의 전 독립적으로 완료할 목록:

- [x] QQP cache threshold 구현을 ML/Kubeflow 범위에서 제외한다.
- [x] CPU 데이터 image v2의 RouterBench·RouterArena·feedback 통합 smoke를 통과한다.
- [x] RouterArena 예측 그룹 분포 metric과 SVG 시각화를 구현한다.
- [x] 모델 package manifest 생성기와 label/file/checksum 검증기를 구현한다.
- [x] 절대 metric과 기준 모델 대비 delta를 검사하는 deploy gate core를 구현한다.
- [x] 합성 fixture 기반 전체 단위·계약 테스트를 통과한다.
- [x] 평가·패키징·gate 실행용 image와 Kubeflow component를 작성한다.
- [x] 초기학습·재학습과 독립 자격검증 pipeline IR에 평가·package·gate 단계를
  연결한다.

회의 또는 명시적 인프라 승인 뒤 확정할 목록:

- [ ] GCS bucket, region, Artifact Registry repository와 GKE cluster 이름
- [ ] 비용 예산, GPU quota/node pool, namespace와 Workload Identity service account
- [ ] feedback 운영 export schema, PII/retention과 재학습 trigger 주체
- [ ] 온라인 Router의 모델 package 다운로드, health check, 전환과 rollback 방식
- [ ] 온라인 API label 명칭, 실제 모델 route mapping, timeout/fallback
- [ ] 운영 기준 모델과 deploy gate 수치 임계값

## 4. 현재 완료 기준선

### 4-1. RouterBench 라벨링

- [x] 공식 0-shot/5-shot 파일 URL, 크기와 SHA-256을 고정했다.
- [x] checksum 검증과 중단 재개 다운로드를 구현했다.
- [x] 신뢰 확인 없이는 Pickle을 읽지 않도록 `--trust-pickle` 경계를 구현했다.
- [x] 모델 performance 컬럼과 비용 컬럼을 식별한다.
- [x] 0-shot/5-shot을 `sample_id` 기준으로 canonicalize한다.
- [x] 데이터셋별 source 우선순위와 핵심 eval 범위를 적용한다.
- [x] few-shot 예시와 실제 target question을 분리한다.
- [x] 11개 모델의 관측 성능으로 low/medium/high capability tier를 만든다.
- [x] tier consensus로 `easy/medium/hard`, reason, confidence, weight를 만든다.
- [x] seed 42의 100행 smoke를 생성했다.
- [x] smoke 결과는 easy 42, medium 15, hard 43이다.
- [x] 핵심 eval 전체 34,778행을 라벨링했다.
- [x] 전체 결과는 easy 12,765, medium 5,348, hard 16,665이다.

### 4-2. RouterBench 학습 전처리

- [x] RouterBench wrapper를 제거한 `model_input_text`를 생성한다.
- [x] exact/near duplicate를 `duplicate_group_id`로 묶는다.
- [x] 같은 duplicate group이 split을 넘지 않도록 한다.
- [x] difficulty×domain을 고려한 train/validation/test split을 구현했다.
- [x] confidence weight와 RAG 포함 여부를 config로 관리한다.
- [x] `training.parquet`과 분석용 `metadata.parquet`을 분리한다.
- [x] success rate, cost, response가 학습 feature로 들어가지 않도록 검증한다.
- [x] 전체 34,778행을 전처리해 34,764행의 학습 projection을 만들었다.
- [x] 전체 split은 train 27,808, validation 3,478, test 3,478이다.

### 4-3. RouterArena 전처리·평가 기반

- [x] 공식 full/sub_10/robustness revision과 checksum을 고정했다.
- [x] 공개 `Difficulty`의 `easy/medium/hard`를 변환 없이 검증한다.
- [x] Context, Question, Options를 모델 입력으로 조립하고 Answer를 제외한다.
- [x] robustness를 full과 `Global Index`로 결합한다.
- [x] 빈 질문, 거절 placeholder, 첨부 정답과 source duplicate를 제외한다.
- [x] RouterBench exact training overlap 검사를 구현했다.
- [x] 전체 학습 데이터와 정확히 겹치는 22행을 제외한 full 8,375행 artifact를 만들었다.
- [x] generalization/robustness 100행 smoke artifact를 만들었다.
- [x] Hugging Face 3-class checkpoint predictor와 label-order 검증을 구현했다.
- [x] accuracy, macro-F1, confusion matrix와 routing-direction proxy를 구현했다.
- [x] 전처리·평가 계약에서 사용하지 않는 중간 Bloom 명칭을 제거했다.

### 4-4. 검증과 저장소 위생

- [x] 현재 ML 단위·통합 테스트 40개가 통과한다.
- [x] `ml/data/`, `ml/artifacts/`, virtualenv, bytecode와 egg-info를 Git에서 제외한다.
- [x] training/evaluation/shared 소스 디렉터리를 분리했다.
- [x] 실행 방법과 데이터 계약을 `ml/README.md` 및 세부 명세에 기록했다.

## 5. 다음 구현 체크리스트

### A. 현재 기준선 커밋

- [x] 브랜치가 `feat/data-labeling-training`인지 확인한다.
- [x] 원본 Pickle, Parquet artifact, virtualenv, bytecode가 staging 대상이 아님을 확인한다.
- [x] Python compile과 당시 31개 테스트를 통과한다.
- [x] staged diff의 secret, 대용량 바이너리, 임시 경로를 마지막으로 검사한다.
- [x] commit message를 `[FEAT] RouterBench 전체 라벨링 및 전처리 기반 구현`으로 확정한다.
- [x] 첫 기준선 commit hash `b467dcb`를 기록한다.

### B. 전체 RouterBench 라벨링·전처리

- [x] smoke가 아닌 전체 데이터용 `label` CLI를 추가한다.
- [x] 출력 경로에 dataset/tier/rule version을 포함한다.
- [x] 전체 핵심 eval 행을 라벨링하고 클래스·도메인·reason·confidence 분포를 검토한다.
- [ ] continuous threshold 0.60/0.70/0.80 sensitivity를 실행한다.
- [ ] eval별 acceptance threshold override를 지원한다.
- [ ] RAG 포함/제외 두 설정을 비교하고 한 설정을 baseline으로 확정한다.
- [x] 전체 라벨 산출물에 전처리를 실행한다.
- [ ] split별 행 수와 difficulty×domain 편차 허용 범위를 자동 검증한다.
- [x] 전체 RouterBench와 RouterArena의 exact overlap 검사를 다시 실행한다.
- [ ] `model_tiers.json`과 metrics에 생성 코드 commit을 기록한다.

완료 산출물:

```text
routerbench-labeled/<data-version>__<tier-version>__<rule-version>/labeled.parquet
routerbench-labeled/<data-version>__<tier-version>__<rule-version>/model_tiers.json
routerbench-labeled/<data-version>__<tier-version>__<rule-version>/metrics.json
routerbench-training/<data-version>__<rule-version>__<preprocess-version>/training.parquet
routerbench-training/<data-version>__<rule-version>__<preprocess-version>/metadata.parquet
routerbench-training/<data-version>__<rule-version>__<preprocess-version>/metrics.json
```

### C. ModernBERT 로컬 학습 smoke

- [x] 로컬과 Kubeflow가 같은 학습 코어·CLI를 사용한다는 결정과 근거를 문서화한다.
- [x] base model identifier와 revision을 고정한다.
- [x] `label2id={easy:0, medium:1, hard:2}`와 역방향 `id2label`을 명시한다.
- [x] tokenizer max length와 truncation 정책을 config로 분리한다.
- [x] sample weight를 loss에 반영하는 trainer를 구현한다.
- [x] seed, epoch, batch size, learning rate, warmup, weight decay를 config에 기록한다.
- [x] Colab CUDA에서 100행 artifact의 제한된 forward/backward를 통과한다.
- [x] train/validation loss와 3-class metric을 생성한다.
- [x] `save_pretrained()` checkpoint와 tokenizer를 저장한다.
- [x] checkpoint 재로딩 후 동일 입력 예측 smoke를 통과한다.
- [x] `config.json`의 `id2label`이 `0=easy, 1=medium, 2=hard`인지 자동 검증한다.
- [x] Docker·Colab·로컬 공통 runtime 버전 계약과 fail-fast 검사를 구현한다.
- [x] runtime 버전을 `metrics.json`에 기록한다.

### D. 전체 학습과 평가

- [x] T4 전체 학습용 8,192 token/FP16/gradient-checkpointing config를 만든다.
- [ ] 전체 train split으로 ModernBERT base baseline을 학습한다.
- [ ] validation으로 early stopping과 최종 checkpoint를 선택한다.
- [ ] RouterBench test에서 accuracy, macro-F1, under/over-routing과 class metric을 계산한다.
- [x] RouterArena full에서 예측 그룹별 공식 라벨 count와 행 정규화 비율을 계산하는
  evaluator를 구현한다.
- [x] RouterArena 전체 분포 대비 예측 그룹별 라벨 증감(%p)을 계산하는 evaluator를
  구현한다.
- [x] RouterArena 100% 누적 막대그래프와 정규화 히트맵을 SVG로 생성한다.
- [ ] RouterArena robustness에서 원본/변형 예측 일관성을 계산한다.
- [ ] domain/source dataset별 성능과 truncation 편향을 분석한다.
- [ ] 실패·저신뢰·severe mismatch 표본을 수동 검토 목록으로 저장한다.
- [ ] Claude Haiku/Sonnet/Opus 소규모 표본으로 RouterBench proxy 라벨을 검증한다.

### E. 범위 외: QQP cache threshold 평가

QQP 실험과 온라인 threshold 적용은 `졸업프로젝트 개요.md`의 `서빙·캐시 축`
담당이다. ML 축에서는 QQP 다운로드, 임베딩, threshold sweep, Kubeflow
component와 threshold artifact를 구현하지 않는다. 최종 보고서에서 온라인 팀이
제공한 cache 결과를 RouterBench/RouterArena 라우터 결과와 섞지 않도록 구분해
해석하는 것만 공동 확인한다.

### F. 모델 artifact와 온라인 Router 연동

- [x] 모델 패키지 `model-package-v1` manifest schema를 코드로 고정한다.
- [x] manifest에 model/data/preprocessing/label/evaluation version과 파일별 checksum을
  넣는다.
- [x] checkpoint, tokenizer, config, metrics, manifest의 필수 파일 검증기를 만든다.
- [ ] Python reference predictor가 package directory 하나만 받아 예측하도록 만든다.
- [ ] Router 요청/응답 계약을 문서화한다.
- [ ] FastAPI Router 최소 endpoint 또는 contract fixture를 제공한다.
- [ ] Spring의 LIGHT/MEDIUM/HEAVY와 ML의 easy/medium/hard 중 하나로 이름을 통일한다.
- [ ] `easy→haiku`, `medium→sonnet`, `hard→opus` 매핑을 한 설정 위치에서 관리한다.
- [ ] 모델 로드 실패, timeout, unknown label에 대한 fallback 계약을 합의한다.
- [ ] Spring stub과 실제 predictor 사이 contract integration test를 통과한다.

권장 모델 manifest 최소 필드:

```text
model_version, base_model, files
dataset_version, preprocessing_version, label_rule_version
id2label, tokenizer_max_length
training_version, evaluation_version, gate_status, created_at, code_commit
```

실제 `model-package-v1`은 단일 `checkpoint_sha256` 대신 `files` 객체에
`config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_config.json`,
`metrics.json`, weight 파일 각각의 SHA-256을 기록한다. manifest 파일명은
`model_manifest.json`으로 통일한다.

권장 Router 계약 최소 필드:

```text
request:  request_id, question
response: label, confidence, model_version, latency_ms
```

### G. Kubeflow component와 pipeline

- [x] component 입출력을 파일 artifact와 scalar parameter로 구분한다.
- [x] 공통 `modernbert-train` entrypoint를 가진 T4용 linux/amd64 image를 정의한다.
- [x] GPU image에서 CPU 라벨링·평가 코드와 CLI를 제외한다.
- [x] `llm-router-modernbert-training:cu124-v1` image를 로컬에 빌드하고 설정을 검사한다.
- [x] 완성 image에서 `modernbert-train --help` runtime smoke를 통과한다.
- [ ] RouterBench ingest component를 만든다.
- [ ] canonicalize/relabel CPU component를 만든다.
- [ ] training-preprocess/split CPU component를 만든다.
- [x] ModernBERT tokenizer/train 단일 GPU component를 만든다.
- [ ] RouterBench evaluate CPU/GPU component를 만든다.
- [x] RouterArena full 평가용 단일 GPU component와 전용 image를 만든다.
- [ ] RouterArena robustness 원본/변형 일관성 평가 component를 만든다.
- [x] candidate model package component와 전용 CPU image를 만든다.
- [x] deploy-gate Kubeflow component를 만들고 실패 시 task를 차단한다.
- [ ] 입력 checksum+parameter 기반 cache key가 동작하는지 검증한다.
- [x] CPU/memory/GPU request와 limit, cache와 retry를 component task에 명시한다.
- [ ] 실제 cluster 관측값으로 timeout을 확정한다.
- [ ] local component test와 pipeline compile test를 CI에서 실행한다.

위 RouterBench CPU component 세 항목은 서버에서 초기 데이터를 재생성해야 할 때만
구현하는 선택 항목이다. 현재 기본 초기 학습 pipeline은 로컬 검증·전처리가 끝난
`training.parquet`를 Importer로 받아 GPU 학습부터 시작한다. 반면 feedback `prepare`와
`merge`는 재학습 때 서버에서 실행되므로 별도 CPU component로 구현한다.

서버 없이 먼저 완료할 항목:

- [x] 초기 로컬 처리와 피드백 재학습 명령을 포함하는 범용 CPU 데이터 image v2를
  빌드한다.
- [x] CPU image v2에서 RouterBench 100행, RouterArena 3행과 합성 feedback
  prepare/merge smoke를 통과한다.
- [x] KFP SDK를 `2.17.0`으로 lock하고 KFP v2 Artifact schema로 컴파일한다.
- [x] 각 component가 받는 Artifact path와 scalar parameter 계약을 문서화한다.
- [x] 로컬 image tag를 parameter로 받는 CPU/GPU component wrapper를 작성한다.
- [x] GPU image tag/Artifact Registry URI를 parameter로 받는 학습 wrapper를 작성한다.
- [x] GPU component와 단일 학습 pipeline IR YAML compile test를 통과한다.
- [x] 초기 학습 pipeline을 IR YAML로 컴파일하고 정적 타입 검사를 통과한다.
- [x] 합성 feedback fixture 계약을 통과한 prepare/merge를 연결한 재학습 pipeline을 IR
  YAML로 컴파일한다.
- [x] `llm-router-modernbert-evaluation:cu124-v1` image를 linux/amd64·non-root로
  빌드하고 CLI smoke를 통과한다.
- [x] `llm-router-model-promotion:py312-v1` CPU image를 linux/amd64·non-root로
  빌드하고 CLI smoke를 통과한다.
- [x] 초기학습·재학습 pipeline 뒤에 RouterArena 평가, candidate package와 deploy gate를
  연결하고 IR YAML compile test를 통과한다.
- [x] 이미 생성된 checkpoint만 다시 자격검증할 독립 qualification pipeline을
  컴파일한다.
- [ ] 실제 DB, GCS와 Kubeflow endpoint 없이 실행되는 계약 테스트를 CI에 추가한다.

서버 확보 후에만 완료할 항목:

- [ ] 실제 registry image URI와 immutable digest를 pipeline에 주입한다.
- [ ] KFP backend에 pipeline을 제출하고 CPU/GPU Pod 분리를 확인한다.
- [ ] GCS Artifact 전달과 Workload Identity 권한을 확인한다.
- [ ] cache hit/miss, retry, timeout과 MLMD lineage를 실제 실행에서 검증한다.
- [ ] 100행 cloud smoke를 통과한 뒤 전체 학습 실행을 승인한다.

권장 DAG:

```text
ingest
  -> canonicalize
  -> relabel
  -> training_preprocess
  -> train
  -> [routerbench_eval, routerarena_eval]
  -> deploy_gate
  -> publish_approved_model
```

### H. GCP·GKE·GCS·MLMD

- [ ] 월 예산 알림과 GPU 비용 실행 승인 절차를 확인한다.
- [ ] 실행 region과 T4 1개, N1 CPU 4~8 quota를 확인한다.
- [ ] GKE CPU node pool과 scale-to-zero 가능한 T4 node pool을 분리한다.
- [ ] Workload Identity와 최소 권한 service account를 구성한다.
- [ ] GCS bucket과 dataset/model/evaluation prefix 규칙을 확정한다.
- [ ] pipeline root와 component image registry를 확정한다.
- [ ] secret을 이미지·소스·pipeline parameter에 직접 넣지 않는다.
- [ ] MLMD에 dataset, execution, checkpoint, evaluation, gate 관계를 기록한다.
- [ ] Spot T4 중단 시 checkpoint resume와 retry가 동작하는지 검증한다.
- [ ] 첫 100행 cloud smoke 후에만 전체 학습을 실행한다.

권장 GCS 구조:

```text
gs://<bucket>/datasets/routerbench/<version>/
gs://<bucket>/datasets/routerarena/<version>/
gs://<bucket>/datasets/feedback/<batch-id>/
gs://<bucket>/datasets/retraining/<dataset-version>/
gs://<bucket>/models/router-modernbert/<model-version>/candidate/
gs://<bucket>/models/router-modernbert/<model-version>/approved/
gs://<bucket>/evaluations/<model-version>/<evaluation-version>/
```

### I. 배포 게이트와 모델 승격

- [ ] 현재 운영 모델과 후보 모델의 동일 test artifact 비교를 강제한다.
- [ ] RouterBench macro-F1 최소 기준을 정한다.
- [ ] under-routing 또는 severe mismatch 상한을 정한다.
- [ ] RouterArena alignment 악화 허용 범위를 정한다.
- [ ] latency, artifact 무결성, label order 검증을 필수 gate로 둔다.
- [ ] gate 결과에 metric, threshold, pass/fail reason을 저장한다.
- [x] 절대 최소·최대 metric과 기준 모델 대비 delta를 규칙 JSON으로 검사하고 각
  actual/threshold/pass-fail을 보존하는 gate core를 구현한다.
- [ ] 통과한 후보만 `approved/` prefix와 approved manifest로 승격한다.
- [ ] 온라인 Router가 manifest 기준으로 새 모델을 로드하도록 한다.
- [ ] 이전 approved model을 보존하고 rollback 절차를 검증한다.

### J. 에스컬레이션 정정 라벨 재학습

- [x] 온라인/ML 책임 경계와 제안 계약을 `docs/online-ml-handoff-contract.md`에 분리했다.
- [ ] 온라인 팀과 escalation event/export manifest schema를 확정한다.
- [ ] 최소 필드에 event/request id, question, initial/final tier, 만족 여부와 시간을 둔다.
- [ ] batch id, schema version, URI, checksum과 export watermark를 합의한다.
- [ ] 개인정보·민감정보 제거 정책과 retention 기간을 정한다.
- [x] v0.1 합성 event와 checksum manifest fixture를 저장소에 추가한다.
- [x] schema version, checksum, 행 수와 필수 필드를 검사하는 feedback validator를 만든다.
- [x] 원본 export를 ML 내부 표준 6필드 schema로 바꾸는 versioned adapter를 만든다.
- [ ] 중복 event와 반복 질문을 group 단위로 묶는다.
- [ ] 최종 만족 tier를 정정 라벨로 사용하는 조건을 명시한다.
- [x] 합성 계약 기준의 중복 event, 반복 질문 집계와 label 충돌 제외를 구현한다.
- [x] validation/test 정확 일치 누수를 막고 충돌하는 기존 train 행만 교체하는 merge를
  구현한다.
- [x] 모순·저신뢰 event를 자동 학습 projection에서 제외하고 metadata에 사유를 남긴다.
- [ ] RouterBench 원본과 정정 라벨의 sampling/weight 정책을 정한다.
- [x] 정정 라벨 dataset version, source checksum과 병합 lineage manifest를 기록한다.
- [ ] 누적 N건 또는 성능 drift 기반 재학습 trigger를 실험값으로 관리한다.
- [x] 재학습 pipeline에도 초기학습과 같은 평가·패키지·배포 gate DAG를 연결한다.
- [ ] 실제 재학습 모델로 gate 실행을 통과시킨다.

권장 escalation event 최소 필드:

```text
event_id, request_id, conversation_id, question
initial_label, initial_model, final_label, final_model
user_satisfied, escalation_count, created_at
router_model_version
```

Kubeflow는 PostgreSQL table을 직접 읽지 않는다. 온라인 팀이 versioned batch를
export하고, ML schema adapter가 이를 canonical feedback dataset으로 변환한다. 실제
schema가 확정되기 전에는 합성 fixture로 validator, dedup, merge와 pipeline 연결만
검증한다.

### K. 실험 결과와 최종 제출

- [ ] 모든 표·그래프에 model/data/code/evaluation version을 표시한다.
- [ ] always-Opus 대비 비용·품질 결과를 계산한다.
- [ ] cache 적용 전후 비용과 latency를 분리해 보고한다.
- [ ] RouterBench·RouterArena 라우터 결과와 온라인 팀의 cache 결과를
  서로 다른 의미로 해석한다.
- [ ] 반복 실행 seed와 평균·분산 또는 신뢰구간을 기록한다.
- [ ] 최종 model card와 data card를 작성한다.
- [ ] 로컬 smoke, cloud train, 평가, rollback runbook을 작성한다.
- [ ] 보고서와 발표자료의 최종 metric artifact를 freeze한다.

## 6. 마일스톤별 완료 조건

| 마일스톤 | 완료 조건 |
| --- | --- |
| M1 Data Baseline | 전체 RouterBench 라벨·split artifact와 품질 검사가 승인됨 |
| M2 Local Model | 로컬 smoke checkpoint를 재로딩해 3-class 예측 성공 |
| M3 Evaluation | RouterBench holdout과 RouterArena 분포 분석 결과 생성 |
| M4 Cloud Train | GKE T4 학습과 MLMD/GCS 계보가 재현됨 |
| M5 Promotion | 후보 모델이 gate를 통과해 approved manifest로 승격됨 |
| M6 Online Link | Spring→FastAPI→ModernBERT 계약 smoke와 rollback 통과 |
| M7 Retraining | escalation export가 versioned dataset으로 재학습에 투입됨 |
| M8 Results Freeze | 비용·품질·cache·latency 결과와 실행 문서 확정 |

## 7. 바로 이어서 할 순서

1. [ ] 초기 처리와 feedback 재학습 CLI를 담는 CPU 데이터 image v2를 검증한다.
2. [ ] 온라인 팀 전달 계약을 검토받고 미확정 schema·trigger·PII 항목을 합의한다.
3. [x] feedback export manifest와 합성 event fixture를 작성한다.
4. [x] KFP CPU/GPU component wrapper와 artifact 계약을 구현한다.
5. [x] 초기 학습 pipeline을 작성하고 로컬에서 IR YAML compile test를 통과한다.
6. [x] feedback validator·adapter·dedup·RouterBench merge core를 구현한다.
7. [x] 재학습 pipeline 골격을 작성하고 합성 fixture로 compile/contract test를 통과한다.
8. [ ] RouterArena 예측 그룹 분포 시각화와 모델 package/deploy gate를 구현한다.
9. [ ] 서버 확보 후 registry/GCS/IAM을 연결하고 100행 cloud smoke를 수행한다.
10. [ ] cloud smoke 통과 후 전체 학습·평가·승격과 온라인 Router 통합을 실행한다.

## 8. 관련 문서

- `docs/routerbench-data-labeling-spec.md`
- `docs/routerarena-evaluation-preprocessing-spec.md`
- `docs/routerarena-routing-evaluation-spec.md`
- `docs/modernbert-training-spec.md`
- `docs/online-ml-handoff-contract.md`
- `docs/mvp-scope-and-checklist.md`
- `docs/졸업프로젝트 개요.md`
- `ml/README.md`
