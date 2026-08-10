# RouterArena Routing Evaluation Specification

## 1. 목적

RouterBench로 학습한 ModernBERT 3-class router checkpoint에 RouterArena의
`model_input_text`를 넣고, 예측한 `easy/medium/hard`와 RouterArena가 공개한 공식
3단계 `Difficulty` 라벨을 비교한다. 공개 Parquet에는 질문별 Bloom 6단계 라벨이
없으므로 6단계 분포나 6→3 변환은 평가 범위에 포함하지 않는다.

이 평가는 라벨 정의와 데이터 출처가 다른 환경에서의 transfer alignment를 측정하며
RouterArena를 학습이나 threshold fitting에 사용하지 않는다. 같은 라벨 이름을
사용하더라도 RouterArena 라벨을 RouterBench 라우팅 목표의 절대 정답으로 간주하지
않는다.

RouterArena의 라벨은 “어떤 Claude tier가 충분한가”를 RouterBench와 같은 방식으로
측정한 라벨이 아니다. 따라서 일치율과 under/over-routing은 **외부 3단계 라벨에 대한
alignment proxy**이며, 운영상 Haiku/Sonnet/Opus의 품질 부족·비용 낭비를 직접
증명하지 않는다. 배포 판단은 RouterBench holdout 및 실제 Claude 표본 검증과 함께
수행한다.

## 2. 입력 계약

### 모델

Hugging Face `AutoModelForSequenceClassification`과 `AutoTokenizer`로 읽을 수 있는
checkpoint를 사용한다.

- 출력 class는 정확히 3개여야 한다.
- 권장 `id2label`: `0=easy`, `1=medium`, `2=hard`.
- checkpoint가 기본 `LABEL_0/1/2`를 쓰면 config의 `label_order`를 적용한다.
- checkpoint label과 config label 집합이 다르면 평가를 중단한다.

### 데이터

RouterArena 전처리기의 `evaluation.parquet`을 읽는다.

```text
sample_id
model_input_text
difficulty_label
evaluation_split
source_sample_id
```

같은 디렉토리의 `metadata.parquet`이 있으면 domain, category, source dataset을
결합해 세부 metric을 만든다. `Answer`는 추론 입력과 prediction artifact에 넣지 않는다.

## 3. 추론 규칙

- `model_input_text`만 tokenizer에 전달한다.
- batch inference와 CPU/CUDA/MPS를 지원한다.
- 기본 max length는 프로젝트 ModernBERT 사양에 맞춘 8,192 token이다.
- truncation 전 token 수와 잘림 여부를 행별로 기록한다.
- probability, 최대 확률 confidence, model-forward latency를 기록한다.
- GCS model artifact는 Kubeflow component가 로컬 경로로 받은 뒤 evaluator에 전달한다.

기본 label-to-route 매핑:

```text
easy   -> haiku
medium -> sonnet
hard   -> opus
```

## 4. 비교 규칙

### 4-1. 주 분석: 예측 그룹 내부의 RouterArena 라벨 분포

모델 예측으로 먼저 `predicted easy/medium/hard` 세 집단을 만든 뒤, 각 집단에서
RouterArena 공식 라벨이 차지하는 개수와 비율을 계산한다.

```text
P(routerarena_label | predicted_group)
```

주 결과는 다음 두 형태로 저장·시각화한다.

- 예측 그룹을 행, RouterArena 라벨을 열로 둔 3×3 count 및 행 정규화 표
- 예측 그룹별 RouterArena 라벨 비율을 나타낸 100% 누적 막대그래프와 정규화 히트맵

RouterArena 전체 라벨 분포도 함께 두고, 각 예측 그룹에서 라벨 비율이 전체 대비
몇 %p 증가·감소했는지 기록한다. 원본 클래스 수가 다르므로 count만으로 집단 특성을
판단하지 않는다.

반대 조건부 분포 `P(predicted_group | routerarena_label)`도 confusion matrix의
반대 방향 정규화 보조표로 제공해 각 RouterArena 라벨이 어디로 라우팅되는지 확인한다.

### 4-2. 보조 분석: 동일 라벨명 기준의 일치도와 방향성

난이도 순서는 `easy < medium < hard`로 고정한다.

```text
routing_delta = predicted_rank - routerarena_rank

0보다 작음: under_routing
0:          correct
0보다 큼:  over_routing
```

`hard -> easy`는 tier distance 2인 severe under-routing proxy로 별도 집계한다.
accuracy, macro-F1과 방향성 지표는 주 분석을 보완하는 alignment metric으로 해석한다.

## 5. 산출물

```text
ml/artifacts/routerarena-routing-evaluation/
├── predictions.parquet
├── metrics.json
└── model_info.json
```

### predictions.parquet

```text
sample_id, source_sample_id, evaluation_split
model_input_text
routerarena_label, predicted_label, predicted_route
prediction_confidence
prob_easy, prob_medium, prob_hard
routerarena_rank, predicted_rank, routing_delta, tier_distance
is_label_match, is_under_routed, is_over_routed, routing_result
input_token_count, was_truncated, inference_latency_ms
domain, category, source_dataset, label_source  # metadata가 있을 때
```

### metrics.json

- 예측 그룹별 RouterArena 라벨 count, 행 정규화 비율, 전체 분포 대비 %p 차이
- RouterArena 라벨별 예측 그룹 비율
- accuracy, macro precision/recall/F1
- class별 support/precision/recall/F1
- 3×3 confusion matrix
- under-routing count/rate
- over-routing count/rate
- severe under-routing count
- mean absolute tier error
- RouterArena/predicted label 분포
- Haiku/Sonnet/Opus 예상 routing 분포
- domain/source dataset별 accuracy, macro-F1, under/over-routing
- truncation count/rate
- model-forward latency mean/p50/p95
- 모델 경로, device, batch size, max length와 label order

## 6. 실행

PyTorch는 CPU/CUDA 이미지별 배포 방식이 달라 기본 전처리 의존성과 분리했다.

```bash
.venv/bin/pip install -e 'ml[model-evaluation]'

.venv/bin/routerarena-evaluate \
  --model-path /path/to/routerbench-modernbert-checkpoint \
  --evaluation-path ml/artifacts/routerarena-evaluation-full/evaluation.parquet \
  --metadata-path ml/artifacts/routerarena-evaluation-full/metadata.parquet \
  --output-dir ml/artifacts/routerarena-routing-evaluation \
  --batch-size 8 \
  --max-length 8192 \
  --device auto
```

## 7. Kubeflow 책임 경계

평가기 코드는 checkpoint 추론, 라벨 비교와 metric 생성을 담당한다. Kubeflow는 다음을
담당한다.

- RouterBench checkpoint와 RouterArena evaluation artifact 전달
- full/robustness 평가 병렬 실행
- model/data/evaluation version ML Metadata 계보
- CPU/GPU resource, retry, cache 설정
- 현재 배포 모델과 후보 모델 metric 비교
- 승인된 gate 기준에 따른 배포 또는 보류

## 8. 현재 검증 상태와 남은 결정

- [x] Hugging Face 3-class checkpoint predictor를 구현했다.
- [x] label mapping 검증을 구현했다.
- [x] accuracy/macro-F1/confusion matrix를 구현했다.
- [x] under/over/severe under-routing metric을 구현했다.
- [x] domain/source dataset별 metric을 구현했다.
- [x] prediction/metric/model-info artifact writer를 구현했다.
- [x] synthetic predictor 기반 단위·통합 테스트를 통과했다.
- [x] 공식 공개 라벨을 별도 변환 없이 `easy/medium/hard`로 비교하기로 확정했다.
- [x] 예측 그룹별 RouterArena 라벨 count/비율과 전체 분포 대비 차이를 metric으로
  생성한다.
- [x] 100% 누적 막대그래프와 행 정규화 3×3 히트맵을 SVG artifact로 생성한다.
- [x] evaluation semantics를 `routerarena-three-class-alignment`로 확정하고
  RouterArena 라벨을 `routerarena_label`로 명시한다.
- [ ] 실제 RouterBench checkpoint 경로를 확정한다.
- [ ] 실제 checkpoint의 `id2label`을 확인한다.
- [ ] T4 기준 batch size/max length를 smoke로 확정한다.
- [ ] 후보 모델과 기존 모델을 비교할 deploy gate 임계값을 확정한다.
