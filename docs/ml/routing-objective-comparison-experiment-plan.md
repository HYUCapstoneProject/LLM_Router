# 동일 원천 데이터 기반 라우팅 목적함수 비교 실험 계획

> 2026-09-20 범위 갱신: [졸업 프로젝트 실험 계약](graduation-project-experiment-charter.md)을
> 우선한다. 최소 실행은 R0/R1/R2이며 R3는 확장 후보로 보존한다. BGE+kNN은 캐시와 역할이
> 다른 저비용 label-voting 기준선으로 포함한다. R0의 현재 `ordinal` 구현은 일반 3-class CE다.
> 모델/표현 비교와 목적함수 비교, 공식 논문 adapter 비교를 분리하며 어떤 방법의 우승도
> 사전에 가정하지 않는다. 기존 코드·config·미커밋 실험 작업을 이 주석으로 변경하지 않는다.

기준일: 2026-09-18  
브랜치: `feat/routing-objective-comparison`  
상태: **실행 전 사전등록 계획** — validation lock 전까지 test와 RouterArena 결과를 모델·목적함수 선택에 사용하지 않는다.

관련 문서:

- `integrated-routing-data-and-model-comparison-plan.md`: RouterBench+xRouteBench 통합, T1/T2, 백본 비교
- `routerarena-routing-evaluation-spec.md`: RouterArena 외부 정렬평가 계약
- `routerbench-data-labeling-spec.md`: 기존 3단계 난이도 라벨 정의
- `modernbert-training-spec.md`: 현행 ModernBERT 학습·checkpoint 계약
- `workstream-checklist.md`: 프로젝트 전체 진행 상태의 단일 체크리스트

이 문서는 기존의 “같은 3-class 목적함수에서 encoder를 교체하는 비교”와 별도로,
**동일한 query×model 원천 결과에서 서로 다른 라우팅 학습 신호와 의사결정 규칙을 비교**한다.
기존 통합 계획의 데이터 정제와 split은 재사용하되, 모델별 결과를 하나의 난이도 라벨로
압축하기 전의 품질·성공·비용 신호를 추가로 보존한다.

## 0. 결정 요약

### 0-1. 연구 질문

> 동일한 RouterBench+xRouteBench 질문, split, 후보 실행 결과, 비용 회계와 query-only
> single-dispatch 조건에서 ordinal difficulty classification, per-tier success prediction,
> pairwise preference ranking, direct regret minimization 중 어떤 학습 목적이 가장 좋은
> 비용–품질 Pareto frontier를 만드는가?

### 0-2. 비교 축 분리

| 단계 | 바꾸는 것 | 고정하는 것 | 목적 |
| --- | --- | --- | --- |
| A. 목적함수 비교 | R0~R3 학습 신호·loss·decision rule | ModernBERT-base, 입력, split, action, 학습 예산 | 라우팅 기법 효과 격리 |
| B. 백본 비교 | TF-IDF, MiniLM, RoBERTa, ModernBERT, DeBERTa, Qwen | A에서 선택한 목적함수 | encoder 효과 격리 |
| C. 운영 검증 | abstract tier→Haiku/Sonnet/Opus mapping | 선택된 라우터·운영 정책 | 실제 Claude 정합성 확인 |

현재 `integrated-routing-data-and-model-comparison-plan.md`는 B를 다루고, 이 문서는 A를
다룬다. A의 우승 목적함수를 확정하기 전에는 “어느 backbone이 우수하다”와 “어느
라우팅 기법이 우수하다”를 한 결론으로 합치지 않는다.

### 0-3. 핵심 비교군

| ID | 기법 | 학습 신호 | 온라인 결정 | 추가 LLM 호출 |
| --- | --- | --- | --- | ---: |
| R0 | ordinal difficulty | T1 `easy/medium/hard` | cost penalty를 적용한 3-class score | 0 |
| R1 | pointwise success | tier별 accept 여부 | 예상 성공도−비용 penalty | 0 |
| R2 | pairwise preference | tier 쌍별 품질 우위·동률 | latent rank score−비용 penalty | 0 |
| R3 | direct regret | 품질−λ×비용의 oracle action과 regret | regret score−비용 penalty | 0 |

모든 방법은 질문을 한 번 인코딩하고 `low/medium/high` 중 하나를 고른 뒤 선택한 LLM을
한 번만 호출한다. 임베딩 cache 조회, 이웃 검색, 응답 생성 후 verifier, 자동 cascade는
비교군에 포함하지 않는다.

### 0-4. 우선 제외

- kNN, similarity-weighted routing, semantic router: 이미 앞단의 BGE 임베딩 cache와
  방법론적 역할이 겹친다.
- AutoMix, FrugalGPT, BEST-Route, response verifier/cascade: 이미 사용자 escalation이
  설계되어 있고 두 번 이상의 downstream 호출 가능성이 있다.
- contextual bandit: 배포 시 선택한 arm의 feedback만 관측하는 별도 정보 조건이므로
  full-feedback offline 방법과 같은 표에서 우열을 확정하지 않는다.
- IRT-Router online warm-up: semantic similarity를 사용하므로 제외한다. similarity를 뺀
  IRT core는 R0~R3 이후의 해석 가능성 보조실험 후보로만 둔다.
- API 기반 LLM-as-router: 라우터 자체 토큰 비용과 지연을 추가하고 동일 학습 예산 통제가
  어려우므로 본 실험에서 제외한다.

## 1. 논문 근거와 이 실험의 위치

1. [RouterBench](https://arxiv.org/abs/2403.12031)는 동일 질문에 대한 다수 모델의 품질·비용
   결과를 제공하므로 label classification 외에 model selection 학습 신호를 만들 수 있다.
2. [RouteLLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html)은
   strong/weak 응답의 pairwise preference로 win probability를 학습한다. 본 실험의 R2는
   similarity-weighted variant를 제외하고 이를 세 개 tier의 공통 score로 확장한다.
3. [Causal LLM Routing](https://arxiv.org/abs/2505.16037)은 품질·비용 예측을 분리하면
   오차가 누적될 수 있음을 지적하고 최종 decision regret을 직접 최소화한다. 본 실험의
   R3는 observational causal correction이 아니라 full-feedback에서 계산 가능한
   regret-weighted objective를 사용한다.
4. [IRT-Router, ACL 2025](https://arxiv.org/abs/2506.01048)는 model ability와 query
   attributes의 상호작용을 명시적으로 모델링한다. 해석 가능성 확장 후보이나 semantic
   warm-up은 현재 cache와 겹치므로 본 비교에서 제외한다.
5. [LLMRouter/xRouteBench](https://arxiv.org/abs/2608.06867)는 context encoder, model
   encoder, scoring, decision rule, learning signal을 분리하고 같은 query·candidate pool·metric
   아래 비교해야 함을 강조한다. 본 문서는 이 분리를 세 개 abstract action에 적용한다.
6. [LLMRouterBench](https://arxiv.org/abs/2601.07206)는 통일된 평가에서 여러 최신 라우터가
   비슷한 성능을 보이고 일부는 Best Single을 안정적으로 넘지 못했다고 보고한다. 따라서
   always-tier, Best Single, Oracle을 필수 기준선으로 둔다.

## 2. 불변 조건과 허용 정보

### 2-1. 공통으로 고정하는 것

- 원천: RouterBench text-only + xRouteBench `llmrouter_generic`
- 질문 집합과 duplicate group
- 기존 통합 데이터의 train/validation/RouterBench test/xRouteBench official test
- `model_input_text`와 최대 512 token
- query-only 입력: category, dataset name, 정답, response, model name을 입력에 넣지 않는다.
- action space: `low`, `medium`, `high`
- encoder: `answerdotai/ModernBERT-base@8949b9...`
- optimizer, learning rate, batch, epoch, seed 정책
- validation budget과 test 공개 시점
- 비용 회계와 metric 계산 코드
- RouterArena를 모델·threshold·penalty 선택에 쓰지 않는 계약

### 2-2. 같게 만들지 않는 것

같은 원천 데이터를 사용한다는 이유로 모든 방법에 T1/T2 label을 강제하지 않는다. 그러면
R1~R3도 결국 난이도 분류기가 되어 연구 질문이 사라진다. 대신 같은 원천 matrix에서
다음 view를 파생한다.

| 방법 | 파생 view |
| --- | --- |
| R0 | `difficulty_label`, 선택적으로 `difficulty_label_t2` |
| R1 | `success_low`, `success_medium`, `success_high` |
| R2 | 세 tier 품질의 pairwise difference와 tie |
| R3 | `quality_t - λ × decision_cost_t`와 oracle regret |

## 3. 공통 3-action outcome matrix

### 3-1. 모델 pool 차이 처리

RouterBench와 xRouteBench의 후보 모델 ID가 다르므로 model ID를 합치지 않는다. 각 source의
train split에서만 Pareto frontier를 만들고, 다음 deterministic 규칙으로 대표 arm 세 개를
고른다.

1. 모델별 train mean quality와 train mean actual cost를 계산한다.
2. 더 비싸면서 품질이 같거나 낮은 dominated model을 제거한다.
3. Pareto frontier의 최저 비용 모델을 `low`로 둔다.
4. 최고 평균 품질 모델을 `high`로 둔다.
5. 정규화 cost-quality 직선에서 수직 거리가 가장 큰 knee model을 `medium`으로 둔다.
6. 동률은 품질 내림차순, 모델 이름 오름차순으로 결정한다.
7. 선택된 arm과 통계는 `arms.json`, `candidate-statistics.parquet`에 봉인한다.

이 규칙은 validation/test를 보지 않고 정한다. 세 source-local 모델은 공통 action
`low/medium/high`로 투영되며, 서비스에서는 이 action을 Haiku/Sonnet/Opus에 매핑한다.
이 매핑은 실제 Claude 품질을 증명하지 않으며 후속 450건 end-to-end 검증에서 확인한다.

### 3-2. 비용 두 종류

- `decision_cost_t`: train mean cost로 만든 source-local 0~1 고정값. 라우터 의사결정에만
  사용한다. 출력 길이를 미리 아는 leakage를 방지한다.
- `actual_cost_t`: 이미 실행된 해당 응답의 실제 input/output token 비용. 평가에만 사용한다.

라우터는 online에서 realized output token을 알 수 없으므로 `actual_cost_t`를 입력이나
decision rule에 넣지 않는다.

### 3-3. 행 schema

```text
sample_id
source_dataset
objective_split
model_input_text
difficulty_label
difficulty_label_t2
duplicate_group_id
sample_weight

model_low / model_medium / model_high
quality_low / quality_medium / quality_high
success_low / success_medium / success_high
decision_cost_low / decision_cost_medium / decision_cost_high
actual_cost_low / actual_cost_medium / actual_cost_high
```

선택된 세 arm 중 하나라도 품질·success·비용이 결측이면 해당 dataset version은 실패시킨다.
자동 대체나 평균 대입은 하지 않는다.

## 4. 목적함수 명세

모든 방법은 ModernBERT의 CLS representation에서 세 개 logit을 출력한다. head 수와 출력
크기를 같게 유지해 encoder compute 차이가 방법 비교에 섞이지 않게 한다.

### 4-1. R0 ordinal difficulty

```text
L_R0 = CrossEntropy(logits, T1 difficulty label)
```

T1이 주 결과이며 T2는 seed 42 sensitivity만 수행한다. 비용 curve를 만들 때 validation에서
동결한 penalty를 세 class score에 적용한다. T1/T2 중 더 좋은 결과를 사후 선택하지 않는다.

### 4-2. R1 pointwise success

```text
L_R1 = mean_t BCEWithLogits(score_t, success_t)
```

추론 시 sigmoid score에서 cost penalty를 뺀다. `success_t`는 exact metric은 1.0,
continuous metric은 기존 계약대로 0.70 이상일 때 1이다. 세 성공확률에 단조 제약을 강제하지
않는다. 실제 모델은 특정 query family에서 작은 모델이 큰 모델보다 나을 수 있기 때문이다.

### 4-3. R2 pairwise preference

각 pair `(i,j)`에 대해 `quality_i-quality_j`의 부호를 target으로 사용한다.

```text
P(i > j | q) = sigmoid(score_i - score_j)
L_R2 = weighted pairwise BCE
```

품질 차이가 클수록 loss weight를 크게 하고 exact tie는 target 0.5, weight 0.25로 둔다.
RouteLLM의 nearest-query similarity 방식은 사용하지 않는다.

### 4-4. R3 direct regret

고정 λ grid 각각에 대해:

```text
utility_t(q, λ) = quality_t(q) - λ × decision_cost_t
oracle(q, λ) = argmax_t utility_t(q, λ)
regret_t = utility_oracle - utility_t
```

모델 score에도 같은 cost penalty를 적용하고 oracle action에 대한 cross entropy를
best–second utility gap으로 가중한다. λ는 train-only config에서 사전 고정하며 RouterArena로
조정하지 않는다.

## 5. 학습 실행 계획

### 5-1. 공통 학습 설정

| 항목 | 값 |
| --- | --- |
| backbone | ModernBERT-base pinned revision |
| max length | 512 |
| epoch | 3 |
| micro batch | 8 |
| gradient accumulation | 4 |
| effective batch | 32 |
| optimizer | AdamW |
| learning rate | `2e-5` |
| weight decay | `0.01` |
| warmup | 10% |
| precision | CUDA fp16 |
| gradient clipping | 1.0 |
| canonical seed | 42 |
| finalist seeds | 17, 73 |

### 5-2. 단계별 실행 수

1. 네 방법 모두 1,200 train / 300 validation / 40 optimizer step smoke
2. 네 방법 모두 seed 42 full run
3. seed 42 validation NAUC 상위 두 방법을 provisional finalist로 동결
4. finalist만 seed 17, 73 full run
5. 세 seed 평균 NAUC가 높은 방법을 winner로 정하되 test 실행 모델은 canonical seed 42로 고정
6. 네 방법의 canonical seed 42를 sealed test에서 한 번씩 평가
7. 같은 네 canonical run을 RouterArena에 inference-only로 한 번씩 평가

총 full training은 8회다: 네 방법 seed 42 네 번, finalist 두 방법의 추가 seed 두 개씩 네 번.
T2 R0 sensitivity를 실행하면 짧은 추가 run 한 번이 생긴다. smoke 결과는 우승자 선정에
포함하지 않는다.

## 6. 비용–품질 의사결정과 validation lock

방법별 raw score scale이 다르므로 같은 숫자의 penalty를 공정하다고 간주하지 않는다.
사전 고정 penalty grid를 validation에서만 sweep하고 다음 relative cost budget별 최적
penalty를 lock한다.

```text
0%, 10%, 25%, 50%, 75%, 100%
```

relative cost는 always-low 평균비용을 0, always-high 평균비용을 1로 정규화한다. validation
lock에는 방법별 budget→penalty, provisional finalists, final winner, canonical seed를 쓴다.
test에서는 이 penalty를 그대로 적용하고 재탐색하지 않는다.

## 7. 평가 지표와 기준선

### 7-1. 주 지표

- `quality_cost_nauc`: always-low와 always-high 사이 quality gap recovery의 budget AUC
- budget별 mean quality와 actual cost
- high-tier 품질의 95%, 98%를 달성하는 최소 relative cost
- 품질 oracle 대비 regret
- low/medium/high call rate
- RouterBench/xRouteBench source별 quality와 source macro quality
- 라우터 batch-1 latency, peak VRAM, artifact 크기

### 7-2. 안전·진단 지표

- T1 label 대비 under-routing, over-routing, hard→low
- category/source별 route 분포
- 모델별 calibration은 R0/R1에만 직접 해석하고 R2/R3 raw score에는 확률 해석을 붙이지 않는다.

### 7-3. 필수 기준선

- always-low
- always-medium
- always-high
- Best Single: 전체 test에서 사후 선택하지 않고 validation 평균 quality 최고 고정 tier
- random with fixed seed
- quality oracle: 질문별 최고 quality tier
- utility oracle: 고정 λ별 최고 utility tier

### 7-4. 통계

- 질문 단위 paired bootstrap 2,000회
- 방법별 seed 평균·표준편차
- winner와 R0의 default 50% budget 품질·비용 차이 95% CI
- source별 bootstrap과 통합 macro 결과를 함께 보고
- CI가 0을 포함하면 “동급”으로 보고 단순성·latency가 낮은 방법을 운영 권고

## 8. test와 RouterArena의 역할

### 8-1. sealed RouterBench+xRouteBench test

주 cost-quality 결론은 저장된 모델 실행 결과를 replay해 만든다. 추가 LLM API 비용이 없다.
validation lock이 `final`이 아니면 test CLI는 실행을 거부한다. 네 방법 모두 같은 lock으로
한 번만 평가한다.

### 8-2. RouterArena label alignment

RouterArena의 `easy/medium/hard`는 Bloom taxonomy 기반 외부 난이도이며 “어느 tier가
충분한가”의 정답이 아니다. 따라서 다음만 보고한다.

- accuracy, macro-F1, balanced accuracy
- under/over-routing
- hard→low 비율
- tier 분포

이 결과로 method, seed, penalty, threshold를 다시 선택하지 않는다. R0가 직접 3-class label을
학습했기 때문에 label alignment는 R0에 구조적으로 유리할 수 있음을 명시한다.

### 8-3. 후속 Claude end-to-end

실제 비용–품질 외부검증이 필요하면 RouterArena 또는 별도 450건 표본에 Haiku/Sonnet/Opus
응답을 각각 한 번씩만 생성해 공통 outcome matrix를 만든다. 모든 라우터가 이 matrix를
재사용하므로 라우터 수만큼 API 호출을 반복하지 않는다. 이 단계 전까지 abstract tier
결과를 Claude 성능으로 표현하지 않는다.

## 9. Colab notebook 실행 순서

경로: `ml/notebooks/routing_objective_comparison/`

| 순서 | 파일 | 런타임 | test 접근 | 완료 산출물 |
| ---: | --- | --- | --- | --- |
| 00 | `00_prepare_outcome_matrix.ipynb` | CPU | 금지 | outcome dataset, arms, manifest |
| 01 | `01_smoke_all_objectives.ipynb` | T4 | 금지 | 네 objective smoke `_SUCCESS` |
| 02 | `02_ordinal_difficulty.ipynb` | T4 | 금지 | R0 seed42 full, T2 sensitivity |
| 03 | `03_pointwise_success.ipynb` | T4 | 금지 | R1 seed42 full |
| 04 | `04_pairwise_preference.ipynb` | T4 | 금지 | R2 seed42 full |
| 05 | `05_direct_regret.ipynb` | T4 | 금지 | R3 seed42 full |
| 06 | `06_validation_lock_and_finalists.ipynb` | CPU→T4 | 금지 | provisional top2, seed17/73, final lock |
| 07 | `07_sealed_test_and_routerarena.ipynb` | T4/CPU | 허용 | sealed test와 RouterArena 보고 |

각 notebook은 Drive의 code archive와 직전 단계 `_SUCCESS`를 확인하고, local scratch에서
완료한 뒤 `.staging`→final rename으로 게시한다. test 파일은 07 이전 notebook이 복사하지
않는다.

## 10. 산출물 계약

```text
MyDrive/LLM_Router/routing-objective-comparison/v1/
├── dataset/
│   ├── manifest.json
│   ├── arms.json
│   ├── candidate-statistics.parquet
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test_routerbench.parquet
│   ├── test_xroutebench.parquet
│   └── _SUCCESS
├── smoke/{method}/seed-42/
├── runs/{method}/seed-{42,17,73}/
│   ├── model/
│   ├── validation-scores.parquet
│   ├── validation-curve.parquet
│   ├── metrics.json
│   ├── manifest.json
│   └── _SUCCESS
├── selection/
│   ├── provisional-lock.json
│   └── final-lock.json
└── evaluation/
    ├── sealed-test/
    └── routerarena-alignment/
```

`manifest.json`은 base dataset manifest checksum, config checksum, 입력 parquet checksum,
method, seed, smoke/full 여부, test/RouterArena 접근 여부를 보존한다.

## 11. 사전 실패 조건과 중단 규칙

- Pareto frontier가 세 모델 미만이면 자동 tier 생성 중단
- selected arm의 train mean cost 또는 quality 순서가 증가하지 않으면 중단
- 선택된 arm 결과가 한 행이라도 결측이면 중단
- duplicate group이 split을 넘으면 중단
- base dataset coverage resolution이 complete가 아니면 중단
- smoke에서 NaN loss, OOM, prediction row mismatch가 발생하면 full run 금지
- 네 canonical seed42 run이 모두 없으면 provisional lock 금지
- finalist 추가 seed가 모두 없으면 final lock 금지
- final lock 없이는 test와 RouterArena 평가 금지

## 12. 해석 규칙

1. R1이 R0보다 낫다면 난이도 label compression보다 tier별 성공 신호가 유효하다고 해석한다.
2. R2가 R1보다 낫다면 절대 score calibration보다 상대적 model preference가 안정적이라고
   해석한다.
3. R3가 가장 낫다면 비용을 사후 threshold로 붙이는 것보다 학습 objective에 포함하는 것이
   유효하다고 해석한다.
4. 차이가 통계적으로 작으면 가장 단순하고 빠른 R0/R1을 운영 권고한다.
5. 모든 learned router가 Best Single을 넘지 못하면 데이터 규모나 backbone보다 먼저
   candidate complementarity와 label/outcome noise를 재검토한다.
6. Oracle gap이 작으면 라우터 개선보다 candidate pool 선정이 우선이다.
7. RouterArena alignment가 낮아도 sealed cost-quality가 좋으면 실패로 단정하지 않는다.
   두 평가는 서로 다른 target을 측정한다.

## 13. 본 실험 이후

1. 우승 목적함수를 기존 백본 비교 계획에 적용한다.
2. ModernBERT와 경량 MiniLM의 품질–latency Pareto를 다시 계산한다.
3. 운영 후보를 선택하면 Spring의 tier→Claude mapping은 별도 version으로 관리한다.
4. 450건 Claude 공통 outcome matrix로 abstract tier transfer를 검증한다.
5. 실제 선택 arm feedback이 충분히 쌓인 뒤 contextual bandit을 별도 online track으로 연다.
