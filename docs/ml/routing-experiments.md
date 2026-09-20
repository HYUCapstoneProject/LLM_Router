# 라우팅 비교 실험

기준일: 2026-09-20. 이 문서는 최종 라우팅 실험의 범위·학습 목표·공정 비교 조건을 정의한다. 연구 질문과 문서 지도의 시작점은 [README](README.md)다. 데이터 변환은 [데이터·전처리](data-and-preprocessing.md), 지표와 선택 절차는 [평가](evaluation.md), 실행 순서는 [실행·산출물](execution-and-artifacts.md)을 따른다.

## 1. 비교 질문과 범위

세 축을 구분한다.

| 축 | 고정 | 변경 | 설명 가능한 차이 |
| --- | --- | --- | --- |
| 표현·모델 | 같은 질문·split·T1 3-class label | TF-IDF, 고정 BGE+kNN, ModernBERT | 동일 난이도 목표를 추정하는 표현·학습 방식 |
| 학습 목표 | ModernBERT, 입력, 실제 세 후보, 학습 예산 | R0 CE / R1 success / R2 pairwise | 같은 원천 결과를 학습 신호로 표현하는 방식 |
| 논문 라우터 | 한 source, 같은 실제 세 후보·split·원천 결과 | RouterDC와 우리 비교군 | 완성된 라우팅 알고리즘과의 비용–품질 차이 |

캐시와 재학습은 별도 실험에서 검증한다. 주 라우팅 비교 도중 캐시를 켜거나 feedback으로 특정 비교군만 업데이트하지 않는다. 모든 비교군은 동일한 공개 원천 증거를 사용하지만 target 변환까지 같게 강제하지 않는다.

분류 확률은 난이도 label의 확률이며 LLM 정답 확률이 아니다. 후보의 상대 점수도 확률이 아니다. 각 출력의 의미를 보존하고 모두를 같은 accuracy 표 하나로 순위 매기지 않는다.

## 2. 두 데이터 실행 계약

### A. 통합 난이도와 symbolic-tier 비교

RouterBench+xRouteBench의 동일한 통합 train/validation을 사용한다. T1이 주 라벨, T2는 제한된 민감도 분석이다. source별 실제 모델 pool이 다른 상태에서 통합 난이도 표현을 학습하는 실험이다.

기존 `routing-objective-comparison-v1`은 source마다 선택한 실제 세 모델을 공통 low/medium/high index로 투영해 통합 R0/R1/R2를 학습한다. source metadata는 evaluator가 올바른 outcome/cost를 찾는 데만 쓰고 질문 encoder에 붙이지 않는다. 이 결과를 동일한 전역 모델 세 개의 성능이라고 부르지 않는다.

### B. 최소 논문 비교의 실제 3-action 실험

우선 xRouteBench generic 하나의 source/pool에서 R0/R1/R2와 RouterDC를 동일 train으로 다시 학습한다. 후보 세 모델은 모든 방법에 동일하다. RouterBench 재현은 별도 source 실험으로 확장한다. A의 통합 checkpoint와 B의 source-only checkpoint를 동일 조건의 직접 비교처럼 발표하지 않는다.

B의 T1은 해당 source의 train만 대상으로 동일한 threshold grid와 group OOF 절차에서 별도 선정해 B의 모든 분류기가 공유한다. A의 통합 train에서 고른 T1을 가져오지 않는다. source macro와 source 최소 F1 항은 그 한 source의 F1로 계산하고, class·bootstrap·category coverage도 해당 source 안에서 검사한다. 예를 들어 기존 클래스별 10%·500행 gate를 만족하지 못하면 그 사실을 보고하며 test를 보고 완화하지 않는다. B의 T1 값은 A와 같다고 보장하지 않고 별도 manifest에 봉인한다.

현 outcome builder는 A의 통합 projection을 생성한다. B의 source-specific runner와 외부 공식 구현 adapter는 구현 보완이 필요하다. 사용할 후보 수만 3으로 바꾸었다고 재현이 완료된 것으로 보지 않는다.

## 3. 최소 비교군과 선택 확장

| 방법 | 필수 여부 | 목적 |
| --- | --- | --- |
| TF-IDF + Logistic Regression | 필수 | 저비용 text 분류 기준선 |
| Frozen BGE + kNN | 필수 | 고정 의미 표현과 가까운 학습 사례에 의한 분류 |
| ModernBERT + R0 CE | 필수 | 기본 난이도 분류기 |
| ModernBERT + R1 success | 필수 | 후보별 성공 예측 |
| ModernBERT + R2 pairwise | 필수 | 후보 간 상대 품질 예측 |
| RouterDC, 동일 세 후보 | 필수 논문 비교 1개 | 공식 방식과의 직접 비교 |
| RoBERTa-base + CE | 자원 여유 후 우선 확장 | 근접 규모 backbone 비교 |
| GraphRouter, CARROT, R3 regret | 선택 확장 | 다른 구조·학습 목표의 추가 검증 |

DeBERTa·Qwen 등 이전의 광범위한 backbone sweep은 최종 최소 범위가 아니다. RouteLLM 원형의 두 후보 비교도 3-action 방법을 사용할 수 없는 경우에만 별도 두 후보 실험으로 둔다. 이미 존재하는 구현과 notebook은 연구 선택지를 보존한 것이며 전부 실행하라는 지시가 아니다.

## 4. 표현·모델 비교

### TF-IDF + Logistic Regression

train text에 word `(1,2)`-gram, `min_df=2`, `sublinear_tf=true`, 최대 feature 100,000을 fit한다. Logistic Regression은 `C=1`, `class_weight=None`, `lbfgs`, 최대 iteration 2,000, seed 42를 기준으로 한다. validation/test에 vectorizer를 fit하지 않는다.

이것은 같은 3-class 목표를 다른 특징과 분류기로 푸는 기준선이다. 이를 독립적인 비용 최적화 목적함수라고 설명하지 않는다. epoch가 없으므로 neural epoch 수와 형식적으로 맞추지 않고 fit 시간·메모리·artifact 크기를 기록한다.

### Frozen BGE + kNN

서비스와 같은 `BAAI/bge-large-en-v1.5`의 고정 revision을 사용한다. 현재 서비스 예시 revision은 `d4aa6901d3a41ba39fb536a557fa166f842b0e09`이며 실험 manifest에서 실제 사용 hash를 다시 고정한다. 출력은 1,024차원 L2-normalized vector, 입력 상한 512 token이다.

1. train 질문만 임베딩해 T1 label과 함께 index에 저장한다.
2. validation에서 `k ∈ {3,5,11,21}`와 균등/거리 가중 투표를 선택한다.
3. 새 질문과 cosine 유사도가 높은 train 이웃을 찾아 class별 표를 합산한다.
4. 가장 높은 label을 반환한다. 거리 weighting 함수·epsilon·동률 순서도 config로 동결한다.

이웃 다섯 개가 easy/easy/easy/medium/hard이면 균등 투표 결과는 easy다. 같은 질문의 test 결과를 index에 추가하지 않는다. kNN의 label index는 서비스의 응답 캐시와 다르며, [캐시 실험](semantic-cache.md)의 상태와 합치지 않는다.

BGE weight를 업데이트하지 않으므로 fine-tuning 비용은 없지만 pretrained encoder의 파라미터·임베딩 생성·index 저장·검색 비용은 있다. “파라미터 0” 또는 “추론 비용 0”이라고 보고하지 않는다. 투표 비중은 score이며 별도 검증 없이 calibrated probability로 부르지 않는다.

### ModernBERT와 근접 규모 모델

ModernBERT-base는 3-class supervision으로 encoder와 head를 fine-tune한다. checkpoint는 `answerdotai/ModernBERT-base`, revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`을 기준으로 한다. 원형의 긴 context 기능과 운영의 8,192 token 설정은 주 비교의 512 token 조건과 구분한다.

ModernBERT-base 약 149M, RoBERTa-base 약 125M은 근접 규모이지 동일 파라미터 수가 아니다. DeBERTa-v3-base를 추가하면 backbone 86M뿐 아니라 embedding 약 98M을 포함한 총 약 184M을 보고한다. [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), [RoBERTa](https://huggingface.co/FacebookAI/roberta-base), [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base).

모델별 파라미터 수·학습 가능한 수를 실제 로드 후 계산한다. 사전학습 corpus, tokenizer, 사전학습 목적이 다르므로 관측된 성능 차이를 architecture 한 가지의 인과 효과로 단정하지 않는다. 학습 목표만의 영향을 보려면 다음 절의 동일 backbone 실험을 사용한다.

## 5. 같은 backbone의 학습 목표

R0/R1/R2는 동일 pretrained ModernBERT와 세 logit 출력 크기를 사용한다. seed별 head 초기화·data order·최대 optimizer step을 맞춘다. eligible sample의 주 비교 loss weight는 모두 1이다. [공통 가중치 정책](data-and-preprocessing.md)을 따른다.

### R0: 3-class cross entropy

```text
L_R0 = CE(z(q), T1_difficulty(q))
inference_score = softmax(z)
```

원천 결과를 하나의 easy/medium/hard label로 압축해 학습한다. 기존 코드의 method 이름 `ordinal`은 이 CE 구현을 가리키며 ordinal regression 전용 loss라는 뜻이 아니다. 학술 표기는 R0 또는 3-class CE를 사용한다.

### R1: 후보별 성공 예측

```text
L_R1 = mean_t BCEWithLogits(z_t(q), success_t(q))
inference_score_t = sigmoid(z_t)
```

후보별 성공 0/1을 독립적으로 예측한다. 세 후보가 모두 성공하거나 모두 실패할 수 있으므로 class softmax를 사용하지 않는다. source별 metric acceptance는 [데이터 계약](data-and-preprocessing.md)에 고정한다. 비싼 모델이 모든 질문에서 더 잘한다고 가정해 출력에 단조 제약을 주지 않는다.

한 번 관측된 결과를 이용해 추정하는 성공 가능성이지 후보 모델의 참 성공 확률을 직접 측정한 값은 아니다. 연속 quality와 이산 success를 혼동하지 않는다.

### R2: 결과 기반 pairwise preference

```text
P(i > j | q) = sigmoid(z_i(q) − z_j(q))
target = 1 if quality_i > quality_j else 0 if quality_i < quality_j else .5
weight = abs(quality_i − quality_j) for non-tie, .25 for exact tie
L_R2 = 품질 차이로 가중한 세 pair BCE의 정규화 평균
inference_score = z
```

동일 질문에서 관측된 후보 간 품질 우열을 학습한다. 사람 선호 데이터 또는 신규 Claude judge를 사용한다는 의미가 아니다. 동률을 버리지 않고 .5 target으로 처리한다. `quality`의 원천별 척도 차이는 source별 평가와 exact/continuous 하위 분석에 남긴다.

품질이 같은 후보의 비용 차이는 target을 임의로 승패로 바꾸는 대신 아래의 decision rule에서 처리한다. 같은 점수·비용의 최종 동률은 고정된 action ID 순서로 해결한다.

### 무엇이 기존 성공률 라벨과 다른가?

동일한 공개 원천을 사용한다는 점은 같다. 학습에 남기는 정보가 다르다. 예를 들어 실제 세 후보 결과 X=(1,0,1), Y=(0,1,1)은 전체 성공률이 2/3으로 같아도 가장 싼 성공 후보가 다르다. R0 압축 라벨은 이 차이를 잃을 수 있고 R1/R2는 후보별 정보를 남긴다.

이 예시는 전체 성공률 압축의 설명이며 운영 v2의 tier-consensus 규칙을 그대로 재현한 예시는 아니다. 품질 관계가 단조적이고 label이 최적 행동을 충분히 나타낸다면 방법 간 차이는 작을 수 있다. 개선이 없는 결과도 연구 결과로 보고한다.

### R3: 선택 확장

기존 구현의 `regret`은 `quality−λ×decision_cost` 최적 action에 대한 CE를 utility gap으로 가중하는 변형이다. 기대 regret 자체를 미분하는 모든 알고리즘과 동치라고 하지 않는다. λ grid를 train 전에 고정하며 R3는 최소 실행에서 제외한다.

## 6. 실제 세 후보와 의사결정

각 source train에서 후보별 평균 quality와 actual cost를 계산한다. 지배당한 모델을 제거한 Pareto frontier에서 다음을 선택한다.

1. 최저 비용 후보 → low.
2. 최고 평균 품질 후보 → high.
3. 두 끝점을 정규화해 연결한 선에서 거리가 가장 큰 중간 후보 → medium.

동률은 품질 내림차순, model ID 오름차순으로 결정한다. 유효한 frontier가 세 개 미만이면 3-action 계약 실패로 보고한다. validation/test를 보고 좋은 후보를 다시 고르지 않는다. 결과와 train 통계는 `arms.json`과 `candidate-statistics.parquet`에 봉인한다.

```text
decision_cost_t = (train_mean_cost_t − train_mean_cost_low)
                  / (train_mean_cost_high − train_mean_cost_low)
action(q) = argmax_t [inference_score_t(q) − λ × decision_cost_t]
```

현재 inference rule은 후보별 observed output length를 사용하지 않는다. actual cost는 선택 후 evaluator가 집계한다. 각 방법의 score scale이 다르므로 같은 λ 숫자가 같은 비용 조건이라는 뜻이 아니다.

λ 후보 grid는 config에 사전 고정하고 validation에서 budget별 정책을 선택한다. test는 이 정책을 그대로 실행한 실제 quality/cost를 보고한다. test에서 target budget에 맞춰 λ를 재선택하지 않는다. 후보 score의 calibration, normalized area, quality target 비용 규칙은 [평가](evaluation.md)에 따른다.

세 action을 서비스의 Haiku/Sonnet/Opus로 연결하는 것은 별도 mapping이다. 과거 후보의 품질·가격을 현재 Claude와 같다고 가정하지 않으며 신규 Claude API는 학습에 사용하지 않는다.

## 7. 필수 비학습 기준선

| 기준선 | 정의 | 해석 |
| --- | --- | --- |
| always-low / medium / high | 모든 질문에 같은 실제 후보 선택 | 단일 모델 서비스와 비교 |
| best-single | train 평균 품질로 정한 후보 하나 | 가장 비싼 모델과 최고 평균 품질이 다를 수 있음 |
| usage-matched random | validation에서 고정한 모델 사용 비율로 무작위 배정 | 사용량을 넘어 질문별 배정이 유용한가 |
| oracle | 모든 test 후보 outcome을 보고 최적 선택 | 배포 불가능한 비교 상한 |

random 예시: 우리 방법이 low/medium/high를 600/300/100회 사용하면 random도 그 비율을 쓰되 어느 질문에 어느 후보를 줄지 섞는다. 100개 seed의 평균·분포를 보고한다. 같은 호출 비율이어도 응답 길이가 달라 실제 비용은 다를 수 있다.

별도 사후 진단으로 test action 배열 자체를 shuffle할 수 있다. test 정답은 쓰지 않지만 배포 가능한 사전 정책과 구분하고 모델 선택에 재사용하지 않는다. 통합 실험에서는 source 안에서 shuffle하며 source×category까지 보존한 변형은 추가 진단으로 명명한다.

always-high의 주 지표는 기록된 답변 품질·비용이다. 모든 난이도를 hard라고 예측하는 상수 분류 baseline은 별도 진단이다. oracle도 품질 최대 또는 고정 utility 최대 등 목적을 명시하고 비용 동률을 포함해 규칙을 고정한다.

## 8. 논문 라우터

### RouterDC: 첫 공식 비교군

[공식 RouterDC](https://github.com/shuhao02/RouterDC)는 질문 encoder와 모델 embedding을 dual contrastive 방식으로 학습한다. 임의 후보 중 하나를 선택하므로 동일한 세 실제 후보에 적용할 수 있다. 보편적인 인간 인지 난이도 세 class를 학습하는 논문은 아니다.

원 RouterBench loader의 11개 후보 가정을 3개로 연결한다. original top-k positive/negative 설정을 그대로 쓰면 세 후보에서 겹칠 수 있으므로 분리·tie·후보 수를 검증한다. 공개된 다른 pool의 checkpoint 출력 이름만 바꾸지 않고 공통 train으로 재학습한다.

외부 trainer가 `test`라는 이름의 split로 best checkpoint를 고르더라도 여기에 연결할 것은 우리 validation이다. 진짜 test는 최종 evaluator에만 전달한다. [LLMRouter의 RouterDC trainer](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/routerdc/trainer.py)를 사용할 경우에도 이 경계를 확인한다.

공식 설정의 재현과 공통 encoder로 바꾼 ablation을 구분한다. 논문 방식 자체를 비교할 때 encoder가 다르면 총 compute·파라미터 수를 공개하며 동일 backbone 인과 실험이라고 부르지 않는다.

### 선택 확장

- [GraphRouter](https://github.com/ulab-uiuc/GraphRouter): query/task/model graph와 후보 수를 세 개로 구성하고 다시 학습한다. 새 LLM 결과 생성기를 실행하지 않고 공개 outcome으로 edge를 만든다. inference에서 정답 task metadata를 공짜로 주지 않으며 필요하면 task predictor 비용을 포함한다.
- [CARROT](https://github.com/somerstep/CARROT): 후보별 품질·비용 예측 방식으로 label-voting kNN과 구별한다. 기본 준비 코드의 OpenAI embedding 호출·key 의존을 제거한 local-only adapter가 필요하다.
- RouteLLM 원형의 두 후보 방법: 독립적인 2-action 범위에서 모든 비교군의 후보·학습 데이터를 같이 제한한다. 이 결과를 세 후보 주 결과와 직접 혼합하지 않는다.

## 9. 공정 학습과 실행 예산

| 항목 | 주 비교 기준 |
| --- | --- |
| 입력 | 동일 정제문, neural encoder 최대 512 token |
| 초기화 | pinned pretrained revision; R0/R1/R2 동일 backbone |
| train distribution | 자연분포, 일반 shuffle, effective weight 1 |
| canonical seed | 42 |
| finalist 반복 | 전체에서 최대 2개 experiment/method; 최소는 B의 validation nAUC 상위 2개에 seed 17, 73 추가 |
| neural 기본 상한 | 3 epochs, 동일 optimizer step 상한 |
| batch | micro 8 × accumulation 4 = effective 32; eval 16 |
| optimizer | AdamW, LR `2e-5`, weight decay .01, warmup .10, clip norm 1 |
| precision | 지원 CUDA에서 fp16; 실제 device/runtime 기록 |
| pilot | 최대 train 1,200행 / validation 300행 / optimizer 40 step |

여기서 512 token은 tokenizer별 길이 제한이며 동일한 문자 수가 아니다. 각 모델의 truncation 비율·영역·잘린 선택지/조건을 기록한다. TF-IDF는 현 구현상 전체 정제문을 받으므로 입력량 차이를 공개한다. encoder와 동일 text window를 쓰는 추가 진단을 할 때 window 규칙을 사전 고정하고 별도 결과로 둔다.

모델별 소규모 LR 튜닝을 추가한다면 같은 trial 수·validation 접근 횟수·step 예산으로 제한한다. 한 모델만 더 많은 epoch나 LR 후보를 허용하지 않는다. 실제 GPU 시간·CPU 시간·peak memory·총/학습 파라미터 수·artifact/index 크기를 함께 기록한다. 큰 모델이 당연히 이긴다는 가정을 두지 않는다.

R0/R1/R2의 primary 목적함수 비교는 고정 training budget의 종료 checkpoint를 사용한다. epoch 선택을 추가할 경우 세 방법에 동일 validation checkpoint-selection 규칙을 적용하고 새 run contract로 남긴다. 표현 비교는 validation 분류 성능으로, model-selection 비교는 validation 비용–품질 기준으로 선택하므로 동일 가중치 artifact를 재사용하더라도 선정 근거를 구분한다.

1. 데이터·candidate pool·split·T1·공통 가중치 계약을 검증한다.
2. 경량 기준선을 실행하고 neural/외부 adapter pilot으로 row 정렬·loss·메모리·시간을 확인한다.
3. ModernBERT R0/R1/R2와 RouterDC의 canonical run을 순차 수행한다. 통합 A와 source-only B의 run을 구분한다.
4. B의 validation nAUC로 finalist 최대 2개를 정해 seed 반복한다. A/표현 비교는 최소에서 seed 42 결과임을 명시하며 추가 반복은 확장 예산으로 분리한다. T2는 정해진 최소 범위만 수행한다.
5. 설정을 lock한 뒤 sealed test와 RouterArena 외부 진단을 실행한다.

이 순서는 GPU 학습이나 유료 API 호출을 이미 수행했다는 의미가 아니다. 비용 pilot 없이 full sweep의 시간·가격을 확정하지 않는다.

## 10. 구현 상태와 완료 기준

현재 소스: [TF-IDF](../../ml/src/llm_router_labeling/comparison/tfidf_baseline.py), [목적함수](../../ml/src/llm_router_labeling/objective_comparison/objectives.py), [trainer](../../ml/src/llm_router_labeling/objective_comparison/training.py), [평가](../../ml/src/llm_router_labeling/objective_comparison/evaluation.py), [목적함수 config](../../ml/config/routing_objective_comparison_v1.json).

현재 core/adapter와 최종 계약의 차이는 [실행 차이 목록](execution-and-artifacts.md#7-현재-구현과-최종-명세의-차이)에서 관리한다. 구현된 method enum 전체를 최소 실행 범위로 해석하지 않는다.

완료된 run에는 데이터·config·모델 revision hash, 후보 ID, seed, 학습 시간, validation 선정 근거, 봉인된 test prediction이 있어야 한다. test outcome이 trainer 입력으로 들어갔거나 duplicate group이 split을 넘은 run은 주 비교에 사용하지 않는다. 인지 수준 정렬은 [평가](evaluation.md)의 보조 실험이며 이 결과만으로 라우터의 내부 판단 원리를 입증했다고 주장하지 않는다.
