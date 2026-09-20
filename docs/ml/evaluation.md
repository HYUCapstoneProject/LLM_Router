# 최종 실험 평가 계약

기준일: 2026-09-20. 이 문서는 최종 실험의 평가 기준이다. 사양 확정과 평가 실행 완료는 구분한다. 연구 전제와 실행 순서는 [ML 문서 안내](README.md), 학습 데이터는 [데이터·전처리](data-and-preprocessing.md), 비교군과 학습 조건은 [라우팅 비교](routing-experiments.md)를 따른다.

## 1. 서로 다른 세 평가 축

| 평가 축 | 근거 | 주 지표 | 허용되는 해석 |
|---|---|---|---|
| 난이도 분류 | 봉인된 성공률 기반 3단계 라벨 | source-macro-F1, balanced accuracy, QWK | 학습 목표의 일반화 |
| 모델 선택 | 질문×실제 후보 모델의 공개 outcome·cost | 비용별 품질, 품질별 비용, regret | 주어진 공개 후보 pool에서의 선택 성능 |
| 시스템 | 시간 순서의 요청·cache·feedback 기록 | 전체 품질·비용·호출·지연·망각 | 반복과 변화 조건에서의 시스템 효과 |

RouterArena와 인지 수준 감사는 외부 정렬 진단이다. 같은 `easy/medium/hard` 이름을 사용해도 라벨 생성 원리가 다르다. 분류 confidence는 선택 LLM의 정답 확률이 아니며, 분류 정확도를 실제 Claude 답변 정확도로 바꾸어 표기하지 않는다. 성공률 예측·pairwise·논문 router는 행동 선택이 주목적이므로 난이도 일치율만으로 우열을 정하지 않는다.

## 2. 데이터 봉인과 비교 단위

1. 원본 revision/checksum, 전처리, 라벨 규칙, split, 중복 group, 후보 pool과 비용표를 실행 manifest로 고정한다.
2. 모든 비교군은 같은 질문과 같은 원천 outcome을 사용한다. 각 기법의 target 표현은 달라도 된다.
3. 후보 선택과 모델 fitting은 train에서, checkpoint·threshold·penalty·calibration·winner 선택은 validation에서만 한다. train-only OOF로 정한 라벨 threshold는 외부 평가로 다시 조절하지 않는다.
4. test와 RouterArena는 정책을 동결한 뒤 최종 평가한다. 결과를 보고 모델·임계값을 바꾸면 새 개발 단계로 기록하고 새로운 미사용 holdout이 필요하다.
5. RouterArena를 epoch별 의사결정, 반복 승격 gate, feedback 생성에 쓰지 않는다. 재학습 승격은 별도 `D_gate`를 사용한다.
6. 후보 pool이 다른 RouterBench/xRouteBench는 source별 결과를 먼저 낸다. source macro 평균은 두 pool을 같은 실제 모델 세 개로 만든다는 뜻이 아니다.
7. 난이도 test는 자연분포를 보존한다. 균형 표본 진단을 추가한다면 원래 test와 구분한다.
8. source/category/domain·정답·모델 outcome·실제 생성 token 수는 평가 metadata다. query-only router의 입력으로 제공하지 않는다.

질문·근접 중복·변형을 같은 group으로 다룬다. 데이터 누수 제거는 [공통 전처리](data-and-preprocessing.md)의 보호 우선순위를 따른다. 전체 학습·튜닝 데이터에 대한 누수 검사 없이 smoke의 overlap 0건을 최종 비오염 증거로 사용하지 않는다. 검사의 입력 artifact hash·실행 여부·겹친 ID 목록을 남긴다. RouterArena 입력이 optional인 builder가 반환한 overlap 0건만으로 검사가 실제 실행되었다고 판단하지 않는다.

## 3. 난이도 분류와 표현·모델 비교

라벨 순서는 `easy=0, medium=1, hard=2`다. checkpoint의 `id2label`과 config를 검증하고, 집합이 다르면 실패시킨다. 기본 `LABEL_0/1/2`는 봉인한 `label_order`로 해석한다.

필수 결과는 accuracy, macro/weighted-F1, class별 precision/recall/F1/support, balanced accuracy, QWK, ordinal MAE, under/over-routing, `hard→easy`, confusion matrix다. `hard→easy` 비율은 실제 hard 수를 분모로 쓰며 hard가 없으면 N/A로 보고한다. 클래스가 없는 slice의 수치와 전체 3-class 결과를 같은 증거 수준으로 비교하지 않는다.

전체 행 가중 평균과 source별 결과, source별 macro-F1의 산술평균을 함께 보고한다. domain/category는 원 source 의미를 유지하고 없는 metadata를 추정해 채우지 않는다. 작은 cell의 support를 항상 표시한다. 기존 코드의 class별 최소 100/30행에 따른 `confirmatory/exploratory/insufficient_coverage`는 표본 커버리지 표기이며, 통계적 유의성을 자동 보장하는 이름이 아니다.

표현 비교의 validation 요약값은 다음과 같다. 개별 지표를 숨기거나 품질·비용의 단일 절대 척도로 사용하지 않는다.

```text
Q_model = 0.45 × source_macro_F1
        + 0.15 × balanced_accuracy
        + 0.15 × (QWK + 1) / 2
        + 0.15 × (1 − hard_to_easy_rate)
        + 0.10 × (1 − ECE)
```

표현 비교의 최소 결과는 seed 42의 validation Q_model로 정책을 동결한다. 차이가 0.005 미만이면 hard→easy 비율, p95 지연, peak memory, artifact 크기 순으로 낮은 방법을 우선한다. 이 0.005는 사전 선택 규칙이며 통계적 동등성의 증명이 아니다. 결정적인 TF-IDF/kNN을 seed 이름만 바꿔 반복한 결과를 독립 학습 증거로 세지 않는다.

학습 목표 비교의 선정 지표는 source별 비용–품질 nAUC의 macro 평균이며, 단일-source 논문 비교는 해당 source의 nAUC다. Q_model로 model-selector winner를 바꾸지 않는다. 추가 학습 seed 예산은 전체 실험에서 최대 두 `(experiment_id, method)`에만 배정하며, 최소 실험에서는 실제 세 후보를 공유하는 B의 validation nAUC 상위 두 방법에 seed 17/73을 추가한다. 정확한 동률은 validation 평균 비용, p95 지연, 고정 method ID 순으로 결정한다. 반복한 방법은 세 seed 평균 validation nAUC로 최종 정책을 동결하고 test 결과로 재선정하지 않는다. A/표현 비교까지 반복하려면 별도 확장 예산으로 기록한다. 서로 다른 축의 winner가 같을 필요는 없다.

모델 크기·GPU 시간·p95 지연·메모리·artifact 크기를 함께 보고하고, 크기가 다른 모델의 차이를 학습 목표의 효과로 단정하지 않는다. 학습 목표 비교는 동일 backbone으로 별도 수행한다. 외부 평가에서 예상과 다른 결과가 나와도 그 점수로 runner-up을 새 winner로 바꾸지 않는다.

### Calibration

- Q_model의 ECE는 모든 3-class 표현 방법의 **미보정 class confidence**로 계산한다. LR/neural은 class probability, kNN은 비음수 가중 득표를 합계 1로 정규화한 값이다. 고정 15개 등간격 bin과 reliability diagram으로 실제 라벨 일치와 비교한다. kNN 득표를 검증된 정답 확률이라고 부르지 않는다.
- neural classifier의 temperature 보정은 별도 calibration 진단이다. validation에 5-fold cross-fitting을 적용하고 test 전 최종 temperature는 전체 validation에 한 번 적합한다. `calibration.json`에 fit split hash를 기록하며 보정 전·후를 함께 보고한다. 보정 후 ECE로 일부 방법의 Q_model만 교체하지 않는다.
- pairwise의 임의 scale score는 class probability가 아니므로 같은 ECE에 넣지 않는다.
- R1은 모델별 binary success probability이므로 후보별 calibration을 별도로 평가한다. 3-class softmax ECE와 같은 수치로 순위를 매기지 않는다.
- calibration 구현이 없는 run은 raw 지표만 보고하며, 문서에 적힌 보정 정책을 수행한 것처럼 표시하지 않는다.

## 4. 공개 outcome을 이용한 모델 선택 평가

질문 i, 선택 후보 a_i에 대해 품질은 `q[i,a_i]`, 실제 기록 비용은 `c[i,a_i]`다. tier 그룹의 평균 성공률을 실제 단일 모델 1회 응답으로 간주하지 않는다. 주 논문 비교는 같은 source의 실제 후보 세 개를 고정한다. source-local symbolic tier 투영 실험은 별도 이름으로 보고한다.

결정 비용과 평가 비용을 구분한다. 선택 시 사용할 수 있는 비용은 공개 가격·입력 길이·train에서 추정한 출력 길이 등이며, 선택 후 알 수 있는 실제 출력 token 수를 미리 제공하지 않는다. 평가에는 기록된 비용을 사용한다. 출처·시점·통화가 불명확하면 정규화 비용으로 표시하고 달러 절감이라고 부르지 않는다.

| 결과 | 정의와 제한 |
|---|---|
| quality–cost curve | validation에서 동결한 정책 집합의 test 품질·실제 비용 점 |
| budget별 품질 | 해당 budget용으로 validation에서 고른 penalty를 그대로 적용 |
| 품질 목표별 비용 | 목표 정책을 validation에서 선택; test 목표 미달도 그대로 보고 |
| oracle regret | 모든 후보 결과를 보는 oracle과 품질·효용 차이; 배포 가능 방법 아님 |
| 후보 사용량 | source별 low/medium/high 비율과 실제 cost |
| source/domain 분석 | 이득·손해가 어느 문제 집합에 모이는지 paired 비교 |

test에서 가장 좋아 보이는 penalty를 사후 선택한 frontier는 주 성능으로 사용하지 않는다. validation budget은 test의 실제 비용 상한을 보장하지 않으므로 test cost와 초과 여부를 보고한다. 목표를 만족하는 정책이 없으면 미달로 남긴다.

비용 정규화와 nAUC는 기준 후보와 범위가 같은 실험 안에서만 비교한다. 항상 강한 모델이 항상 싼 모델보다 품질이 좋다는 보장은 없다. 비용/품질 기준점의 차가 0 이하이면 해당 정규화 결과는 N/A로 두고 raw 품질·비용을 보고한다. oracle의 동률은 가장 싼 후보, 이후 고정 ID 순서로 처리한다.

### 고정·무작위 기준선

- `always-low/medium/high`: 해당 후보의 실제 공개 답변 점수·비용. 전부 같은 difficulty를 예측하는 분류 기준선과 구분한다.
- `best-single`: test를 보기 전에 train 평균으로 고정한 단일 후보. validation 선택 변형을 쓰면 이름·선정 split을 따로 기록한다.
- `matched-random`: validation의 후보 사용 비율을 고정하고 source 안에서 무작위 배정한다. 100개 seed의 평균·분포를 보고한다.
- 사후 설명용 `action-permutation`: test의 action 개수만 보존해 질문 배정을 섞는다. 결과를 모델 선택에 쓰지 않으며 정답을 섞는 실험이 아니다.
- 같은 호출 비율은 같은 실제 비용을 뜻하지 않는다. source×domain 비율까지 보존한 permutation은 별도 진단이다.

신규 Claude 응답을 생성하지 않은 replay에서는 현재 Haiku/Sonnet/Opus의 품질을 측정했다고 주장하지 않는다. 실제 서비스 검증을 추가할 경우 별도 model ID·prompt·thinking/effort·출력 제한·채점기·가격 snapshot을 고정하고 학습 결과와 구분한다.

## 5. RouterArena 외부 평가

### 공식 난이도의 의미

[공식 논문 §3](https://arxiv.org/html/2510.00202v2)는 DeepSeek-V3.1의 Bloom 판정에서 remembering/understanding→easy, applying→medium, analyzing/evaluating→difficult로 묶고 create를 제외했다고 설명한다. **42개 모델의 정답 수 threshold라는 해석은 사용하지 않는다.** [공식 저장소](https://github.com/RouteWorks/RouterArena)의 evaluation-only 조건을 지킨다.

배포 Parquet에는 이미 `easy/medium/hard`인 `Difficulty`가 있고 질문별 6단계 라벨은 없다. 3단계 라벨에서 6단계를 역추정하거나 임의로 2/2/2 묶음을 재생성하지 않는다. 공식 benchmark의 선택 LLM 답변 정확도·비용과 우리의 난이도 정렬 평가는 다르다. 결과 이름은 `RouterArena difficulty-label alignment`이며 공식 Arena Score가 아니다.

### 고정 배포본과 전처리

```text
repository: RouteWorks/RouterArena
revision: a4a062ce3313b56bb09c042e1bc37b61d34e3bd8
dataset_version: routeworks-routerarena-a4a062ce3313b56bb09c042e1bc37b61d34e3bd8
```

| split | 원본 행 | SHA-256 | 역할 |
|---|---:|---|---|
| full | 8,400 | `358a6bb4160320b5cb5b68d6d0a885c89b128b06d3dea267d77ed6daeebe4369` | 최종 외부 평가 |
| sub_10 | 809 | `e7b4b7579e98df0b83202324357d0b91af78666ddbf1038a513d19edaf58efd5` | smoke |
| robustness | 420 | `6012eeea11e2d7b2018155e65c35ad887e9daa70e45aa38a40321da74d0cc2e9` | 변형 일관성 |

`sub_10`과 robustness의 ID는 full에 포함되므로 독립 표본으로 합산하지 않는다. 원본 schema는 `Category, Domain, Dataset name, Global Index, Context, Question, Options, Answer, Metadata, Keywords, Difficulty`다.

- HTML entity 복원, Unicode NFKC, 줄바꿈·제어문자 정규화를 적용한다. 입력은 `Context:` 블록, 질문, `A)/B)` 선택지 순으로 조립한다.
- Answer/Difficulty/source/domain/category를 입력에 넣지 않는다. 빈 질문, 동일 입력 중복, 거절 placeholder, 변형 끝의 embedded boxed answer를 제외하고 원인과 원본 ID를 보존한다.
- robustness의 빈 Difficulty는 `Global Index`로 full에서 상속한다. join 실패가 하나라도 있으면 중단한다. 원문을 `reference_model_input_text`로 보존한다.
- 전처리에서 자르지 않고 추론에서 학습과 같은 tokenizer·max length를 적용한다. 최종 공통 비교는 **512 tokens**, 기존 운영 8192-token checkpoint 진단은 별도 run이다.
- RouterBench+xRouteBench 전체 개발 데이터와 cross-source 누수를 검사한다. 동일 원본 benchmark/근접 중복 검사 결과까지 manifest에 기록한다.

평가 projection은 `sample_id, model_input_text, difficulty_label, evaluation_split, source_sample_id`다. 정답과 원본 정보는 별도 metadata에 저장하고 모델에는 `model_input_text`만 전달한다.

### 세 slice와 결과 해석

| slice | 선정 규칙 | 용도 |
|---|---|---|
| rb_service_aligned | 아래 unseen을 제외한 text reasoning/knowledge/math/code | 주 외부 정렬 |
| rb_unseen | WMT19, 음악, 체스, NarrativeQA | OOD 진단 |
| full_clean | 공통 누수·품질 필터를 통과한 전체 | 전체 참고 |

slice는 예측과 Difficulty를 보기 전 source와 서비스 범위로 고정한다. 최신 통합 데이터와의 누수 제거가 행 수를 바꾸므로 옛 clean 8,375/8,397행 수치를 최종 입력 크기로 복사하지 않는다. 각 slice의 행·라벨·source 분포를 새 manifest에서 계산한다. 공식 리더보드용 프로토콜과 custom clean 결과를 혼용하지 않는다.

주 그림은 `P(Arena label | predicted group)`의 count·행 정규화 3×3 표와 전체 라벨 비중 대비 증감이다. 반대 조건부 분포도 함께 낸다. macro-F1·balanced accuracy·QWK·ordinal MAE·`hard→easy`를 보조 정렬 지표로 보고한다. 여기서 under/over-routing은 **외부 라벨 대비 방향성**이며 Claude 품질 부족이나 비용 낭비를 직접 입증하지 않는다.

robustness는 원문-변형 예측의 유지율·방향 변화와 두 입력의 라벨 정렬을 paired로 비교한다. 이미 공식 라벨을 상속한 변형의 의미 보존 여부 및 품질 필터를 함께 보고한다. action selector를 상/중/하로 투영하는 경우 별도 표시하고, RouterArena에 존재하지 않는 후보 outcome이나 비용을 만들어내지 않는다.

## 6. 인지 수준 보조 감사

연구 질문은 “성공률 기반 라벨과 그 라벨로 학습한 모델 출력이 독립적인 인지 수준 판정과 얼마나 정렬되는가?”다. GPT는 **평가용 주석자**이며 새 학습 label 공급자가 아니다. 감사 결과를 보고 라벨 threshold나 모델을 재선택하지 않는다.

### 표본·블라인드 주석

- 제안 시작 규모: RouterBench/xRouteBench × easy/medium/hard × 50문항 = 300문항. 각 cell에서 domain을 고려해 무작위 표본한다. 규모와 비용은 실행 전 봉인할 값이며 현재 측정값이 아니다.
- 같은 표본으로 모델 일반화도 평가하려면 학습·튜닝에 사용하지 않은 test에서 뽑는다. train 표본은 라벨 성격 분석만 가능하다.
- GPT에는 질문과 필요한 context/options만 제공한다. 기존 difficulty, 성공률, router 예측, source/dataset 이름은 숨긴다. 글자 수나 희귀 지식을 높은 인지 수준의 근거로 삼지 않도록 rubric에 명시한다.
- `Remember/Understand/Apply/Analyze/Evaluate/Create/판정 불가` 중 판정과 짧은 근거를 수집한다. prompt/model version·sampling 설정·입력 hash·응답을 보존한다. 자기보고 confidence를 사실상 정답 확률로 쓰지 않는다.
- 무작위 60문항을 두 사람이 GPT 결과와 기존 label을 보지 않고 독립 판정하는 것을 제안한다. 불일치 문항만 골라 검토하지 않는다. 전체와 영역별 합의도·불일치 유형을 기록한다.
- 3단계 비교에서는 기억·이해→하, 적용→중, 분석·평가→상으로 사전 매핑한다. Create/판정 불가는 별도 집계하며 임의로 상에 넣지 않는다.

RouterArena의 judge prompt는 정답과 dataset 정보도 포함한다. 위 감사는 같은 Bloom 체계를 사용하지만 입력 정보를 제한한 독립 절차이며 공식 주석의 완전 재현이라고 부르지 않는다. [논문 부록 D.1](https://arxiv.org/html/2510.00202v2#A4.SS1)

### 비교와 통제 문항쌍

| 비교 | 답하는 질문 |
|---|---|
| 기존 label ↔ 인지 주석 | 성공률 기반 데이터 구분과 인지 수준의 관계 |
| 모델 prediction ↔ 인지 주석 | 보지 않은 질문에서 출력의 인지 수준 정렬 |
| 모델별 paired 차이 | 동일 문항에서 어느 표현·목표가 더 정렬되는가 |
| 사람 ↔ GPT | 자동 주석 자체의 신뢰도와 한계 |

3×3 표, 순서 가중 일치도, source/domain별 결과와 CI를 보고한다. 균등 층화 300문항의 비율은 자연분포 비율이 아니므로 층별 결과와 원 모집단 비중으로 가중한 결과를 구분한다. 기억 문제도 희귀 지식이면 실패율이 높을 수 있고 분석 문제도 충분한 맥락이면 쉬울 수 있다. 낮은 일치도를 곧 라벨링 실패로 해석하지 않는다.

추가 통제쌍은 (a) 같은 소재에서 사실 확인→규칙 적용→근거 분석으로 실제 요구를 바꾼 쌍, (b) 뜻을 유지하고 길이·문체만 바꾼 쌍, (c) “분석하라” 같은 단어만 넣고 실제 요구는 바꾸지 않은 쌍이다. 사람이 요구 수준을 검토하고 주제·길이·배경지식 차이를 최대한 통제한다. 첫 유형은 예상 방향 변화, 나머지는 불필요한 예측 변화를 측정한다. 원질문 group 단위로 split/CI를 처리한다.

외부 라벨 일치와 통제쌍 반응은 행동적 근거다. 모델 내부가 Bloom 인지 단계를 추론한다는 메커니즘 증명으로 확대하지 않는다. [LLM judge 연구](https://arxiv.org/abs/2306.05685), [CheckList 행동 평가](https://aclanthology.org/2020.acl-main.442/).

## 7. 통계·효율·재현성

- 동일 질문의 예측을 ID로 결합하고 paired 차이를 분석한다. 기본 bootstrap은 10,000회, seed 42, 95% CI다. source와 가능한 label 층을 유지하되 근접 중복·변형·반복 요청은 **group 단위**로 resample한다.
- seed 42/17/73의 학습 변동과 질문 bootstrap의 표본 변동을 별도 보고한다. 한 seed의 CI가 모든 학습 불확실성을 대체하지 않는다.
- 미리 정한 주 비교와 탐색적 domain slice를 구분한다. 많은 slice 중 좋은 결과만 선택하지 않는다. CI가 0을 포함하면 우월성을 확정하지 않으며, 포함하지 않는다고 실용적 이득이 자동 확보되지 않는다.
- 지연은 같은 hardware/runtime/precision에서 warmup 후 측정한다. batch 1의 요청 지연과 batch 처리량을 구분하고 tokenizer·embedding·search·model forward·전체 router를 분해한다.
- GPU/CUDA 동기화와 warm/cold 조건을 기록한다. batch elapsed를 batch size로 나눈 수치를 batch-1 p95라고 쓰지 않는다. 모델 loading 시간은 별도다.
- neural/embedding 공통 512-token 비교에서 원 token 수, truncation 비율, 잘린 위치/라벨/domain slice를 보고한다. TF-IDF의 전체 정제문 입력과 tokenizer별 길이 차이는 [라우팅 비교](routing-experiments.md)에 명시한 조건으로 별도 보고한다.
- inference 가능한 데이터에서만 비교하지 않고 실패·OOM·결측·timeout의 수와 분모를 보고한다. 방법마다 다른 test 부분집합으로 만든 평균을 공정 비교라 부르지 않는다.

캐시의 그룹 통계·unknown 품질 범위는 [캐시](semantic-cache.md), feedback의 prequential·망각·승격 지표는 [재학습](retraining-pipeline.md)을 따른다.

## 8. 산출물과 현재 구현 경계

run별 `predictions.parquet`, `metrics.json`, `model_info.json`, dataset/validation lock, calibration, runtime, 실패 내역, 그림을 저장한다. row에는 sample/group/source/split, actual label, prediction/score, selected model, 공개 outcome/cost, token/truncation, 단계별 latency와 policy version을 둔다. answer와 민감 원문은 필요 범위의 별도 artifact에 보존한다. 세부 경로와 보존 정책은 [실행·산출물](execution-and-artifacts.md)을 따른다.

평가 코드 진입점은 [분류 evaluator](../../ml/src/llm_router_labeling/comparison/evaluation.py), [목표별 evaluator](../../ml/src/llm_router_labeling/objective_comparison/evaluation.py), [선정·봉인 평가](../../ml/src/llm_router_labeling/objective_comparison/selection.py), [RouterArena 전처리](../../ml/src/llm_router_labeling/evaluation/routerarena_preprocessing.py), [RouterArena 추론](../../ml/src/llm_router_labeling/evaluation/routerarena_routing_evaluator.py)이다. 각 구현에서 최종 프로토콜까지 남은 작업은 [실행 차이 목록](execution-and-artifacts.md#7-현재-구현과-최종-명세의-차이)에 모은다.

기존 config의 RouterArena max length 8192와 epoch 관찰, 운영 gate 코드를 그대로 실행한 결과를 이 최종 프로토콜을 충족한 것으로 표시하지 않는다. 문서 통합은 학습·평가 실행이나 운영 코드 변경이 아니다.
