# 졸업 프로젝트 전제와 공정 비교 실험 계약

기준일: 2026-09-20. 상태: 사용자 확정 전제와 이를 구체화한 실험 설계. 성능 결과·실험 실행 완료를 의미하지 않는다. 프로젝트 결정 원장은 memorize이며, 이 문서는 재현 가능한 실험 사양이다.

## 1. 프로젝트의 변하지 않는 연구 질문

> 어떤 조건에서 임베딩 캐시와 에스컬레이션 피드백 학습을 결합한 라우팅 시스템이 더 나은 비용–품질 균형을 만드는가?

시스템의 세 축은 **임베딩 캐싱, 질문 기반 라우팅, 에스컬레이션 데이터에 의한 재학습**이다. 특정 backbone이나 특정 Claude 세대의 우승은 전제가 아니다. 개선이 없는 조건과 품질·비용 손해가 발생하는 조건도 결과로 보고한다.

- Claude Haiku/Sonnet/Opus는 실제 서비스를 구현할 때 호출하는 대상이다. 신규 Claude 응답·judge·선호 데이터를 대량 생성해 학습하지 않는다.
- 학습과 비교는 공개된 RouterBench/xRouteBench의 동일 질문·모델 결과를 재사용한다. 원천에 이미 포함된 과거 Claude 결과는 공개 benchmark의 일부이며 새로운 API 호출과 구분한다.
- 같은 질문·split·후보 action·비용·평가 규칙을 공유한다. 기법이 사용하는 target의 표현 차이는 허용하고 명시한다.
- RouterArena는 외부 진단용이다. 초기학습, feedback, threshold 선택, winner 선택, 반복 승격 gate에 사용하지 않는다.
- 최소 라우팅 비교, 캐시 실험, 피드백 실험을 우선한다. 모든 backbone×목적함수×seed 조합을 먼저 실행하지 않는다.
- 현재 운영 모델과 설정은 이 문서 작성만으로 변경하지 않는다. 실험은 별도 artifact/version/pointer를 사용한다.

상세 설계: [에스컬레이션](escalation-offline-experiment-plan.md), [캐시](semantic-cache-experiment-plan.md).

## 2. 먼저 구분할 세 가지 성능

| 질문 | 정답/평가 근거 | 주요 지표 |
|---|---|---|
| 질문 난이도를 상·중·하로 분류하는가? | 봉인된 benchmark 유래 difficulty label; RouterArena는 별도 외부 라벨 | macro-F1, balanced accuracy, confusion matrix, quadratic weighted kappa |
| 같은 예산에서 좋은 모델을 선택하는가? | 질문×후보 모델의 실제 공개 점수·비용 | quality–cost curve, budget별 품질, quality target 비용, oracle regret |
| 캐시와 feedback을 더하면 어떤가? | 시간 순서 요청 replay, 실제 선택·오답·업데이트 이력 | 총 품질·총 비용·호출 수·지연·적응·망각 |

상/중/하 label이 같은 것은 같은 측정 대상을 뜻하지 않는다. 분류 confidence도 선택 LLM의 정답 확률이 아니다. 성공률 예측·논문 router의 action을 easy/medium/hard로 이름만 바꾸어 난이도 정답과 동일한 것으로 처리하지 않는다.

### RouterArena의 기존 설명 오류 정정

[RouterArena 논문 v2 §3](https://arxiv.org/html/2510.00202v2)는 DeepSeek-V3.1이 Bloom 인지 수준을 판정하고 remembering/understanding→easy, applying→medium, analyzing/evaluating→difficult로 묶었다고 설명한다. 이는 RouterBench 후보 모델 성공률 threshold가 아니다. 기존 프로젝트 자료의 “42개 모델 중 정답 수 4/19 경계로 RouterArena Difficulty 생성” 설명은 이 공식 라벨의 근거로 사용하지 않는다.

공식 benchmark는 선택된 모델의 답변 정확도와 비용을 평가한다. 우리 `Difficulty` 예측 일치율은 그중 **외부 난이도 정렬 진단**이며 공식 Arena Score를 재현한 것이 아니다. [공식 저장소](https://github.com/RouteWorks/RouterArena)는 evaluation-only 사용을 요구한다.

RouterArena와 잘 맞는다고 현재 Claude tier가 충분하다는 결론은 나오지 않는다. 반대로 잘 맞지 않는다고 유용한 cost-aware routing이 불가능하다는 결론도 나오지 않는다.

## 3. 현재 라벨링을 정확하게 기억하기

현재 저장소에는 서로 다른 계약이 공존한다.

| 계약 | 변환 | 위치·상태 |
|---|---|---|
| 운영 v2 | low 성공률≥0.67이면 easy, 아니면 medium≥0.67이면 medium, 아니면 high≥1/3이면 hard; unresolved 제외 | `routerbench_labeling_v2.json`, 현행 feedback base contract |
| RouterBench v3 | 전체 solve rate≥0.64 easy, ≥0.27 medium, 그 외 hard | `routerbench_labeling_v3.json`, 별도 자연분포 실험 |
| 통합 데이터 로컬 artifact | train-only 탐색에서 T1 hard=0.27 / easy=0.55, T2 hard=0.36 / easy=0.55 | 2026-09-20 읽은 `router-model-comparison-v1/dataset/manifest.json`; 운영 적용 아님 |

통합 artifact의 현 행 수는 train 31,417 / validation 3,907 / RouterBench test 3,472 / xRouteBench test 3,729다. 이는 이번 읽기에서 확인한 데이터 상태이며 새 continual split의 크기로 재사용해서는 안 된다. 새로운 feedback split은 별도로 봉인한다.

로컬 v2 tier 파일에는 low에 Mistral-7B·Claude Instant v1·CodeLlama-34B·WizardLM-13B, medium에 Claude v1/v2·Llama2-70B·GPT-3.5, high에 Mixtral-8x7B·Yi-34B·GPT-4-1106-preview가 들어 있다. xRouteBench 후보는 Qwen·Gemma·Llama·Mixtral·GPT-OSS·DeepSeek 등의 별도 18개 모델이다. **두 데이터셋이 같은 low/medium/high 실체를 공유하지 않는다.**

학습/평가 비용은 source별 원래 모델 pool 안에서 계산한다. source가 다른 model ID를 하나의 실제 LLM처럼 합치거나, 그 달러 비용을 현재 Claude 가격으로 대체하지 않는다.

## 4. Claude 서비스 모델 선택

2026-09-20 공식 문서에서 확인한 현재 Haiku/Sonnet/Opus 계열의 서비스 기본 제안은 다음과 같다. 이는 과거 benchmark와 능력이 일치한다는 추천이 아니라 현재 서비스 역할에 대한 mapping이다.

| 서비스 route | 모델 / Claude API ID | 표준 입력·출력 가격, USD/백만 tokens |
|---|---|---|
| easy / 저비용 | Haiku 4.5 / `claude-haiku-4-5-20251001` | 1 / 5 |
| medium / 중간 | Sonnet 5 / `claude-sonnet-5` | 2 / 10 |
| hard / 상위 | Opus 5 / `claude-opus-5` | 5 / 25 |

출처: [모델 목록](https://platform.claude.com/docs/en/models/overview), [가격](https://platform.claude.com/docs/en/about-claude/pricing), [상태·폐기 일정](https://platform.claude.com/docs/en/about-claude/model-deprecations), [모델 ID 규칙](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions). 실제 계정의 model access는 호출 전 확인한다. 일반 token 가격이며 prompt-cache·batch·도구·리전 요금은 포함하지 않았다.

같은 4.5 세대로 고정한 별도 서비스 비교가 필요하다면 `claude-haiku-4-5-20251001`, `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`도 위 상태 문서상 active다. 그러나 이 조합이 과거 RouterBench의 능력 간격에 더 가깝다는 공통문항 근거는 없고 lifecycle도 다시 확인해야 하므로 기본 추천을 바꾸지 않는다. 과거 Haiku 3/3.5와 Sonnet 3.7은 native Claude API에서 retired여서 역사적 난이도 재현 목적으로 선택하지 않는다.

[Haiku 4.5 발표](https://www.anthropic.com/news/claude-haiku-4-5)는 일부 coding 성능이 Sonnet 4에 근접한다고 설명한다. 세대가 바뀌면 작은 모델도 과거 어려운 문항을 해결할 수 있다는 단서이지, 서로 다른 benchmark 전체의 등가성을 입증하지는 않는다. 구형 모델과의 난이도 스펙트럼에 “얼추 맞는” 현 Claude 조합을 공개 평균 점수만으로 확정할 수 없다.

서비스 검증은 학습과 분리한다. 모델 ID, prompt, context, output limit, thinking/effort 정책을 동결하고 기존 서비스 로그 또는 선택적으로 소규모 동일 질문 평가에서 tier별 성공·비용을 관찰한다. 이것을 gradient/threshold fitting에 사용하지 않으면 학습의 공정성은 유지된다. 이 문서는 유료 호출 실행을 승인하거나 실시한 기록이 아니다.

## 5. 비교 축과 최소 비교군

### A. 표현/모델 비교: target은 같은 3-class

| 방법 | 무엇이 다른가? | 실험 역할 |
|---|---|---|
| TF-IDF + Logistic Regression | 단어/구문 빈도 특징 + 선형 경계 | 필수 저비용 기준선; 새 논문 라우팅 목적함수라고 부르지 않음 |
| Frozen BGE + kNN | 고정 의미 공간의 가까운 학습 질문 label 투표 | 필수 저비용 비모수 기준선 |
| ModernBERT + CE | 난이도 supervision으로 encoder와 head를 fine-tune | 우리 기본 방법 |
| RoBERTa-base + CE | 같은 목표·비슷한 encoder 규모의 다른 사전학습 모델 | near-size 비교 1개 우선; 예산 확보 후 |

ModernBERT-base 149M과 RoBERTa-base 약 125M은 같은 자릿수이지만 정확히 같은 크기는 아니다. DeBERTa-v3-base는 backbone 86M + embedding 약 98M을 합쳐 약 184M이므로 86M만 적어 동급이라고 주장하지 않는다. [ModernBERT 카드](https://huggingface.co/answerdotai/ModernBERT-base), [RoBERTa 카드](https://huggingface.co/FacebookAI/roberta-base), [DeBERTa 카드](https://huggingface.co/microsoft/deberta-v3-base).

파라미터가 커지면 해당 과제의 성능이 반드시 증가하는 것은 아니다. 다만 관찰된 차이를 architecture만의 효과라고 단정할 수도 없다. tokenizer, 사전학습 데이터/목표, 문맥길이도 함께 다르다. 공통 512-token 제한·같은 sample·같은 tune 횟수·같은 optimizer-step 상한과 실제 GPU 시간·총/학습가능 파라미터를 보고한다. 각 모델 최적화를 위한 작은 LR grid는 동일한 탐색 예산으로 허용한다. 긴 문맥 이점은 별도 실험으로 둔다.

### B. 학습 목표 비교: backbone은 ModernBERT로 고정

| 방법 | train의 공개 결과에서 만드는 target | 추론 출력 |
|---|---|---|
| R0 3-class CE | 결과를 threshold로 압축한 하나의 difficulty label | 세 difficulty class 확률 |
| R1 pointwise success | 후보별 성공 0/1 벡터 | 후보별 예상 성공 확률 |
| R2 pairwise | 같은 질문에서 후보 i와 j의 점수 차이·동률 | 후보의 상대 점수 |

Regret R3는 최소 실험에서 제외하고 기존 구현을 확장 후보로 보존한다. 기존 코드의 `ordinal` 이름은 일반 3-class cross entropy를 뜻하며 ordinal regression 전용 loss를 구현한 것은 아니다.

### 같은 성공률 데이터인데 왜 다른가?

같은 원천을 사용한다는 지적은 맞다. 차이는 **어떤 정보를 남긴 target으로 무엇을 최적화하는가**다. 설명용으로 세 모델 결과가 X=(1,0,1), Y=(0,1,1)이면 둘 다 전체 성공률 2/3이지만 X에서 저비용 모델 성공, Y에서 중간 모델 성공이라는 차이는 사라진다. 이 예시는 전체 solve-rate 라벨의 정보 손실 설명이며 운영 v2 tier-consensus 예시와 혼동하지 않는다.

R0는 압축 label 재현, R1은 모델별 성공도, R2는 상대 품질을 학습한다. R1의 inference 확률은 한 번 실행된 공개 0/1 결과에서 추정하는 값이지 모델의 참 성공확률을 측정한 것은 아니다. R2의 offline performance preference는 사람 선호나 새 Claude judge 결과가 아니다. exact tie는 동률 target으로 두고, 동일 quality 내 저비용 선호는 사전 고정한 decision rule로 처리한다.

모델 성능이 완전히 단조적이고 세 label이 최적 행동을 완전히 표현하는 상황이면 R0/R1/R2가 비슷해질 수 있다. 차이가 있다는 결론을 사전에 전제하지 않는다.

### BGE+kNN의 구체 동작

1. **train 질문만** pinned BGE로 1024차원 L2-normalized vector로 변환한다. 난이도 label과 함께 index에 저장한다. BGE weight는 업데이트하지 않는다.
2. validation에서 k∈{3,5,11,21}와 균등/거리 가중 투표 중 선택해 동결한다.
3. 새 질문을 같은 BGE로 변환하고 cosine 상위 k개 train 이웃을 찾는다.
4. easy/medium/hard 득표 합이 가장 큰 label을 반환한다. 동률 규칙도 사전 고정한다.

예를 들어 k=5 이웃 label이 easy/easy/easy/medium/hard이면 easy다. 비용 penalty가 필요하면 정규화 득표를 score로 사용하되 보정된 확률이라고 부르지 않는다. label lookup index는 항상 train-only이고, 응답 cache는 요청에 따라 바뀌는 별도 상태다. ModernBERT는 gradient로 질문 표현과 경계를 학습하지만 kNN은 기존 질문의 이웃 관계로 판단한다. 목적은 같아도 추정 방식·적응·메모리/검색 비용이 다르다. BGE도 큰 pretrained model이므로 “파라미터 0”이라고 보고하지 않는다.

## 6. 고정/무작위 baseline의 의미

`always-low`, `always-medium`, `always-high`, train 평균 성능으로 고른 `best-single`을 둔다. 가장 비싼 모델과 최고 평균 품질 모델은 같지 않을 수 있다. oracle은 질문의 모든 정답 결과를 보는 이론적 비교 상한이며 배포 가능한 router가 아니다.

주 평가에서 always-high는 모든 문제에 “hard”라는 정답 label을 붙여 F1을 재는 것이 아니다. **항상 같은 고성능 후보가 낸 기록된 답변 점수와 비용**을 집계한다. 난이도 분류의 constant-hard baseline은 따로 가능하지만 다른 실험이다.

### 호출 비율을 맞춘 random

우리 router가 1,000개에 low 600 / medium 300 / high 100을 배정했다고 하자. random도 총 600/300/100회를 쓰되 **어느 질문이 어느 모델을 받는지 무작위로 섞는다**. 우리 품질이 더 좋으면 모델 사용 비중 외에 질문별 배정이 유용했다는 증거다.

- 실행 가능한 주 baseline: validation에서 정한 비율을 test에 적용, seed 100개로 평균·분포 보고.
- 사후 설명용 permutation: test에서 우리 router가 만든 action 배열 자체를 shuffle. 정답 label은 사용하지 않으며 이 결과로 모델/threshold를 다시 고르지 않는다.
- 같은 호출 비율도 실제 token 비용은 다를 수 있으므로 **동일 비용 보장**이라고 쓰지 않는다. 실제 cost를 따로 보고 budget-matched curve도 비교한다.
- RouterBench와 xRouteBench처럼 candidate pool이 다르면 source 안에서 shuffle한다. source×domain까지 보존한 random은 더 엄격한 별도 진단으로 명시한다.

## 7. 3개 모델을 쓰는 논문 라우터의 비교 계약

3개의 난이도 label을 출력하는 논문과 임의 N개 후보 모델 중 하나를 선택하는 논문은 다르다. 후자의 후보 pool을 동일한 3개로 제한하면 우리 서비스와 같은 **3-action 비교**가 가능하지만 자동으로 난이도 분류기가 되지는 않는다.

공식 구현의 3-candidate adapter를 준비해 RouterDC를 1순위, GraphRouter를 2순위로 둔다. 최소 학습 실행은 먼저 한 가지 공식 방법만 한다. 공개 checkpoint가 다른 model ID를 학습했다면 출력 이름만 바꾸지 말고 공통 train으로 다시 학습한다. RouteLLM의 원래 2-model 설정은 3-action adapter가 불가능할 때 별도 두 후보 실험으로 보존한다.

| 논문 방법 | 확인한 공식 구현 | 3-candidate 적용과 남은 작업 |
|---|---|---|
| RouterDC, NeurIPS 2024 | 질문 encoder와 model embedding의 dual contrastive learning. [공식 코드](https://github.com/shuhao02/RouterDC) | 원 RB loader는 11개지만 출력 수를 바꿀 수 있음. 3개에서는 positive/negative top-k 중복·tie 처리를 수정. encoder fine-tuning이 필요 |
| GraphRouter, ICLR 2025 | query/task/model graph, local MiniLM embedding. [공식 코드](https://github.com/ulab-uiuc/GraphRouter) | `llm_num`과 graph를 3개 pool로 구성. 공개 QA checkpoint의 기존 pool과 달라 공통 데이터 재학습 필요. graph 입력·계산량이 추가됨 |
| CARROT, 확장 후보 | 모델별 quality/cost 예측, local SBERT+kNN 변형. [공식 코드](https://github.com/somerstep/CARROT) | label voting kNN과 다른 score-regression. 기본 준비 코드의 OpenAI embedding 호출을 사용하지 않는 local-only adapter 필요 |

외부 framework의 이름이 test인 split을 그대로 믿지 않는다. 조사한 [LLMRouter RouterDC trainer](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/routerdc/trainer.py)는 평가 점수로 best checkpoint를 저장한다. 여기에 우리 validation만 연결하고 진짜 test는 최종 별도 evaluator에만 전달한다. GraphRouter의 새 LLM 결과 생성 스크립트도 실행하지 않고 공개 outcome으로 graph edge를 만든다. 원 source의 task/category label을 inference에서 요구하면 query-only 조건에서 제외하거나 학습된 task predictor를 포함한 변형으로 명시하며, test의 정답/분류 metadata를 공짜 입력으로 주지 않는다.

각 source의 train에서만 비용–품질 Pareto를 계산해 저비용·중간·상위 대표 세 모델을 정하고 봉인한다. 원천의 11/18 모델을 3개 그룹으로 뭉쳐 그룹 평균을 실제 1회 호출 결과인 것처럼 사용하지 않는다. 그룹 기반 분류와 대표 모델 기반 행동을 별도 필드로 기록한다. 유효한 세 Pareto 후보가 없으면 임의로 만들지 말고 3-tier 비교 부적합을 보고한다.

3-model 공식 구현을 적용할 때도 실제 데이터에서 model count 변경·API 의존·negative sampling·heldout indexing을 smoke로 확인하기 전에는 재현 완료라고 하지 않는다. inference는 질문 텍스트와 공개된 후보 metadata만 허용하고 해당 test 질문의 정답, 생성 응답, 실제 output token 수는 제공하지 않는다.

최소 공식 방법 비교는 하나의 source/pool(우선 xRouteBench generic) 안에서 모든 비교 방법을 동일 train으로 재학습한다. 다른 source는 별도 재현 또는 symbolic tier-transfer 실험으로 둔다. RouterBench와 xRouteBench의 서로 다른 실제 model IDs를 합쳐 “전역 모델 3개”라고 부르지 않는다. 기존 통합 R0/R1/R2 실험은 source-local 세 action으로 투영한 별도 계약이며, source별 결과와 투영의 한계를 보고한다.

## 8. 공통 데이터·평가와 실행 순서

1. 현재 v2 production checkpoint는 보존하고 historical baseline으로만 평가한다. train 오염이 없는 holdout을 확인한다.
2. 공통 benchmark 버전·모델 pool·train/validation/test·중복 group을 봉인한다. metadata와 checksum은 provenance용이며 모델 입력이 아니다.
3. R0/TF-IDF/kNN은 동일 difficulty target, R1/R2/공식 model-selector는 동일 raw outcome에서 각 기법 target을 만든다. 이것이 같은 증거에 기반한 비교다. target까지 모두 같게 강제하면 목적함수 비교가 사라진다.
4. source-local routing score는 source별로 평가하고 macro 평균을 추가한다. 다른 후보 pool의 지표를 같은 모델의 정답률처럼 해석하지 않는다.
5. 새 학습은 1,200행/40 step pilot으로 시간·메모리를 측정한다. 생성 API는 사용하지 않는다. 이미 생성된 benchmark judge 점수는 재사용 사실을 표기한다.
6. 경량 기준선→ModernBERT R0/R1/R2→공식 router 1개를 canonical seed 42로 수행한다. finalist 최대 2개만 seed 17/73 반복한다. RoBERTa 근접 규모 비교는 이 결과와 분리해 추가한다.
7. threshold·비용 penalty·checkpoint는 validation으로 동결한다. test curve의 각 점도 validation에서 정한 정책의 실제 test cost/quality로 그린다. test에서 예산을 다시 맞춰 최적점을 찾지 않는다.
8. RouterBench test / xRouteBench 공식 test가 주 outcome 평가다. RouterArena aligned/unseen/full은 동결 후 한 번의 외부 difficulty 진단이다. 추가 데이터는 LLMRouterBench의 공개 scored bundle 중 공통 후보·정답·비용과 overlap 검사를 통과한 경우에만 확장한다. label-only 데이터로 cost-quality를 만들어내지 않는다.
9. 같은 질문/duplicate group 단위 paired bootstrap CI를 보고한다. 유의하지 않은 차이를 우열로 발표하지 않는다. 초기학습/추론/embedding/index 비용도 별도 기록한다.

낮은 RouterArena 점수를 보고 반복 변경한 모델은 더 이상 그 데이터에 대한 완전한 untouched 평가라고 주장할 수 없다. 과거 관찰 이력은 공개하고 새 설계의 sealed holdout을 별도로 유지한다.

## 9. 발표 산출물

- 동일 예산의 품질, 동일 품질의 비용, always/random/oracle 대비 위치.
- 난이도 혼동행렬과 domain·길이별 오류: cost-quality와 별도 화면.
- 학습 목표만 변경한 controlled ablation, near-size backbone 비교: 서로 다른 결론.
- feedback 이후 unseen 성능과 기존 영역 retention, 실패·거부까지 포함한 비용.
- cache hit/false reuse/품질/절감과 반복율 변화.
- 각 결과의 관측 범위: 공개 모델 outcome replay / 라우터 실제 실행 / 가정 기반 simulation / Claude 실제 서비스 검증.

관련 기존 계획: [통합 데이터·backbone](integrated-routing-data-and-model-comparison-plan.md), [목적함수 비교](routing-objective-comparison-experiment-plan.md), [운영 지속학습](continual-retraining-strategy.md). 기존 실행 범위가 이 문서와 충돌하면 2026-09-20 최소 실험 범위를 우선하며 과거 artifact와 운영 설정은 보존한다.
