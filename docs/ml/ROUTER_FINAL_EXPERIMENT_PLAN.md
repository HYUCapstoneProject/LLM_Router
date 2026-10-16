# 라우터 최종 실험 계획서

실험 코드: **RT-08 · EV-10** (EV-06·07·08은 EV-10으로 통합, EV-09는 보류) · [실험 목록](../../experiments/REGISTRY.md)

작성일: 2026-10-16. **실행 전 계획**이며 운영 변경 승인이 아니다. 남은 라우터 실험의 기준 문서다. [라우터 평가 계획서](ROUTER_EVALUATION_V2_PLAN.md)(10-13)와 [논문 라우터 비교 계획서](ROUTER_COMPARISON_EXPERIMENT.md)(10-16 개정)의 결정을 이어받아 하나로 정리했고, 두 문서와 내용이 다르면 이 문서를 따른다.

## 0. 한눈에

| 항목 | 내용 |
|---|---|
| 답할 질문 | ① 같은 기준·같은 데이터에서 어떤 라우터가 좋은가? 우리 라우터는 논문 라우터와 비교해 어디쯤인가? ② 라우터가 "κ에 맞춘 고정 시작 단계"를 이기는 조건은 무엇인가? ③ 새 세대 모델의 데이터를 넣으면 나아지는가? ④ v1 채점 오류가 학습된 라우터를 나쁘게 만들었나? |
| 실험 | **RT-08**(④, 작게), **EV-10**(①~③, 본 실험). EV-09(실제 Claude 해결률)는 보류 |
| 단계 정의 | 원천마다 가성비 경계 근처 모델을 가격대 셋으로 묶은 **모델 집단**(2.1절) |
| 학습 데이터 | A 분할(RouterBench·xRouteBench, v2 채점) + LLMRouterBench·SPROUT의 train 50%. 원천별 비중을 같게 둔다 |
| 평가 데이터 | A 분할 봉인 test, LLMRouterBench·SPROUT test 35%(봉인), RouterArena(진단) |
| 비용 | 서비스 가격 비율 Haiku:Sonnet:Opus = **1:3:5**, 단계별 1회 호출 비용으로 고정. 길이 반영은 민감도 분석 |
| 평가 방식 | 두 갈래: ① 비용 고려(손익분기점 κ\*) ② 확실히 풀 수 있는 가장 약한 단계(τ = .8과 τ 곡선) |
| 자원 | 개인 Colab T4. 상한은 두지 않는다. EV-10 약 12~14시간, RT-08 약 1시간(추정). seed 반복은 판정 경계 근처일 때만 |
| 새 LLM 호출 | 없다(EV-09 제외). 기록된 결과만 재생한다 |

## 1. 출발점: 지금까지 확인한 것

| 사실 | 근거 |
|---|---|
| 학습한 라우터는 고정 단계 혼합선보다 +.02~.04 위에서 멈춘다. 다른 LLM 8개의 정답 여부를 알아도 +.052다 | EV-04 [라우팅 천장 분석](../../experiments/results/routing-ceiling-20261013/REPORT.md) |
| 상향을 허용하면 low부터 시작이 oracle 수준의 최종 해결률을 낸다(v2: RouterBench 93.5%, xRouteBench 88.3%). 라우터가 바꾸는 것은 첫 시도 성공과 비용의 교환이다 | EV-04 4절 |
| 어댑터를 거치면 R0와 R1의 효용이 같다. 상향을 고려한 규칙은 κ별 최선의 고정 시작 단계와 거의 같다 | EV-05 [통일 평가 시제품](../../experiments/results/unified-eval-prototype-20261013/README.md) |
| 재학습(F1~F5)은 어떤 방식도 초기 R0를 넘지 못했다 | RF-02~04, [재학습 계획서](RETRAINING_EFFECT_EXPERIMENT.md) 15.1절 |
| v1 채점 오류는 방법 순위를 바꾸지 않았다(Kendall τ .93~1.0). 라벨은 같은 경계에서 8.4%(GSM8K 35%), R1 학습 정답은 4.6~7.8% 바뀐다 | DT-02, XA-07 |
| 데이터의 단계 간 비용 차이(high가 low의 17~88배)는 서비스 Claude 가격(1:3:5)보다 훨씬 크다 | 2.3절 |
| 세대가 바뀌면 같은 문항의 난이도 순서가 꽤 달라진다(같은 MMLU 문항 2023 → 2025 일치도 .67) | XA-03 |
| RouterArena 공식 난이도는 모델 성능이 아니라 Bloom 인지 단계(DeepSeek-V3.1 판정)로 붙인 라벨이다 | [ML 명세](README.md) 4장 |

## 2. 확정한 설계

### 2.1. 단계 = 모델 집단

**왜:** 프로젝트의 원래 목표는 일반화된 상·중·하를 Claude Haiku·Sonnet·Opus에 잇는 것이다. 지금처럼 원천마다 대표 모델 하나를 한 단계로 쓰면 특정 모델에 묶이고, 그 모델의 우연한 정답·오답에 흔들린다.

**규칙:** 원천마다 다음 두 단계로 나눈다. 통계는 train(외부 데이터는 재분할 후 train)에서만 낸다.
1. 같은 가격 이하에서 가장 좋은 성능보다 .03 넘게 낮은 모델은 뺀다(가성비 경계 근처 모델만 남긴다).
2. 남은 모델을 로그 비용으로 세 묶음(1차원 k-means)으로 나눠 low·medium·high로 둔다.

아래는 A는 train, 외부 데이터는 기존 calibration 분할로 미리 계산한 예시다. 재분할 후 새 train으로 다시 계산해 00 노트북에서 고정한다.

| 원천 (남은 모델 수) | low | medium | high | 단계별 평균 성능 |
|---|---|---|---|---|
| RouterBench (11개 중 5) | Mistral-7B, WizardLM-13B | Mixtral-8x7B, Yi-34B | GPT-4 | .53 / .70 / .82 |
| xRouteBench (18개 중 8) | gpt-oss-20b, Gemma-2-9B | gpt-oss-120b, Mistral-Small-24B | Llama3-70B, Llama-3.3-70B, Qwen3-Next-80B, Cogito-671B | .68 / .71 / .73 |
| LLMRouterBench (13개 중 6) | DeepSeek-V3 2종, Qwen3-235B | Qwen3-235B-Thinking | GPT-5, Gemini 2.5 Pro | .41 / .54 / .55 |
| SPROUT 일반 대화 (13개 중 10) | Llama-3.2-3B, Granite 2B·8B | Llama 3.1 8B·70B, Llama-3.3-70B, Mixtral-8x7B, GPT-4o-mini | GPT-4o, Claude 3.5 Sonnet | .82 / .89 / .94 |

- 네 원천 모두 단계가 올라갈수록 성능이 오른다. 집단 크기는 1~5개다.
- 원천마다 성능 수준이 다른 것(예: LLMRouterBench high .55)은 그 원천의 문제가 어렵기 때문이다. 학습 목표는 집단의 평균 성적이 아니라 질문마다 "이 집단이 풀 확률"이다.
- **단계 성공 확률:** 질문 q에서 집단 g의 모델들이 푼 비율이다. 평가에서는 "집단에서 모델 하나를 무작위로 부른다"고 보고, 상향까지 포함해 정확히 계산한다.

### 2.2. 질문의 3단계 라벨

- **정의:** 풀 확률이 τ(.8) 이상인 가장 싼 집단이다. 어떤 집단도 τ에 못 미치면 풀 확률이 가장 높은 집단으로 둔다(동률이면 싼 쪽).
- 기존 R0 라벨(전체 모델 중 푼 비율, 경계 .27/.64)과 다르다. 기존 라벨은 싼 모델과 비싼 모델을 한데 섞어 셌다.
- 집단이 모델 1~2개로 작으면 τ = .8은 사실상 "집단 모델이 모두 푼다"는 뜻이 된다.

### 2.3. 비용

- **주 방식:** 서비스 가격 비율 Haiku:Sonnet:Opus = 1:3:5를 단계별 1회 호출 비용으로 고정한다(Opus 1회 = 1). 근거는 `backend/src/main/resources/application.yml`의 가격(2026-06: Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15, Opus 4.8 $5/$25, 100만 토큰당)이다. 입력·출력 가격이 모두 1:3:5라 단계 간 비율은 질문 길이와 관계없이 같다.
- **왜 고정인가:** Claude다운 부분은 1:3:5 비율이다. 길이를 반영하려면 Claude의 답변 길이를 데이터 속 모델의 답변 길이로 추정해야 하고, 그러면 원천마다 다른 모델 습관이 다시 비용에 섞인다.
- **민감도:** 비용 = (입력 토큰 + 5 × 출력 토큰 추정) × 단계 가격. 출력 토큰은 단계와 관계없이 그 질문에 대한 모든 모델 출력 길이의 중앙값으로 둔다. 결론이 유지되는지만 확인한다.
- 데이터의 원래 가격은 집단을 나눌 때(어떤 모델이 같은 가격대인가)만 쓴다. SPROUT은 [SPROUT 비용 추산](SPROUT_COST_ESTIMATION.md)의 데이터 저자 단가표를 쓴다.

### 2.4. 학습·평가 데이터

| 데이터 | 학습 (train) | 보정·설정 (dev) | 평가 (test, 봉인) |
|---|---|---|---|
| A 분할: RouterBench + xRouteBench (v2 채점) | 21,406 (RouterBench 제외 문항은 뺌) | D_dev 3,581 | 7,202 (v2 제외 28문항은 뺌) |
| LLMRouterBench (주 평가·같은 계열 8,970) | 50% | 15% | 35% |
| SPROUT (19,458) | 50% | 15% | 35% |
| RouterArena (8,379) | – | – | Bloom 라벨 일치 진단, 외부 데이터와 겹치는 700문항의 해결률 |

- **재분할:** LLMRouterBench·SPROUT의 기존 calibration 30% / test 70%는 test를 한 번도 열지 않았으므로 버리고 새로 나눈다. 중복 묶음·같은 질문 본문·같은 긴 지문 단위로 나눠, 한 묶음이 양쪽에 걸치지 않게 한다.
- **학습에서 빼는 것:** RouterArena와 겹치는 700문항(test에 둔다), A test와 중복되는 문항, LLMRouterBench의 판정 선호 진단(arenahard 740, test에만 진단용으로 둔다).
- **원천별 비중:** 학습 손실에서 원천마다 같은 총 가중치를 준다. 지금은 RouterBench가 A train의 88%다.
- **채점:** A 분할은 v2(정정)로 학습하고 평가한다. 외부 데이터는 원천 자체 채점(LLMRouterBench 1.0, SPROUT 판정 점수 ≥ .70)을 쓴다.
- 학습 행은 약 21k에서 35k로 늘어난다.

### 2.5. 두 갈래 평가

- **어댑터:** 모든 라우터의 점수를 원천(모델 풀)마다 그 원천의 dev에서 단계별 성공 확률로 보정한다(단계별 isotonic). 원천마다 실력 수준이 다른 문제를 여기서 흡수한다. 점수 하나만 내는 라우터도 같은 방식으로 세 단계 확률을 만든다.
- **갈래 ① 비용 고려:** 서비스 효용 U = 최종 해결 − λ × 총비용 − κ × 실패 횟수로 시작 단계를 고른다. 결정 규칙은 "한 번에 고르기"와 "상향을 고려한 시작 단계"(단계 간 실패 상관 ρ는 dev에서 추정) 두 가지다.
  - λ는 고르지 않고 같은 비용끼리 비교한다(비용 .25·.5·.75, Opus만 쓰면 1).
  - **손익분기점 κ\*:** 같은 비용에서 가장 좋은 고정 시작 정책(혼합 포함)을 +.01 넘게(95% 구간이 0을 포함하지 않음) 이기기 시작하는 가장 작은 κ. κ 격자는 0~1, .05 간격이다.
  - 결론은 "총비용 c에서 실패 1회의 무게가 κ\* 이상인 서비스라면 라우터가 이득이다"라는 형태로 쓴다.
- **갈래 ② 확실히 풀 수 있는 가장 약한 단계:** 풀 확률 ≥ τ인 가장 낮은 단계에서 시작하고, 실패하면 상향한다. τ = .8을 대표값으로 쓰고 τ .5~.95 곡선을 함께 보고한다. 보고 항목은 필요 단계 정확도(2.2절 라벨 대비), 첫 시도 성공, 최종 해결, 평균 호출 수, 총비용이다.
- **공통:** 원천별로 따로 보고한다. 차이는 질문 그룹 단위 paired bootstrap 95% 구간으로 낸다. A test는 v1 채점 값도 함께 적는다. 서비스의 실제 κ는 운영 로그의 "불만족 응답 뒤 상향하지 않고 떠난 비율"로 잰다.

## 3. 실험 1: RT-08 v1 채점 오류의 영향

| 항목 | 내용 |
|---|---|
| 왜 | 운영 R0는 채점 오류가 있는 v1으로 학습했다. 평가 순위는 v1·v2에서 같았지만, 학습된 라우터 자체가 나빠졌는지는 다시 학습해야 안다 |
| 무엇 | 기존 정의의 R0(경계 .27/.64)를 A 분할 v2로 한 번 다시 학습해 운영 R0(v1)와 비교한다. 새 정의(2절)와 섞지 않아 채점 정정 효과만 분리된다 |
| 어떻게 | 같은 A train 행(RouterBench 제외 문항만 뺌), ModernBERT-base, seed 42, 512토큰, 같은 step 예산. T4 약 1시간. 평가는 A test에서 대표 모델 기준과 집단 기준(2.5절) 둘 다 |
| 판정 | 비용 .5에서 κ 전 구간에 걸쳐 v2 학습본이 +.01 넘게 높으면 "v1 오류가 운영 R0를 실질적으로 나쁘게 했다"로 기록하고 운영 교체 검토의 근거로 쓴다. 어느 κ에서도 차이가 없으면 "실질적 영향 없음"으로 기록한다. 운영 교체와 배포는 별도 결정이다 |

## 4. 실험 2: EV-10 통합 라우터 비교

### 4.1. 비교 대상

| 구분 | 이름 | 무엇을 예측하나 | 코드 | 트랙 | 장비 |
|---|---|---|---|---|---|
| 우리 | **R0-집단** | 질문의 3단계 라벨(2.2절) 분류 | 우리 학습 코드 | 원형 (ModernBERT) | T4 |
| 우리 | **R1-집단** | 집단별 풀 확률(부드러운 BCE) | 우리 학습 코드 | 원형 (ModernBERT) | T4 |
| 우리 | R1-BGE | R1-집단과 같은 목표를 고정 BGE + MLP로 | 우리 학습 코드 | 공통 | CPU |
| 우리 | 운영 R0 (v1, 기존 정의) | 기존 3단계 난이도 | 재학습 없음, 참고 기준 | 원형 | – |
| 논문 | kNN 효용 라우터 (+MLP·SVM) | 비슷한 과거 질문의 집단별 평균 성공 | [RouterBench 공식](https://github.com/withmartian/routerbench) (MIT) | 공통 | CPU |
| 논문 | MF 라우터 | 질문 × 집단 궁합 점수(행렬 분해) | [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (MIT) | 공통 | CPU, GPU 권장 |
| 논문 | RouterDC | 질문 임베딩과 집단 임베딩의 유사도(대조 학습) | LLMRouter (MIT) | 원형 (mDeBERTa) | T4 |
| 논문 | GraphRouter | 질문–집단 그래프의 연결 강도 | [공식](https://github.com/ulab-uiuc/GraphRouter)·LLMRouter (MIT) | 공통 | T4 권장 |
| 논문 | Hybrid LLM | 작은 집단(low)으로 충분할 확률 | LLMRouter (MIT) | 공통 | CPU |
| 논문 | IRT-Router | 질문 난이도·변별도와 집단 능력으로 성공 확률 | [공식](https://github.com/Mercidaiha/IRT-Router) (라이선스 없음, 연구용) | 공통 | CPU |
| 논문 | RouteLLM-BERT 공개본 | 강한 모델이 필요한 확률. 학습 없이 그대로 | [공식 추론 코드](https://github.com/lm-sys/RouteLLM) (Apache-2.0) + 공개 체크포인트 | 공개본 | T4 수 분 |
| 논문 | RouteLLM-BERT 재학습본 | 같은 구조를 우리 데이터로 학습 | Hugging Face 학습 코드 | 원형 (XLM-RoBERTa) | T4 |
| 논문 | Avengers-Pro | 질문 묶음별로 성능 대비 비용이 가장 좋은 집단 | [공식](https://github.com/ZhangYiqun018/AvengersPro) (라이선스 없음, 연구용) | 공통 | CPU |
| 기준선 | 고정 시작 정책 3개와 혼합선, oracle | – | – | – | CPU |
| 비교용 | R1-집단-A, kNN-A | A 분할만으로 학습한 버전 | – | – | T4·CPU |

### 4.2. 각 라우터를 넣는 이유

- **kNN 효용 라우터:** 비슷한 문제를 누가 잘 풀었는지 그대로 따라 한다. 최근 연구에서 거의 항상 상위권인 강한 단순 기준선이다. 우리 서비스가 이미 계산하는 BGE 임베딩만으로 라우팅하면 어떤지도 답한다.
- **MF:** 추천 시스템처럼 질문과 모델의 궁합을 배운다. 라우팅 논문에서 가장 널리 쓰는 비교 기준이다.
- **RouterDC:** 잘 맞는 모델은 가깝게, 안 맞는 모델은 멀게 배우는 대조 학습이다(NeurIPS 2024).
- **GraphRouter:** 질문과 모델을 그래프로 잇는다(ICLR 2025). 우리 xRouteBench를 만든 그룹의 방법이다.
- **Hybrid LLM:** "작은 모델로 충분한가"를 예측하는 초기 난이도 기반 라우팅이다(ICLR 2024). R0와 발상이 가깝다.
- **IRT-Router:** 문항반응이론으로 질문 난이도와 모델 실력을 나눠 본다(ACL 2025). 난이도를 연속값으로 추정하므로 R0와 가장 가깝고, 추정한 난이도가 우리 라벨과 얼마나 맞는지도 볼 수 있다.
- **RouteLLM-BERT 공개본·재학습본:** 남이 학습한 완성품을 그대로 쓰는 경우와 같은 구조를 우리 데이터로 학습한 경우를 비교한다. "왜 직접 학습하나"에 답한다. RouteLLM의 강한 모델(GPT-4 1106)은 RouterBench의 high와 같은 모델이다.
- **Avengers-Pro:** 질문을 묶음(과제 유형에 가까운 단위)으로 나눠 묶음마다 집단을 정해 둔다. "과제 유형만으로 얼마나 되나"를 대표하는 최근 방법이다.

### 4.3. 공정한 비교 규칙

- **같은 데이터와 같은 단계:** 모든 라우터는 2.4절 train으로 학습하고 2.1절 집단을 단계로 쓴다. 논문 라우터는 원천마다 집단 셋을 "가상의 모델 3개"로 보고 학습한다(성능 = 집단 평균 성공).
- **정답 동점:** "질문마다 가장 잘 푼 후보"를 정답으로 배우는 구현(LLMRouter의 kNN·MLP·MF·GraphRouter)은 후보 순서를 low → medium → high로 고정해 동점이면 싼 단계가 정답이 되게 한다.
- **RouteLLM-BERT 재학습본의 정답:** 원본과 같은 세 분류(강한 모델 승 / 무승부 / 약한 모델 승)를 쓴다. 강한 모델 = high 집단, 약한 모델 = low 집단이다. 집단 평균으로 각 분류의 확률을 만든다(예: 강한 승 = high 풀 확률 × (1 − low 풀 확률)).
- **Hybrid LLM:** 작은 모델 = low 집단, 큰 모델 = high 집단.
- **모델 설명이 필요한 방법**(GraphRouter·IRT-Router)은 집단 설명(소속 모델 이름과 가격대)을 00 노트북에서 고정한다.
- **과제 이름:** 어떤 라우터도 추론 때 정답 과제 이름(MMLU, GSM8K 등)을 받지 않는다.
- **인코더 트랙:** "방법 차이"는 같은 고정 BGE-large-en-v1.5 임베딩(512토큰)을 쓰는 공통 트랙끼리만 주장한다. 원형 트랙(각 논문의 인코더를 fine-tuning)은 파라미터 수와 학습 시간을 함께 적는다. 라이브러리 기본 임베딩(OpenAI 임베딩 등)은 BGE로 바꾸며 OpenAI는 호출하지 않는다.
- **조정 예산:** 학습형 방법마다 dev에서 설정을 최대 8개만 시도한다. 선택 기준은 dev에서 비용 .25·.5·.75와 κ 격자에 걸친 평균 효용(원천별로 구해 평균)이다.
- **반복:** seed 42로 한 번 실행한다. 결과가 판정 경계(±.01) 근처일 때만 seed 17·73을 추가한다.
- **누출 방지:** test outcome은 test 노트북 전까지 SHA만 기록한다. 우리 기존 라우터의 A test 결과는 이미 여러 번 관찰됐으므로 보고서에 그 이력을 적고, 확인 실험의 역할은 처음 여는 외부 test가 맡는다. 외부 코드가 `test`라는 분할로 checkpoint를 고르면 우리 dev를 연결한다.
- **입력 길이:** 모든 인코더는 512토큰으로 비교하고 잘린 비율을 보고한다. 운영 R0의 8,192토큰 추론과는 조건이 다르다.
- **외부 코드:** 우리 저장소에 복사하지 않는다. Colab에서 커밋을 고정해 내려받고 커밋 SHA를 manifest에 남긴다. 라이선스가 없는 코드와 체크포인트(IRT-Router, Avengers-Pro, RouteLLM 공개 체크포인트)는 연구용으로만 실행하고 서비스에 넣지 않으며, 원 논문을 인용한다.

### 4.4. 가설과 사전 판정

| 가설 | 내용 | 판정 방법 |
|---|---|---|
| H1 정체 | 학습형 라우터들의 차이는 대부분 작고 kNN이 상위권이다 | 평균 효용 1위와 각 방법의 차이 구간이 0을 포함하는지, kNN 순위 |
| H2 분류형 대 확률형 | 갈래 ②에서 R0-집단(분류)은 R1-집단(확률)과 비슷하다 | 필요 단계 정확도와 첫 시도 성공의 차이가 ±.02 안인지 |
| H3 해석 | IRT-Router의 난이도 값은 우리 3단계 라벨과 양의 상관을 보이고, RouterArena의 Bloom 라벨과는 상관이 약하다(다른 개념) | 원천별 Spearman 상관과 구간 |
| H4 직접 학습 | RouteLLM-BERT 재학습본이 공개본보다 같은 비용에서 효용이 높다 | 비용 .5에서 κ 전 구간의 효용 차이가 +.01을 넘고 구간이 0을 포함하지 않는지 |
| H5 새 세대 데이터 | 섞어 학습한 R1-집단·kNN이 A만 학습한 버전보다 LLMRouterBench·SPROUT test에서 낫고, A test에서는 크게 나빠지지 않는다 | 외부 test에서 +.01 초과, A test에서 −.01 이내 |
| H6 채점 강건성 | A test의 방법 순위는 v1·v2 채점에서 크게 바뀌지 않는다 | Kendall τ ≥ .7 |
| H7 라우터의 가치 | 서비스 가격(1:3:5)에서는 대부분의 라우터가 작은 κ에서 고정 시작 단계를 이기지 못한다 | 라우터별 손익분기점 κ\*(비용 .25·.5·.75) |

어떤 결과도 운영 라우터를 자동으로 바꾸지 않는다. 운영 변경은 별도 결정이고 배포는 팀원이 맡는다.

### 4.5. 실행 구조

위치는 `ml/notebooks/router_final/`이다. 각 노트북은 앞 단계 산출물의 SHA를 확인한 뒤 시작한다.

| 노트북 | 하는 일 | 장비 | 예상 시간 |
|---|---|---|---|
| `00_prepare` | 외부 데이터 재분할, 원천별 집단 고정, 3단계 라벨과 집단 성공 확률, 서비스 비용, 형식 변환(RouterBench·LLMRouter·IRT-Router·Avengers-Pro), 외부 코드 커밋 고정, 모든 입력 SHA 봉인. test outcome은 SHA만 기록 | CPU | 30분 |
| `01_embed` | 모든 문항(약 7만)의 BGE 임베딩을 한 번 계산해 봉인 | T4 | 45분 |
| `02_rt08` | 기존 정의 R0를 A 분할 v2로 재학습 (RT-08) | T4 | 1시간 |
| `03_ours` | R0-집단, R1-집단(섞은 데이터), R1-집단-A(A만) 학습 | T4 | 4.5시간 |
| `04_common_track` | kNN·MLP·SVM(+kNN-A), MF, GraphRouter, Hybrid LLM, IRT-Router, Avengers-Pro, R1-BGE 학습과 dev 설정 선택 | CPU, MF·GraphRouter는 T4 | 1~2시간 |
| `05_finetune_track` | RouterDC, RouteLLM-BERT 재학습 | T4 | 3~4.5시간 |
| `06_released` | RouteLLM-BERT 공개본으로 dev·test 점수 계산 | T4 | 15분 |
| `07_lock` | 원천별 어댑터 보정, ρ, 결정 규칙, 설정을 `selection-lock.json`으로 고정한다. **여기서 멈추고 사용자에게 lock 내용을 보고한다** | CPU | 15분 |
| `08_test` | 원천 4개의 test를 한 번씩 열어 두 갈래 평가(A test는 v1 병기), CPU 라우팅 시간 측정 | T4, CPU | 1.5시간 |
| `09_seeds` | 판정 경계 근처 결과에만 seed 17·73 추가(잠근 설정 그대로) | T4 | 필요할 때 |
| `10_report` | 표·곡선·bootstrap·보고서 초안 | CPU | 30분 |

- 자원 상한은 두지 않는다. 각 노트북이 끝날 때 사용한 CU를 기록해 보고한다.
- 외부 코드가 우리 형식이나 집단 구성에서 동작하지 않으면 억지로 맞추지 않고 "재현 실패/부분 차용"으로 기록해 결과표에서 구분한다.

### 4.6. 산출물과 보관

- 원본 결과는 Google Drive `LLM_Router/router-final/runs/run001/`에 둔다.
- 저장소에는 요약·그림·manifest·보고서만 `experiments/results/ev10-router-final-<날짜>/`로 넣는다. 큰 parquet·임베딩·checkpoint·외부 코드는 넣지 않는다.
- 필수 기록: 데이터·config·모델 revision hash, 외부 저장소 커밋 SHA, 집단 구성, seed, 시도한 설정 전부, 학습 시간, 파라미터 수, `selection-lock.json`, test 예측.

## 5. EV-09 RouterArena 실측 (보류)

실제 서비스 모델(Claude)의 정답 기록을 만드는 유일한 실험이다. 층화 표본 약 1,000문항을 서비스 세 단계로 실제 호출하고 RouterArena 공식 규칙으로 채점한다. API 비용(수십 달러 추정, 현재 가격 확인 필요)과 새 API 호출 승인이 필요하다. 팀원과 비용을 협의한 뒤, EV-10 결과를 보고 다시 판단한다.

## 6. 하지 않는 것과 이유

| 하지 않는 것 | 이유 |
|---|---|
| CARROT | 비용을 단계별 고정 비율로 두면 CARROT의 핵심인 질문별 비용 예측이 쓸모없어지고, 남는 정확도 예측은 R1-집단과 같다 |
| EquiRouter | 논문이 적은 코드 저장소가 공개돼 있지 않다(10-16 확인) |
| EmbedLLM | 후보가 집단 3개면 많은 모델의 표현을 배우는 이점이 적어 MF로 대표한다 |
| BEST-Route | 한 질문에 모델별 응답이 여러 개 필요한데 기록에는 하나뿐이다 |
| LLMRouter CausalLM 라우터 | GPU 40GB 이상과 Llama-2-7B가 필요해 T4에서 돌릴 수 없다 |
| Router-R1, AutoMix, 다중 라운드 라우터 | 실시간 LLM 호출이 필요하다 |
| RouteLLM MF·SW 공개본 | 질문 임베딩에 OpenAI API가 필요하다 |
| R2, R0-τ, R1-tier | R2는 시제품에서 R1과 차이가 작다. R0-τ는 갈래 ②가 대신한다. R1-tier는 R1-집단이 된다 |
| R3 | G1 분할로만 학습돼 섞을 수 없다 |
| F 계열의 v2 재실행 | 재학습이 초기 R0를 못 넘은 원인이 라벨 비율과 텍스트 정보의 한계로 설명됐다 |
| 512/8192 재비교 | 결론에 영향이 없었다(RT-07) |
| RouterArena 전체 실측 | 비용 대비 정보가 적다. 필요하면 EV-09 표본으로 한다 |

## 7. 결정 기록

| 날짜 | 결정 |
|---|---|
| 2026-10-13 | 논문 라우터 비교는 축소하지 않고 한다. SPROUT 비용은 별도 문서로 정한다. R2·R3 점수는 미리 받지 않고 필요할 때 Colab에서 읽는다. EV-09는 팀원과 비용을 협의할 때까지 보류한다 |
| 2026-10-16 | 판정은 손익분기점 방식으로 한다. 논문 라우터는 kNN 효용 라우터(+MLP·SVM), MF, RouterDC, GraphRouter, Hybrid LLM, IRT-Router(연구용), RouteLLM-BERT(공개본 + 재학습본), Avengers-Pro로 하고 CARROT은 뺀다 |
| 2026-10-16 | 단계는 모델 집단으로 정의한다(2.1). LLMRouterBench·SPROUT를 학습에 넣고 50/15/35로 다시 나눈다(2.4). 비용은 서비스 가격 1:3:5 고정을 주로 하고 길이 반영은 민감도로 둔다(2.3). 평가는 두 갈래로 한다(2.5). 원천별 비중을 같게 둔다 |
| 2026-10-16 | RT-08은 기존 정의 R0만 A 분할 v2로 한 번 학습한다. EV-06·07·08은 EV-10으로 통합한다. A만 학습한 비교용은 R1-집단과 kNN이다. 자원 상한은 두지 않고 seed 반복은 판정 경계 근처일 때만 한다. 계획서는 이 문서 하나로 정리한다 |

## 8. 참고 문헌

- RouterBench: [Hu et al., 2024](https://arxiv.org/abs/2403.12031), [코드](https://github.com/withmartian/routerbench)
- RouteLLM: [Ong et al., ICLR 2025](https://arxiv.org/abs/2406.18665), [코드](https://github.com/lm-sys/RouteLLM)
- Hybrid LLM: [Ding et al., ICLR 2024](https://arxiv.org/abs/2404.14618)
- RouterDC: [Chen et al., NeurIPS 2024](https://arxiv.org/abs/2409.19886)
- GraphRouter: [Feng et al., ICLR 2025](https://arxiv.org/abs/2410.03834), [코드](https://github.com/ulab-uiuc/GraphRouter)
- IRT-Router: [Song et al., ACL 2025](https://arxiv.org/abs/2506.01048), [코드](https://github.com/Mercidaiha/IRT-Router)
- Avengers-Pro: [Zhang et al., 2025](https://arxiv.org/abs/2508.12631), [코드](https://github.com/ZhangYiqun018/AvengersPro)
- LLMRouter 라이브러리: [arXiv 2608.06867](https://arxiv.org/abs/2608.06867), [코드](https://github.com/ulab-uiuc/LLMRouter)
- CARROT·SPROUT: [Somerstep et al., 2025](https://arxiv.org/abs/2502.03261)
- LLMRouterBench: [Li et al., 2026](https://arxiv.org/abs/2601.07206)
- RouterArena: [Lu et al., ICLR 2026](https://arxiv.org/abs/2510.00202)
- Routing Plateau: [Lu et al., 2026](https://arxiv.org/abs/2606.07587)
- Routing Collapse·EquiRouter: [Lai & Ye, 2026](https://arxiv.org/abs/2602.03478)
- Cascade routing: [Dekoninck et al., ICML 2025](https://proceedings.mlr.press/v267/dekoninck25a.html)
- 과제 유형과 라우팅 격차: [arXiv 2608.23023](https://arxiv.org/abs/2608.23023)
