# 논문 라우터와 우리 라우터의 비교 실험 계획서

실험 코드: **EV-06** · [실험 목록](../../experiments/REGISTRY.md)

작성일: 2026-09-30. 개정: 2026-10-09, **2026-10-16(비교 대상·평가 기준 확정)**. **실행 전 설계 문서**이며, 노트북은 10절의 남은 결정을 확인한 뒤 만든다. 운영 변경 승인이 아니다. 평가 기준은 [라우터 평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 2~3절의 통일 기준을 그대로 쓴다.

## 0. 한눈에

| 항목 | 내용 |
|---|---|
| 질문 | 같은 데이터·같은 후보·같은 평가 기준에서 우리 라우터(R0~R2)는 논문 라우터들과 비교해 어디쯤인가? 어떤 라우터든 "κ에 맞춘 고정 시작 단계"를 이길 수 있는가? |
| 논문 라우터 (10-16 확정) | kNN 효용 라우터(+MLP·SVM), MF, RouterDC, GraphRouter, Hybrid LLM, IRT-Router, RouteLLM-BERT(공개 체크포인트 그대로 + 우리 데이터로 학습) |
| 추가 후보 (결정 대기) | CARROT, Avengers-Pro (10절) |
| 우리 라우터 | R0(어댑터), R0-τ, R1, R2, 공통 트랙 변형 R1-BGE·R1-tier |
| 기준선 | 고정 시작 정책(low·medium·high부터 시작, 실패하면 상향)과 그 혼합선, oracle |
| 평가 | 통일 기준. 모든 라우터의 출력을 단계별 성공 확률로 바꾼 뒤 같은 결정 규칙을 쓰고, 같은 비용끼리 비교해 손익분기점 κ\*를 보고한다 |
| 데이터 | 통합 A 분할(train 21,406 / D_dev 3,581 / 봉인 test 7,202). 학습 데이터 버전(v1·v2)은 E1 판정을 따른다. 채점은 v2 기준이고 v1을 함께 적는다 |
| 새 LLM 호출 | 없다. 기록된 결과만 재생한다 |
| 실행 환경 | 개인 Colab T4. 외부 코드는 커밋을 고정해 실행할 때 내려받는다 |
| 예상 자원 | T4 약 4~7시간(8~14 CU 추정), 상한 20 CU |

## 1. 왜 이 실험인가

### 1.1. 문헌에서 확인한 사실

1. **라우팅 정체(plateau).** 21개 라우터를 5개 벤치마크에서 비교한 연구에서 상위권 차이는 작았고, 단순한 kNN이 모든 벤치마크에서 상위 2위 안이었다([Routing Plateau](https://arxiv.org/abs/2606.07587)). LLMRouterBench도 대부분의 방법이 비슷한 성능으로 모이고 단순 기준선을 안정적으로 이기지 못한다고 보고했다([LLMRouterBench](https://arxiv.org/abs/2601.07206)). 강한 단순 기준선을 반드시 넣어야 한다.
2. **라우팅 쏠림(collapse).** 예산이 커지면 싼 모델로 충분한 질문까지 가장 비싼 모델로 보내는 현상이 보고됐다([When Routing Collapses](https://arxiv.org/abs/2602.03478)). 우리 재학습 실험에서도 예측이 한쪽으로 쏠렸다(RF-01~04). 쏠림을 따로 잰다.
3. **과제 유형.** 라우팅 격차의 대부분이 과제 유형으로 설명된다는 보고가 있다([Most of the LLM Routing Gap Is Task Type](https://arxiv.org/abs/2608.23023)). 질문 묶음별로 모델을 고르는 방법(Avengers-Pro)이 이 관점을 대표한다.

이 실험의 목적은 "우리가 최고"를 보이는 것이 아니다. 같은 조건에서 **어디쯤인지, 무엇이 다른지**를 정직하게 보고하는 것이다.

### 1.2. 우리 라우터의 학술적 위치

대부분의 논문 라우터는 "후보마다 맞힐 확률(또는 점수)을 예측해 가장 좋은 후보를 고른다". 우리 R1·R2도 이 주류에 속한다. R1은 후보별 성공 확률을, R2는 후보 간 상대 우열을 예측한다.

문헌에서 드문 것은 운영 라우터 R0다. R0는 질문 난이도를 easy/medium/hard로 분류해 단계에 연결한다. 난이도를 명시적으로 다루는 IRT-Router와, 질문 난이도로 작은 모델과 큰 모델을 고르는 Hybrid LLM이 R0와 가장 가까운 비교 대상이다.

### 1.3. 이번 개정에서 반영한 사실 (10-09 → 10-16)

| 사실 | 계획에 준 영향 |
|---|---|
| 평가 기준을 통일했다. 어떤 라우터든 D_dev에서 단계별 성공 확률로 보정(어댑터)한 뒤 같은 결정 규칙을 쓴다. 시제품에서 어댑터를 거친 R0와 R1의 효용이 같았다([평가 계획서 3절](ROUTER_EVALUATION_V2_PLAN.md)) | 주 지표를 nAUC에서 통일 기준과 손익분기점으로 바꿨다. R0도 어댑터로 비용 조절이 가능해진다 |
| 학습한 라우터는 고정 단계 혼합선보다 +.02~.04 위에서 멈춘다. 상향을 허용하면 low부터 시작이 oracle 수준의 최종 해결률을 낸다([천장 분석](../../experiments/results/routing-ceiling-20261013/REPORT.md)) | 고정 시작 정책을 기준선으로 둔다 |
| v1 채점 오류는 방법 간 순위를 바꾸지 않았다(Kendall τ .93~1.0). v2 전환 여부는 E1이 정한다 | 학습 데이터 버전은 E1 판정을 따른다(5.1절) |
| 비교 대상을 세 조건으로 다시 골랐다: Colab에서 돌 것, 소스코드를 쓸 수 있을 것, 우리 데이터로 학습할 수 있을 것(10-16) | 3·4절 |
| 우리 두 원천 모두 원 제작자가 공개한 라우터 코드가 있다. RouterBench는 공식 코드에 kNN·MLP·SVM 라우터가 있다. xRouteBench는 [LLMRouter](https://github.com/ulab-uiuc/LLMRouter)의 `llmrouter_generic` 데이터와 같고 test 3,729문항도 같다 | 같은 데이터로 다시 학습하기 쉽다 |
| 10-09 개정본의 CARROT·EquiRouter는 "코드 없음"으로 뺐다가, CARROT 저장소(somerstep/CARROT)를 확인해 추가 후보로 되돌렸다. EquiRouter는 논문이 적은 저장소가 공개돼 있지 않다 | 3·4·10절 |

## 2. 우리 라우터

### 2.1. 기존 checkpoint 재사용

| 이름 | 무엇을 예측하나 | 이번 평가에서의 선택 | 점수 위치 |
|---|---|---|---|
| R0 (운영) | 질문 난이도 easy/medium/hard (3-class CE) | 어댑터로 단계별 성공 확률을 만든 뒤 통일 결정 규칙. 원래 argmax 연결은 한 점으로 함께 보고 | 로컬 보관 |
| R1 | 후보별 성공 확률 (BCE) | 어댑터 후 통일 결정 규칙 | 로컬 보관 |
| R2 | 후보 간 상대 우열 (pairwise) | 어댑터 후 통일 결정 규칙 | GCS·Drive. 미리 받지 않고 Colab에서 바로 읽는다([평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 9절 결정 4) |

세 모델은 같은 ModernBERT-base, 같은 train 21,406문항, 512토큰으로 학습돼 있다. E1이 v2 전환을 판정하면 R0는 E1의 v2 학습본을 쓰고, R1·R2도 v2로 다시 학습한다(5.1절).

R0의 학습 신호는 원천의 **전체 후보**(RouterBench 11개, xRouteBench 18개)가 푼 비율이고, 평가는 대표 3개 모델로 한다. 학습 대상과 평가 대상이 어긋난다는 점을 결과 해석에 함께 적는다.

R3는 넣지 않는다. G1 분할(train 18,591문항)로만 학습돼 A 분할과 섞을 수 없다. 필요하면 G1 분할 안에서 R1과 따로 비교한다([평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 3.4절).

### 2.2. R0-τ

R0 확률로 "이 질문이 t단계 이하로 충분할 확률 P(난이도 ≤ t)가 τ 이상인 가장 낮은 단계"를 고르는 비교용 규칙이다. 예를 들어 softmax가 easy .50 / medium .30 / hard .20이면 τ=.45에서 low, τ=.70에서 medium, τ=.90에서 high를 고른다. τ grid는 0~1 사이 201점으로 고정한다. 어댑터가 R0에 비용 조절을 주므로 주 결과는 어댑터 쪽이고, R0-τ는 어댑터 없이 얼마나 되는지 보는 비교로 둔다. 운영 R0의 동작이 아니다.

### 2.3. 공통 트랙 변형: R1-BGE와 R1-tier

R1(ModernBERT fine-tuning)과 논문 라우터(고정 BGE 임베딩)를 바로 비교하면 "방법" 차이와 "인코더" 차이가 섞인다. 그래서 같은 BGE 임베딩 위에 작은 예측기(MLP)를 얹은 변형 두 개를 공통 트랙에 넣는다. CPU로 학습한다.

| 변형 | 학습 대상 | 답하는 질문 |
|---|---|---|
| **R1-BGE** | 대표 3개 모델 각각의 성공 여부 (R1과 같은 목표) | R1-BGE vs R1: 같은 방법에서 인코더 fine-tuning의 효과. R1-BGE vs 공통 트랙 논문 라우터: 같은 인코더에서 방법의 차이 |
| **R1-tier** | 전체 후보를 train 평균 비용으로 세 그룹으로 나눈 뒤, 그룹별 평균 성공률(0~1) | 특정 대표 모델 3개 대신 "싼 급/중간 급/비싼 급 모델이 얼마나 푸는가"를 배우면 나아지는가 |

- R1-tier의 그룹은 00 노트북에서 train 평균 비용 기준으로 3등분해 봉인한다. 대표 모델을 고른 Pareto 규칙과 일부러 다르게 둔다.
- R1-tier도 주 평가는 대표 3개 모델의 실제 결과로 한다. 실제 서비스는 그룹이 아니라 모델 하나를 호출하기 때문이다.

## 3. 비교 대상

### 3.1. 요약 표

| # | 라우터 | 논문 | 무엇을 예측하나 | 코드 | 트랙 | 장비 |
|---|---|---|---|---|---|---|
| 1 | kNN 효용 라우터 (+MLP·SVM) | [RouterBench, 2024](https://arxiv.org/abs/2403.12031) | 비슷한 과거 질문(기본 50개)에서 모델별 평균 성공률 | [RouterBench 공식](https://github.com/withmartian/routerbench) (MIT) | 공통 | CPU |
| 2 | MF 라우터 | [RouteLLM, ICLR 2025](https://arxiv.org/abs/2406.18665) 계열 | 질문 × 모델 궁합 점수 (행렬 분해) | [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) `mfrouter` (MIT) | 공통 | CPU, GPU 권장 |
| 3 | RouterDC | [Chen et al., NeurIPS 2024](https://arxiv.org/abs/2409.19886) | 질문 임베딩과 모델 임베딩의 유사도 (대조 학습) | LLMRouter `routerdc` (MIT) | 원형 (mDeBERTa) | T4 |
| 4 | GraphRouter | [Feng et al., ICLR 2025](https://arxiv.org/abs/2410.03834) | 질문–모델 그래프에서 연결 강도 | [공식](https://github.com/ulab-uiuc/GraphRouter)·LLMRouter (MIT) | 공통 | T4 권장 |
| 5 | Hybrid LLM | [Ding et al., ICLR 2024](https://arxiv.org/abs/2404.14618) | 작은 모델로 충분할 확률 (질문 난이도) | LLMRouter `hybrid_llm` (MIT) | 공통 | CPU |
| 6 | IRT-Router | [Song et al., ACL 2025](https://arxiv.org/abs/2506.01048) | 질문 난이도·변별도와 모델 능력으로 성공 확률 | [공식](https://github.com/Mercidaiha/IRT-Router) (라이선스 없음, 연구용) | 공통 | CPU |
| 7a | RouteLLM-BERT 공개본 | [RouteLLM, ICLR 2025](https://arxiv.org/abs/2406.18665) | 강한 모델(GPT-4)이 필요한 확률 | [공식 추론 코드](https://github.com/lm-sys/RouteLLM) (Apache-2.0) + 공개 체크포인트 | 학습 없음 | T4 수 분 |
| 7b | RouteLLM-BERT 재학습 | 같음 | 같음 | 같은 구조를 우리 데이터로 학습 (Hugging Face 학습 코드) | 원형 (XLM-RoBERTa) | T4 |
| 추가 후보 | CARROT | [Somerstep et al., 2025](https://arxiv.org/abs/2502.03261) | 모델별 정확도와 비용을 따로 예측 | [공식](https://github.com/somerstep/CARROT) (라이선스 없음, 연구용) | 공통(KNN), 원형(RoBERTa) | CPU, T4 |
| 추가 후보 | Avengers-Pro | [Zhang et al., 2025](https://arxiv.org/abs/2508.12631) | 질문 묶음별로 성능 대비 비용이 가장 좋은 모델 | [공식](https://github.com/ZhangYiqun018/AvengersPro) (라이선스 없음, 연구용) | 공통 | CPU |

### 3.2. 각 라우터를 비교할 만한 이유 (쉬운 설명)

**① kNN 효용 라우터: "비슷한 문제를 누가 잘 풀었나"를 그대로 따라 한다.**
새 질문과 가장 비슷한 과거 문제 k개를 찾아, 각 모델이 맞힌 비율의 평균을 낸다. 학습이 거의 없다. 최근 연구에서 거의 항상 상위권인 기준선이다. 우리 라우터가 이것보다 못하면 복잡한 학습의 의미가 없고, 비슷하면 "정체 구간 안"이라는 결론이 된다. 같은 RouterBench 공식 코드의 MLP·SVM 예측 라우터도 함께 돌린다. 우리 서비스는 캐시용 BGE 임베딩을 이미 계산하므로, 이 임베딩만으로 라우팅하면 어떤지에도 답한다.

**② MF 라우터: 추천 시스템처럼 "질문과 모델의 궁합"을 배운다.**
"사용자 취향 × 영화 특징"으로 추천 점수를 내듯, 질문 벡터와 모델 벡터의 조합으로 점수를 낸다. 라우팅 논문에서 가장 널리 쓰이는 비교 기준이다. RouteLLM 원본은 두 모델 전용이고 OpenAI 임베딩을 쓰므로, 여러 모델을 다루는 LLMRouter 구현을 쓴다.

**③ RouterDC: 잘 맞는 모델은 가깝게, 안 맞는 모델은 멀게 배운다.**
질문 임베딩과 모델 임베딩 사이 거리를 대조 학습으로 조정한다. NeurIPS 2024 논문이고 ML 명세에서 첫 공식 비교군으로 정했던 방법이다. 원본 저장소는 라이선스가 없어 LLMRouter의 MIT 구현을 쓴다.

**④ GraphRouter: 질문과 모델을 점, "풀었다"를 선으로 잇는 그래프를 만든다.**
새 질문이 어느 모델과 강하게 이어질지 그래프 신경망으로 예측한다. ICLR 2025 논문이고 우리 xRouteBench를 만든 그룹의 방법이다.

**⑤ Hybrid LLM: "작은 모델로 충분한가"만 예측한다.**
논문은 이를 질문 난이도 기반 라우팅이라고 부른다. 난이도 개념을 쓰는 초기 대표 논문(ICLR 2024)이라 R0와 비교할 의미가 있다. 두 모델 전용이므로 작은 모델 = low, 큰 모델 = high로 학습한다. 점수 하나를 어댑터로 세 단계 확률로 바꿔 세 단계 결과에도 넣는다. 공식 저장소는 사라져 LLMRouter 구현을 쓴다.

**⑥ IRT-Router: 수능 문항 분석처럼 "문제 난이도"와 "모델 실력"을 나눠서 본다.**
문항반응이론(IRT)은 "문제가 얼마나 어렵고 실력 차이를 얼마나 잘 가르는가"와 "응시자의 실력"으로 정답 확률을 설명한다. IRT-Router는 질문 임베딩으로 난이도·변별도를 계산하고 모델마다 실력 값을 배운다. 질문 난이도를 연속값으로 추정하므로 R0와 발상이 가장 가깝고, **IRT가 추정한 난이도와 우리 난이도 라벨이 얼마나 일치하는지**도 볼 수 있다. 코드에 라이선스가 없어 연구용으로만 쓴다(5.9절, 10절 확정 사항).

**⑦ RouteLLM-BERT: 남이 학습한 완성품과, 같은 구조를 우리 데이터로 학습한 것을 둘 다 쓴다.**
RouteLLM의 BERT 계열 분류기(XLM-RoBERTa, 약 3억 파라미터)는 질문을 보고 "강한 모델이 이길 / 비길 / 약한 모델이 이길" 확률을 내고, 그중 "강한 모델이 필요한 확률"으로 라우팅한다. 공개본은 챗봇 아레나 선호 데이터(일부 GPT-4 판정)로 GPT-4(1106)와 Mixtral-8x7B 사이를 고르도록 학습됐다.
- **7a 공개본:** 다시 학습하지 않고 그대로 쓴다. "남이 만든 라우터를 그냥 가져다 쓰면 되지 않나?"에 답한다. RouteLLM의 강한 모델이 우리 RouterBench high 단계와 같은 GPT-4(1106)라 RouterBench 쪽 해석이 자연스럽다.
- **7b 재학습본:** 같은 구조를 우리 데이터로만 학습한다. 7a와 7b의 차이가 곧 "우리 데이터로 직접 학습한 효과"다(H7).
- RouteLLM은 BERT 분류기의 학습 코드를 공개하지 않았다. 그래서 7b는 Hugging Face의 표준 분류 학습 코드로 학습하고, 추론은 RouteLLM 코드를 그대로 쓴다. MF·SW 공개본은 질문 임베딩에 OpenAI API가 필요해 쓰지 않는다.

**추가 후보 ⑧ CARROT: "맞힐 확률"과 "비용"을 따로 예측해 조합한다.**
후보마다 정확도와 비용을 각각 예측한 뒤 둘의 가중합으로 고른다. 이 단순한 방식이 이론적으로 최적 속도를 낸다는 증명(minimax)이 있다. 우리 R1과 구조가 거의 같고, 차이는 비용을 질문마다 예측한다는 점이다. 그래서 "질문별 비용 예측이 도움이 되는가"(H6)를 볼 수 있다. SPROUT 데이터를 만든 저자들이다. 데이터 준비 코드가 OpenAI 임베딩을 쓰므로 BGE로 바꾼다. 코드에 라이선스가 없다.

**추가 후보 ⑨ Avengers-Pro: 비슷한 질문끼리 묶고, 묶음마다 모델을 미리 정한다.**
질문 임베딩을 k-means로 묶고, 묶음마다 "성능 대비 비용이 가장 좋은 모델"을 정해 둔다. 질문 하나하나가 아니라 묶음(과제 유형에 가까운 단위)으로 고르므로, "과제 유형만으로 얼마나 되나"를 대표하는 최근(2025) 방법이다. 임베딩 서비스 주소를 받도록 돼 있어 미리 계산한 BGE 임베딩을 넣는다. 코드에 라이선스가 없다.

### 3.3. 기준선 (학습 없음)

- **고정 시작 정책 3개:** 질문과 상관없이 low·medium·high 중 한 곳에서 시작하고, 실패하면 다음 단계로 올린다. 세 정책을 질문마다 무작위로 섞어 쓰는 **혼합선**이 같은 비용의 비교 기준이다.
- **oracle:** 처음부터 풀 수 있는 가장 싼 단계로 보낸다. 셋 다 못 풀면 low로 본다.
- 10-09 개정본의 "완벽 검증 cascade"는 상향을 포함한 "low부터 시작"과 같으므로 따로 두지 않는다.

## 4. 제외한 라우터와 이유

| 라우터 | 제외 이유 |
|---|---|
| [EquiRouter](https://arxiv.org/abs/2602.03478) | 논문이 적은 코드 저장소가 공개돼 있지 않다(2026-10-16 확인). 쏠림 비교(H4)는 R2(상대 우열 학습)로 대신한다 |
| [EmbedLLM](https://arxiv.org/abs/2410.02223) | 수많은 모델의 표현을 배우는 방법이라 후보가 3개면 이점이 적다. 같은 계열인 MF로 대표한다 |
| BEST-Route | 한 질문에 모델별 응답을 여러 개 뽑아야 하는데 기록에는 하나뿐이다 |
| LLMRouter CausalLM 라우터 | GPU 40GB 이상과 Llama-2-7B가 필요해 T4에서 돌릴 수 없다 |
| Router-R1, [AutoMix](https://arxiv.org/abs/2310.12963), 다중 라운드 라우터 | 실시간 LLM 호출이 필요해 기록 재생으로 평가할 수 없다 |
| RouteLLM MF·SW 공개본 | 질문 임베딩에 OpenAI API가 필요하다 |
| [MixLLM](https://aclanthology.org/2025.naacl-long.545), [UniRoute](https://arxiv.org/abs/2502.08773), Zooter, Arch-Router, LLM이 난이도를 말로 분석하는 방법 | 온라인 학습·새 LLM 일반화·보상 모델 채점·선호 정책 문서·라우팅마다 LLM 호출 등 목표나 조건이 우리와 다르다(10-09 개정본 판단 유지) |

## 5. 공정한 비교 규칙

### 5.1. 데이터와 학습 데이터 버전

- 학습: D_initial_train 21,406문항(R0·R1·R2가 학습한 그 행). 설정 조정·보정: D_dev 3,581문항. 최종 평가: 봉인 test 7,202문항(RouterBench 3,473 + xRouteBench 3,729), 설정을 모두 잠근 뒤 한 번만.
- 후보: 원천별 대표 세 모델과 기록된 성공·비용. RouterBench는 Mistral-7B / Yi-34B / GPT-4(1106), xRouteBench는 gpt-oss-20B / gpt-oss-120B / Cogito-671B다.
- **학습 데이터 버전은 E1 판정을 따른다.** E1이 v2 전환을 판정하면 논문 라우터와 우리 R1·R2를 모두 v2로 학습하고, R0는 E1의 v2 학습본을 쓴다(R1·R2 재학습은 T4 각 약 1시간 추가). 전환하지 않으면 모두 v1으로 학습해 기존 R0·R1·R2와 같은 조건을 맞춘다. 어느 쪽이든 모든 방법이 같은 버전으로 학습한다.
- 평가 채점은 v2를 기준으로 한다. RouterBench 제외 문항(test 28개)을 뺀 분모를 밝히고 v1 값을 함께 적는다.

### 5.2. 후보 표현과 정답 만들기

- 논문 라우터는 원천별 실제 모델 6개(원천당 3개)를 후보로 학습하고, 질문이 속한 원천의 세 모델 중에서 고른다. 우리 R0~R2는 기존처럼 low/medium/high 세 칸을 쓴다.
- "질문마다 가장 잘 푼 모델"을 정답으로 배우는 구현(LLMRouter의 kNN·MLP·MF·GraphRouter)은 동점이면 앞의 모델을 고른다. 후보 순서를 low → medium → high로 고정해 동점이면 가장 싼 단계가 정답이 되게 한다. 셋 다 틀린 질문은 low가 정답이 된다. 우리 oracle 정의와 같다.
- RouteLLM-BERT 재학습(7b)의 정답은 원본과 같은 세 분류다. 강한 모델 = high, 약한 모델 = low로 두고, high만 맞히면 "강한 모델 승", low만 맞히면 "약한 모델 승", 둘 다 맞히거나 둘 다 틀리면 "무승부"다.
- Hybrid LLM은 작은 모델 = low, 큰 모델 = high로 학습한다.
- 모델 설명이 필요한 방법(GraphRouter·IRT-Router의 모델 노드·모델 임베딩)은 모델 이름과 공개 모델 카드 요약으로 만든 설명 문장을 00 노트북에서 고정한다.
- 어떤 라우터도 추론 때 정답 과제 이름(MMLU, GSM8K 등)을 받지 않는다. GraphRouter 구현이 과제 노드를 쓰면 과제 정보는 학습 데이터에만 쓰고, 새 질문은 질문 임베딩만으로 연결한다. 우리 라우터와 같은 조건이다.

### 5.3. 인코더 트랙

| 트랙 | 입력 | 대상 |
|---|---|---|
| 공통 트랙 (주) | 같은 고정 BGE-large-en-v1.5 임베딩(서비스 캐시와 같은 모델, 512토큰) | kNN·MLP·SVM, MF, GraphRouter, Hybrid LLM, IRT-Router, CARROT-KNN·Avengers-Pro(추가 시), R1-BGE, R1-tier |
| 원형 트랙 | 각 논문의 기본 인코더를 fine-tuning | RouterDC(mDeBERTa-v3-base), RouteLLM-BERT 재학습(XLM-RoBERTa), CARROT-RoBERTa(추가 시), 우리 R0·R1·R2(ModernBERT-base) |
| 공개본 | 학습 없음 | RouteLLM-BERT 공개 체크포인트 |

- "방법 차이"는 공통 트랙끼리만 주장한다. 원형 트랙은 파라미터 수와 학습 시간을 함께 적고 같은 인코더 조건이라고 부르지 않는다.
- 라이브러리 기본 임베딩(LLMRouter의 Longformer·Qwen3-Embedding, IRT-Router의 bert-base-uncased, CARROT·Avengers-Pro의 OpenAI 임베딩)은 BGE로 바꾼다. OpenAI 임베딩은 호출하지 않는다.

### 5.4. 조정 예산

- 학습형 방법마다 D_dev에서 **설정 최대 8개**만 시도한다(예: kNN의 k, MF의 차원, IRT의 차원). 설정 후보는 노트북을 만들 때 표로 먼저 고정하고, 시도한 설정과 점수를 모두 남긴다. test는 조정에 쓰지 않는다.
- 설정 선택 기준은 D_dev에서 비용 .25·.5·.75와 κ 격자(0~1)에 걸친 평균 효용이다(원천별로 구해 평균). 동률이면 파라미터 수가 작은 쪽을 고른다.

### 5.5. 평가 규칙 (통일 기준)

- **어댑터:** 모든 라우터의 점수를 D_dev에서 단계별 성공 확률로 보정한다(단계별 isotonic). 점수가 하나뿐인 라우터(RouteLLM-BERT, Hybrid LLM)도 같은 방식으로 세 단계 확률을 만든다. 라이브러리가 고른 모델 이름만 돌려주면 내부 모델에서 점수를 꺼낸다(kNN 확률, MLP·MF 점수, RouterDC 유사도 등). 꺼낼 수 없으면 그 선택을 한 점으로 평가하고 표시한다.
- **결정 규칙:** 한 번에 고르기와 상향을 고려한 시작 단계, 두 가지를 모든 라우터에 똑같이 쓴다. 단계 간 실패 상관 ρ는 D_dev에서 추정한다.
- **판정:** 같은 비용끼리 비교하고, 비용 .25·.5·.75에서 가장 좋은 고정 시작 정책(혼합 포함)을 +.01 넘게 이기기 시작하는 손익분기점 κ\*를 보고한다([평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 2.1절).
- 각 방법이 원래 쓰던 비용 결합 방식(kNN의 willingness-to-pay, IRT·Avengers-Pro의 α 등)의 결과는 부록에만 둔다.

### 5.6. 반복

학습형 방법은 seed 42로 한 번 실행한다. D_dev 평균 효용 상위 2개 학습형 방법에만 seed 17·73을 추가한다(ML 명세 3.9절). kNN처럼 무작위성이 없는 방법은 한 번이다.

### 5.7. 누출 방지

- test outcome은 07 노트북 전까지 어떤 노트북에서도 읽지 않는다. 00 노트북은 test outcome 파일의 SHA만 기록한다.
- 우리 R0·R1·R2의 test 결과는 이미 여러 번 관찰됐다(봉인 평가, λ 탐색, 천장 분석, 통일 평가 시제품). 그래서 우리 방법도 D_dev에서만 고르고, test 값을 보고 조정하지 않는다. 아직 보지 않은 외부 데이터 test(E3)가 확인 실험을 맡는다.
- 외부 코드가 `test`라는 분할로 checkpoint를 고르면 우리 D_dev를 연결한다.

### 5.8. 입력 길이

모든 인코더는 512토큰으로 비교한다. 운영 R0는 8,192토큰 추론을 쓰지만 이번 비교는 512 조건이다. 방법별 truncation 비율을 함께 보고한다.

### 5.9. 외부 코드 관리

- 외부 저장소는 우리 저장소에 복사하지 않는다. Colab에서 실행할 때 커밋을 고정해 내려받고, 커밋 SHA를 manifest에 남긴다.
- 우리가 작성하는 코드는 데이터 형식 변환, 점수 추출, 평가 연결뿐이다.
- 라이선스가 없는 코드와 체크포인트(IRT-Router, RouteLLM 공개 체크포인트, CARROT, Avengers-Pro)는 연구용으로만 실행하고 서비스에는 넣지 않는다. 보고서와 논문에 원 논문을 인용한다.

## 6. 평가 지표 (쉬운 설명)

| 지표 | 쉬운 설명 | 역할 |
|---|---|---|
| 손익분기점 κ\* | 같은 비용에서 이 라우터가 가장 좋은 고정 시작 정책을 이기기 시작하는 "실패 1회의 무게". 낮을수록 더 많은 서비스에서 쓸모 있다. 끝까지 못 이기면 "없음" | **주 지표** |
| 같은 비용 효용 | 비용 .25·.5·.75(always-high = 1)에서 κ별 효용 | **주 지표** |
| 혼합선 대비 이득 | 같은 첫 시도 비용에서 고정 단계 혼합선보다 첫 시도 품질이 얼마나 높은가(RouterBench의 Zero router·AIQ) | 보조 |
| oracle 격차 회복률 | (방법 − 혼합선) / (oracle − 혼합선), 같은 비용(RouteLLM의 APGR) | 보조 |
| 목표 해결률에 필요한 비용 | always-high 첫 시도 성공의 x%에 닿는 최소 비용(RouteLLM의 CPT) | 보조 |
| 4종 세트 | 첫 시도 성공·최종 해결·평균 호출 수·총비용 | 보조 |
| 필요 단계 정확도 | "처음부터 풀 수 있는 가장 싼 단계"를 맞힌 비율과 너무 낮게·높게 고른 비율 | 보조 |
| 비싼 선택 과잉률 | 더 싼 단계도 성공했는데 더 비싼 단계를 고른 비율 | 쏠림 진단 |
| 라우팅 시간·크기 | CPU batch 1의 p50/p95, 파라미터 수, 학습 시간. 서비스가 이미 계산하는 BGE 임베딩을 재사용할 때의 추가 시간을 따로 표시 | 실무 비교 |
| 난이도 해석 | R0 난이도 확률과 IRT-Router 난이도 값이 우리 T1 라벨·RouterArena 공식 라벨과 얼마나 같은 순서인가(순위 상관) | 해석 가능성 |
| 채점 민감도 | v1·v2 채점에서 방법 순위가 얼마나 유지되는가(Kendall τ) | 민감도 |

- 차이는 질문 그룹 단위 paired bootstrap(10,000회, seed 42, 95% 구간)으로 낸다.
- 원천별로 따로 보고한다. xRouteBench는 표본이 작아 원천 단독 우열은 주장하지 않는다. 주장은 두 원천에서 방향이 같은 결과에만 한다.

## 7. 가설과 사전 판정

| 가설 | 내용 | 판정 방법 |
|---|---|---|
| H1 정체 | 학습형 라우터(우리 + 논문)의 차이는 대부분 작고 kNN이 상위권이다 | 평균 효용(5.4절 기준) 1위와 각 방법의 차이 구간이 0을 포함하는지, kNN 순위 |
| H2 난이도 방식 | 어댑터를 거친 R0는 같은 비용에서 성공 예측형(R1·kNN·MF·IRT)과 비슷하다 | 비용 .5에서 효용 차이가 κ 전 구간에서 .01 이내이고 구간이 ±.02 안인지 |
| H3 해석 | IRT-Router의 난이도 값은 우리 T1 라벨·RouterArena 공식 라벨과 양의 상관을 보인다 | 원천별 Spearman 상관과 구간 |
| H4 쏠림 | 상대 우열 학습(R2)은 점수형(R1·kNN·MF)보다 비싼 선택 과잉률이 낮다 | 같은 비용에서 과잉률 차이와 구간 |
| H5 채점 강건성 | 방법 순위는 v1·v2 채점에서 크게 바뀌지 않는다 | Kendall τ ≥ .7이면 유지, 아니면 결론에 주의 문구 |
| H6 비용 예측 (CARROT 추가 시) | 질문별 비용 예측의 이득은 RouterBench에서 거의 없고 xRouteBench에서만 나타난다. 같은 모델 안의 비용 편차 몫이 RouterBench 5%, xRouteBench 19%이기 때문이다(XA-02) | 원천별 CARROT-KNN − R1-BGE의 효용 차이와 구간 |
| H7 직접 학습 | 우리 데이터로 학습한 RouteLLM-BERT(7b)가 공개본(7a)보다 같은 비용에서 효용이 +.01 넘게 높다 | 비용 .5에서 κ 전 구간의 효용 차이와 구간 |

- 어떤 결과도 운영 라우터를 자동으로 바꾸지 않는다. 운영 변경은 별도 결정이다.

## 8. 실행 구조 (10절 확인 후 구현)

### 8.1. 노트북 구성

위치는 `ml/notebooks/router_comparison/`이다. 각 노트북은 앞 단계 산출물의 SHA를 확인한 뒤 시작한다.

| 노트북 | 하는 일 | 장비 | 예상 시간 |
|---|---|---|---|
| `00_prepare` | A 분할과 학습 데이터 버전 확인, 원천별 후보 6개, 정답 만들기(동점 규칙·RouteLLM 세 분류), 모델 설명 문장 고정, R1-tier 그룹 봉인, 외부 저장소 커밋 고정, 형식 변환(RouterBench·LLMRouter·IRT-Router·CARROT·Avengers-Pro), 모든 입력 SHA 봉인. test outcome은 SHA만 기록 | CPU | 20분 |
| `01_embed` | train·D_dev·test·RouterArena 문항의 BGE-large 임베딩을 한 번 계산해 봉인 | T4 | 20~30분 |
| `02_common_track` | kNN·MLP·SVM, MF, GraphRouter, Hybrid LLM, IRT-Router, (CARROT-KNN, Avengers-Pro), R1-BGE, R1-tier 학습과 D_dev 설정 선택(방법당 ≤8개) | CPU, MF·GraphRouter는 T4 선택 | 1~2시간 |
| `03_finetune_track` | RouterDC, RouteLLM-BERT 재학습, (CARROT-RoBERTa) 학습과 D_dev checkpoint 선택 | T4 | 2~4시간 |
| `04_released` | RouteLLM-BERT 공개 체크포인트로 D_dev·test 점수 계산 | T4 | 10분 |
| `05_ours` | R0·R0-τ·R1·R2의 D_dev·test 점수 준비. R2는 Drive·GCS에서 바로 읽는다 | CPU | 10분 |
| `06_lock` | 모든 방법의 어댑터 보정·ρ·결정 규칙을 `selection-lock.json`으로 고정한다. **여기서 멈추고 사용자에게 lock 내용을 보고한다** | CPU | 10분 |
| `07_test` | 봉인 test 1회 평가(v2 채점, v1 병기), 손익분기점·지표 계산, CPU 라우팅 시간 측정 | CPU | 30분 |
| `08_seeds` | D_dev 상위 2개 학습형 방법에 seed 17·73 추가(잠근 설정 그대로) | T4 | 1~2시간 |
| `09_report` | 표·곡선·bootstrap·REPORT 초안 | CPU | 30분 |

### 8.2. 산출물과 보관

- 원본 결과: Google Drive `LLM_Router/router-comparison/runs/run001/`(캐시 실험과 같은 구조).
- 저장소 반입: 요약·그림·manifest·REPORT만 `experiments/results/ev06-router-comparison-<날짜>/`에 넣는다. 큰 parquet·임베딩·checkpoint·외부 코드는 넣지 않는다.
- 필수 기록: 데이터·config·모델 revision hash, 외부 저장소 커밋 SHA, 후보 ID, seed, 시도한 설정 전부, 학습 시간, 파라미터 수, `selection-lock.json`, test 예측.

### 8.3. 자원과 중단 조건

- 실행 환경은 개인 Colab T4다. GCP Vertex·GKE는 쓰지 않는다.
- 예상 총량은 **T4 약 4~7시간, 약 8~14 CU(추정)**다. 01 단계 실측으로 다시 추정한다.
- 상한은 **20 CU**다. 남은 단계 추정이 상한을 넘으면 08 seed 반복과 추가 후보의 RoBERTa 변형부터 줄이고, 그래도 넘으면 멈추고 보고한다. 방법별 조정 예산(8개)은 줄이지 않는다.
- 외부 코드가 우리 데이터 형식이나 후보 구성에서 동작하지 않거나 원 논문 설정과 크게 다르면, 억지로 맞추지 않고 "재현 실패/부분 차용"으로 기록한 뒤 결과표에서 구분한다.
- 새 LLM 호출이나 유료 API는 쓰지 않는다.

### 8.4. 다른 실험과의 순서

E2(평가 코드 고정) → E1(학습 데이터 버전 판정) → **EV-06** → E3(외부 데이터 전이. EV-06에서 잠근 라우터를 그대로 넣는다) → 보고서. 순서와 E1·E2·E3의 내용은 [라우터 평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 6·8절이 기준이다.

## 9. 위험과 대응

| 위험 | 대응 |
|---|---|
| 외부 코드가 우리 형식이나 후보 3개에 맞지 않는다 | 00 단계에서 각 저장소의 입력 형식을 먼저 확인한다. 막히면 같은 방법의 LLMRouter 구현으로 바꾸고 이름을 구분한다. 그래도 안 되면 "재현 실패"로 기록한다 |
| 라이브러리가 고른 모델 이름만 돌려준다 | 내부 모델에서 점수를 꺼낸다. 꺼낼 수 없으면 선택 한 점으로 평가하고 표시한다(5.5절) |
| 라이선스가 없는 코드 | 연구용으로만 실행하고 저장소·서비스에 넣지 않는다(5.9절) |
| v1 학습 라벨 오류가 절대 수치를 왜곡한다 | 학습 데이터 버전은 E1 판정을 따르고, 평가는 v2 채점으로 하며, H5로 순위 유지를 확인한다 |
| xRouteBench 표본이 작아 결과가 흔들린다 | 원천별로 보고하고, 두 원천의 방향이 같을 때만 주장한다 |
| 우리 방법만 test를 이미 봤다 | 5.7절 규칙과 E3 확인 실험. 보고서에 관찰 이력을 적는다 |
| 인코더가 방법마다 달라 "방법 차이"로 읽힌다 | 5.3절 트랙 구분. 공통 트랙끼리만 방법 차이를 주장한다 |

## 10. 결정 항목

**확정된 사항**

| 항목 | 내용 | 날짜 |
|---|---|---|
| 비교 대상 | kNN 효용 라우터(+MLP·SVM), MF, RouterDC, GraphRouter, Hybrid LLM 모두 | 2026-10-16 |
| IRT-Router | 라이선스가 없지만 연구용으로 쓴다(5.9절) | 2026-10-16 |
| RouteLLM | 공개 체크포인트 그대로(7a)와 우리 데이터 학습본(7b)을 둘 다 쓴다 | 2026-10-16 |
| 평가 기준 | 통일 기준과 손익분기점 방식([평가 계획서](ROUTER_EVALUATION_V2_PLAN.md) 9절 결정 1) | 2026-10-16 |
| 논문 비교 유지 | 평가 계획서로 축소·대체하지 않고 별도로 한다(평가 계획서 9절 결정 6) | 2026-10-13 |

**남은 결정**

| # | 결정 | 선택지 | 권장 |
|---|---|---|---|
| 1 | CARROT 추가 | 추가 / 제외 | 추가. R1과 구조가 같아 질문별 비용 예측의 효과(H6)를 볼 수 있다. KNN 변형(공통 트랙, CPU)은 넣고, RoBERTa 변형은 자원이 남을 때만 |
| 2 | Avengers-Pro 추가 | 추가 / 제외 | 추가. 질문 묶음 단위로 고르는 최근 방법이라 "과제 유형만으로 얼마나 되나"를 대표하고 CPU로 끝난다 |
| 3 | R1-tier | 포함 / 제외 | 포함. CPU로 끝나고, "모델을 급으로 묶어 배우면 나아지나"라는 10-09 논의를 직접 검증한다 |
| 4 | 자원 상한 | 개인 Colab T4, 20 CU / 다른 값 | 20 CU. 넘을 것 같으면 seed 반복부터 줄이고 보고한다 |

확정되면 ML 명세 3.3·3.8절의 비교군 표를 이 문서에 맞춰 고치고, 그 뒤에 노트북을 만든다. 노트북을 만든 뒤에도 `06_lock`에서 한 번 멈추고 lock 내용을 확인받은 다음 test를 연다.

## 11. 참고 문헌

- RouterBench: [Hu et al., 2024](https://arxiv.org/abs/2403.12031), [코드·데이터](https://github.com/withmartian/routerbench)
- RouteLLM: [Ong et al., ICLR 2025](https://arxiv.org/abs/2406.18665), [코드](https://github.com/lm-sys/RouteLLM)
- Hybrid LLM: [Ding et al., ICLR 2024](https://arxiv.org/abs/2404.14618)
- RouterDC: [Chen et al., NeurIPS 2024](https://arxiv.org/abs/2409.19886), [코드](https://github.com/shuhao02/RouterDC)
- GraphRouter: [Feng et al., ICLR 2025](https://arxiv.org/abs/2410.03834), [코드](https://github.com/ulab-uiuc/GraphRouter)
- IRT-Router: [Song et al., ACL 2025](https://arxiv.org/abs/2506.01048), [코드](https://github.com/Mercidaiha/IRT-Router)
- CARROT·SPROUT: [Somerstep et al., 2025](https://arxiv.org/abs/2502.03261), [코드](https://github.com/somerstep/CARROT)
- Avengers-Pro: [Zhang et al., 2025](https://arxiv.org/abs/2508.12631), [코드](https://github.com/ZhangYiqun018/AvengersPro)
- LLMRouter 라이브러리: [arXiv 2608.06867](https://arxiv.org/abs/2608.06867), [코드](https://github.com/ulab-uiuc/LLMRouter)
- LLMRouterBench: [Li et al., 2026](https://arxiv.org/abs/2601.07206)
- Routing Plateau: [Lu et al., 2026](https://arxiv.org/abs/2606.07587)
- Routing Collapse·EquiRouter: [Lai & Ye, 2026](https://arxiv.org/abs/2602.03478)
- 과제 유형과 라우팅 격차: [arXiv 2608.23023](https://arxiv.org/abs/2608.23023)
- EmbedLLM: [Zhuang et al., ICLR 2025](https://arxiv.org/abs/2410.02223)
- AutoMix: [Madaan et al., NeurIPS 2024](https://arxiv.org/abs/2310.12963)
- RouterArena: [Lu et al., ICLR 2026](https://arxiv.org/abs/2510.00202)
