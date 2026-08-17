# LLM Routing 논문 조사 (2024–2026)

기준일: 2026-08-17
작업 브랜치: `feat/data-labeling-training`
관련 문서: `docs/ml/routerbench-data-labeling-spec.md`, `docs/ml/routerarena-routing-evaluation-spec.md`

## 1. 수집 범위와 기준

- **포함**: 쿼리(질문)를 분석해 **작업 난이도/복잡도를 평가**하고, 그 결과에 따라 **성능·비용이 다른 LLM 후보 중 하나(또는 캐스케이드 순서)를 매칭**하는 라우팅 연구. 즉 "이 질문은 쉬우니 소형 모델로, 어려우니 대형 모델로" 유형의 난이도 기반 model routing.
- **제외**: 작업 종류(이미지 생성 vs 텍스트 생성, 코드 vs 번역 등 서로 다른 목적)에 따라 별도 전문 모델로 나누는 **task-type routing / MoE 계열 routing**. Mixture-of-Experts 내부 라우팅(FFN expert 선택), 멀티모달 modality routing 논문은 제외했다.
- **기간**: 2024–2026(현재). FrugalGPT, Tryage, AutoMix, Zooter 등 2023년 발표 논문은 난이도 기반 routing의 직접적 기반 연구이지만 조사 기간 밖이라 [9장](#9-참고-2024년-이전-선행-연구)에 별도로만 정리했다.
- **신뢰도 표기**: 라우팅 벤치마크(RouterBench/RouterArena) 사용 여부는 논문 본문·공식 자료를 확인했으나, 일부는 초록·2차 자료 기반 판단이라 `(확인 필요)`로 표시했다. 인용 표시가 없는 항목은 논문 원문 또는 공식 설명에서 직접 확인한 내용이다.

## 2. 문서 구성

- 3장: RouterBench/RouterArena 벤치마크 자체
- 4장: RouterBench 또는 RouterArena 데이터셋을 실험에 사용한 논문
- 5장: RouterBench/RouterArena를 사용하지 않고 자체 데이터셋·다른 벤치마크로 평가한 논문
- 6장: 위 두 그룹을 통합한 비교 표(라우팅 기법 / 라우팅 모델 / 모델 종류)
- 7장: 서베이·메타분석 논문
- 8장: 전체 링크 모음
- 9장: 2024년 이전 선행 연구(범위 밖, 참고용)
- 10장: 조사 방법 메모

> 2026-08-17 갱신: 이전 버전의 "미검증 논문" 9건을 전부 원문 대조로 검증해 4/5/6/7장에 편입했다(검증 로그는 10장 참고).

---

## 3. RouterBench / RouterArena 벤치마크 자체

| 이름 | 발표 | 핵심 내용 | 링크 |
| --- | --- | --- | --- |
| **RouterBench** | 2024-03 (ICML 2024 Workshop) | 11개 LLM × MMLU/MT-Bench/MBPP/HellaSwag/Winogrande/GSM8K/ARC 등 7개 태스크에 대한 40만5천여 건의 사전 계산 추론 결과. Non-predictive/predictive router 비교와 AIQ(비용-품질 곡선 아래 면적) 지표를 제안 | [arXiv:2403.12031](https://arxiv.org/abs/2403.12031) · [GitHub](https://github.com/withmartian/routerbench) |
| **RouterArena** | 2025-10 (ICLR 2026) | 23개 오픈소스 데이터셋을 LLM 기반 난이도 라벨링으로 재구성한 8,400개 질의, 9개 도메인 × 44개 카테고리 × 난이도 3단계(easy/medium/hard). Accuracy/Cost/Optimality/Robustness/Latency 5개 지표와 라이브 리더보드 제공 | [arXiv:2510.00202](https://arxiv.org/abs/2510.00202) · [GitHub](https://github.com/RouteWorks/RouterArena) |
| RouterEval | 2025-03 (EMNLP 2025 Findings) | RouterBench와 별개의 자체 데이터(8,500개 이상 LLM, 12개 평가셋, 2억 건 이상 성능 레코드)로 "router 성능이 후보 수에 따라 스케일업되는지"를 검증. RouterBench/RouterArena 데이터를 재사용하지 않음 | [arXiv:2503.10657](https://arxiv.org/abs/2503.10657) |
| LLMRouterBench | 2026-01 (ACL 2026 Findings) | 21개 데이터셋 × 33개 모델, 40만 건 이상 인스턴스. RouterBench·RouterEval 등 기존 벤치마크의 한계(작은 후보 풀, 비용 미고려 등)를 지적하며 통합·확장한 대규모 벤치마크 | [arXiv:2601.07206](https://arxiv.org/abs/2601.07206) · [GitHub](https://github.com/ynulihao/LLMRouterBench) |

---

## 4. RouterBench 또는 RouterArena 데이터셋을 사용한 논문

RouterBench/RouterArena 자체 논문을 제외하고, 실험에서 두 데이터셋 중 하나 이상을 직접 사용한 것으로 확인된 논문이다.

| 논문 | 발표 | 사용 데이터셋 | 비고 |
| --- | --- | --- | --- |
| **Eagle: Efficient Training-Free Router for Multi-LLM Inference** | 2024-09 (NeurIPS 2024 workshop) | RouterBench 전체 태스크(MMLU, MT-Bench, GSM8K, ARC-C, Winogrande, MBPP, HellaSwag) | 논문 본문에서 RouterBench 전체 데이터셋으로 baseline(SVM/KNN/MLP) 대비 성능을 직접 비교 |
| **LLM Routing with Dueling Feedback** | 2025-10 | RouterBench | Dueling-bandit 기반 온라인 라우팅을 RouterBench 로그로 평가 |
| **LLMRank: Understanding LLM Strengths for Model Routing** | 2025-10 | RouterBench zero-shot 변형(36,497개 프롬프트) | RouterBench 0-shot 서브셋을 라우팅 실험용으로 재가공해 사용 |
| **BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute** | 2025-06 (ICML 2025) | RouterBench `(확인 필요 — 2차 자료 기준)` | DeBERTa-v3-small 멀티헤드 분류기로 난이도 추정 후 모델·샘플링 수를 함께 선택 |
| **RouterArena 논문 내 비교군(CARROT, MIRT-BERT, NIRT-BERT, RouteLLM, RouterDC, GraphRouter)** | 2025-10 | RouterArena | RouterArena 저자들이 기존 라우팅 기법 6종을 RouterArena 데이터로 재구현·재평가한 비교 실험. **RouteLLM·RouterDC·GraphRouter 원 논문 자체는 RouterBench/RouterArena를 쓰지 않았음**(→ 5장에 별도 수록) |
| **A Unified Approach to Routing and Cascading for LLMs** | 2024-10 | RouterBench(주 실험) + ARC-Challenge/MMLU-Pro/MixEval/GSM8K(보조 실험) | 라우팅과 캐스케이딩을 하나의 최적화 문제(cascade routing)로 통합한 프레임워크. 주 실험 전체를 RouterBench로 수행 |
| **Federate the Router: Learning Language Model Routers with Sparse and Decentralized Evaluations** | 2026-01 | "RouterBench-Data"(RouterBench 11개 LLM × 8개 데이터셋 기반 재구성) + 자체 ProxRouter-Data | 연합학습 환경에서 RouterBench 파생 데이터를 두 벤치마크 중 하나로 사용 |
| **VDAR-Router: Adaptive LLMs Routing via Verbalized Query Difficulty Analysis Retrieval** | 2026-07 | RouterBench + LLMRouterBench + ArenaExpert5K | 3개 평가 데이터셋 중 2개가 RouterBench 계열(RouterBench, LLMRouterBench) |
| **OrcaRouter: A Production-Oriented LLM Router with Hybrid Offline–Online Learning** | 2026-05 | RouterArena | 공개 RouterArena 리더보드에서 "OrcaRouter-Adaptive"로 2위(Arena Score 72.08, 정확도 75.54%, 1,000쿼리당 $1.00) 기록 |

> 참고: LLMRouterBench(3장)는 RouterBench·RouterEval 데이터를 통합해 확장했다고 명시하므로, RouterBench 계열 데이터를 부분적으로 포함한다.
> 참고: **The Routing Plateau**(2606.07587)는 21개 기존 라우팅 기법을 RouterBench·R2-Bench·EmbedLLM-bench·CARROT/SPROUT·Nine-by-30k(신규 구성) 5개 벤치마크로 비교한 메타분석 논문이라 RouterBench를 사용하지만, 개별 라우팅 기법을 제안하지 않아 이 표 대신 7장에 수록했다.

---

## 5. RouterBench/RouterArena를 사용하지 않은 논문

각자 자체 수집 데이터셋(Chatbot Arena, MMLU/GSM8K 조합, 자체 대화 로그 등) 또는 RouterBench/RouterArena 이외의 벤치마크로 평가한 논문이다.

| 논문 | 발표 | 평가 데이터 |
| --- | --- | --- |
| RouteLLM: Learning to Route LLMs with Preference Data | 2024-06 (ICLR 2025) | Chatbot Arena 선호 데이터 + MMLU/GSM8K/MT-Bench |
| Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing | 2024-04 (ICLR 2024) | 자체 난이도 라벨 데이터셋 |
| GraphRouter: A Graph-based Router for LLM Selections | 2024-10 (ICLR 2025) | 자체 구성 query-LLM 그래프 데이터셋 |
| RouterDC: Query-Based Router by Dual Contrastive Learning | 2024-09 (NeurIPS 2024) | 자체 in-distribution/OOD 태스크 조합 |
| MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs | 2024-07 | 자체 분류/QA 태스크 세트 |
| EmbedLLM: Learning Compact Representations of Large Language Models | 2024-10 | 자체 모델-질의 상호작용 데이터 |
| PickLLM: Context-Aware RL-Assisted Large Language Model Routing | 2024-12 (AAAI 2025 workshop) | 자체 비용/지연/정확도 시뮬레이션 환경 |
| IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory | 2025-06 (ACL 2025) | 20개 LLM × 12개 데이터셋(자체 조합, RouterBench 미포함) |
| Arch-Router: Aligning LLM Routing with Human Preferences | 2025-06 | 자체 대화형 도메인/액션 라벨 데이터셋 |
| MixLLM: Dynamic Routing in Mixed Large Language Models | 2025-02 | 자체 컨텍스추얼 밴딧 시뮬레이션/실서비스 로그 |
| Causal LLM Routing: End-to-End Regret Minimization from Observational Data | 2025-05 | 관측 데이터(observational log) 기반 자체 데이터셋 |
| LLM Router: Rethinking Routing with Prefill Activations | 2026-03 | 자체 prefill activation 데이터셋(Encoder-Target Decoupling 실험) |
| vLLM Semantic Router / When to Reason: Semantic Router for vLLM | 2025-10~2026-03 | MMLU-Pro 등 자체 라우팅 정책 실험 |
| RADAR: Reasoning-Ability and Difficulty-Aware Routing | 2025-09 (NeurIPS 2025 workshop) | 수학/STEM/QA 자체 조합 데이터셋 |
| CP-Router: An Uncertainty-Aware Router Between LLM and LRM | 2025-05 (AAAI 2026) | 자체 난이도-불확실성 데이터셋 |
| ICL-Router: In-Context Learned Model Representations for LLM Routing | 2025-10 (AAAI 2025) | 자체 in-context 능력 임베딩 데이터셋 |
| RouterEval | 2025-03 | 자체 8,500개 LLM 성능 레코드(3장 참고) |
| AutoMix: Automatically Mixing Language Models(2023 발표, 2024 NeurIPS 게재) | 2023-10 / 2024 게재 | 자체 5개 데이터셋 |
| Routing, Cascades, and User Choice for LLMs | 2026-02 | 데이터셋 없음(순수 이론·게임이론 분석, 학습된 라우터 미구현) |
| Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving | 2026-06 | 자체 클러스터링·비용 데이터(RouterBench/RouterArena 언급 없음) |
| UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing | 2026-05 | 실서비스 워크로드(4B/12B 모델의 75,000건 개체명 인식 요청, H100 서빙 로그) |
| Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems | 2026-02 | 자체 제안 벤치마크 RouterXBench(라우터 능력·시나리오 정합성·교차 도메인 강건성 3축) |

---

## 6. 논문별 라우팅 기법 · 라우팅 모델 · 모델 종류 비교표

"모델 종류"는 라우터가 내부적으로 사용하는 예측기의 유형을 의미한다(분류에 최적화된 소형 인코더 / 소형 LLM / GNN·행렬분해 임베딩 / 밴딧·RL 정책 / 학습 없는 휴리스틱 등). RouterBench/RouterArena 사용 여부는 4·5장 기준을 따른다.

| 논문 | 라우팅 기법 | 라우팅 모델(아키텍처) | 모델 종류 | RB/RA 사용 |
| --- | --- | --- | --- | --- |
| RouteLLM (2406.18665) | 선호 데이터 기반 이진 분류(강한 모델 vs 약한 모델) | Matrix Factorization router / BERT 분류기 / Causal LLM(파인튜닝) 등 4종 router 비교 | 분류에 최적화된 소형 모델(행렬분해·BERT) + 소형 LLM 분류기 | 미사용 |
| Hybrid LLM (2404.14618) | 난이도 예측 후 소/대형 모델 이진 라우팅, 품질-비용 트레이드오프를 test-time에 조정 | DeBERTa 기반 회귀/분류기 | 분류·회귀에 최적화된 소형 인코더 | 미사용 |
| GraphRouter (2410.03834) | 질의-태스크-LLM을 이종 그래프로 모델링 후 링크 예측 | Graph Neural Network(GNN) | 그래프 기반 임베딩 모델 | 미사용 |
| RouterDC (2409.19886) | 듀얼 대조학습으로 질의-LLM 임베딩 정렬 | 인코더 + LLM 임베딩(대조학습) | 임베딩 기반 소형 인코더 | 미사용 |
| Eagle (2409.15518) | 전역/지역 ELO 랭킹 기반, 학습 없는(training-free) 라우팅 | ELO 랭킹 모듈(글로벌+로컬) | 학습 없는 휴리스틱/랭킹 모델(분류기 아님) | **사용(RouterBench)** |
| MetaLLM (2407.10834) | 멀티암드 밴딧으로 비용-정확도 보상 최적화 | 밴딧 정책(reward function 기반) | 강화학습형 밴딧 정책(경량) | 미사용 |
| EmbedLLM (2410.02223) | LLM 자체를 벡터로 임베딩 후 선형 계층으로 라우팅 | 인코더-디코더 임베더 + 선형 라우팅 계층 | 임베딩 기반 경량 모델 | 미사용 |
| PickLLM (2412.12170) | 강화학습(정책경사/Q-learning)으로 질의별 LLM 선택 | 학습 오토마톤 / stateless Q-learning | 강화학습 정책(경량) | 미사용 |
| IRT-Router (2506.01048) | 문항반응이론(IRT)으로 LLM 능력과 질의 난이도를 동시 모델링 | IRT 잠재변수 모델(다차원 능력·난이도 추정) | 통계적 잠재변수 모델(해석가능) | 미사용(RB 미포함 12개 데이터셋) |
| Arch-Router (2506.16655) | 사용자 정의 도메인/액션 선호에 질의를 매칭 | 1.5B 소형 LLM | 소형 LLM(경량 파인튜닝) | 미사용 |
| MixLLM (2502.18482) | 컨텍스추얼 밴딧 + 정책경사로 질의-LLM 배정 | 도메인 태그 강화 임베딩 + 정책경사 메타 결정자 | 임베딩 + 강화학습 정책 | 미사용 |
| BEST-Route (2506.22716) | 난이도 추정 후 모델과 응답 샘플 수(best-of-n)를 함께 선택 | DeBERTa-v3-small 멀티헤드 분류기 | 분류에 최적화된 소형 인코더 | 사용(RouterBench) `(확인 필요)` |
| LLM Routing with Dueling Feedback (2510.00841) | 듀얼링(쌍대 비교) 밴딧으로 온라인 학습 | 듀얼링 밴딧 알고리즘 | 강화학습형 밴딧 정책 | **사용(RouterBench)** |
| LLMRank (2510.01234) | LLM별 강점 프로파일을 학습해 랭킹 기반 라우팅 | 랭킹 학습 모델 | 임베딩/랭킹 기반 경량 모델 | **사용(RouterBench zero-shot)** |
| Causal LLM Routing (2505.16037) | 관측 데이터에서 회귀(regret) 최소화를 직접 학습(디커플링 없이 end-to-end) | 분류 기반 서로게이트 목적함수를 사용하는 정책 모델 | 분류/정책 결합 경량 모델 | 미사용 |
| LLM Router: Prefill Activations (2603.20895) | 의미론적 질의 특징 대신 prefill 내부 활성값으로 정답 여부 예측 | SharedTrunkNet(Encoder-Target Decoupling) | 활성값 기반 경량 예측 모델 | 미사용 |
| vLLM Semantic Router / When to Reason (2603.04444, 2510.08731) | 13종 신호(의미·안전·모달리티)를 불리언 규칙으로 조합해 추론모드/모델 결정 | ModernBERT 기반 다중 분류기(LoRA) | 분류에 최적화된 소형 인코더(ModernBERT) | 미사용 |
| RADAR (2509.25426) | 추론능력·난이도 인식 라우팅(수학/STEM 중심) | 난이도 분류기(세부 아키텍처 미확인) | 분류 모델(경량, 확인 필요) | 미사용 |
| CP-Router (2505.19970) | 불확실성 신호로 LLM vs LRM(추론모델) 사이 라우팅 | 불확실성 기반 임계값/분류기 | 경량 분류/임계값 모델 | 미사용 |
| ICL-Router (2510.09719) | in-context 예시로 학습한 모델 능력 임베딩으로 라우팅 | in-context 임베딩 표현 모델 | 임베딩 기반 경량 모델 | 미사용 |
| A Unified Approach to Routing and Cascading (2410.10347) | 라우팅+캐스케이딩 통합, 재라우팅 허용 순차 캐스케이드. 비용-품질 스코어 τ=q̂−λĉ 로 모델 선택 | 정확도 예측용 소형 분류기 + perplexity 기반 선형 품질 추정기, 토큰 수 기반 비용 추정기 | 분류/회귀 결합형 경량 추정기 | **사용(RouterBench, 주 실험)** |
| Federate the Router (2601.22318) | 클라이언트별 로컬 평가 데이터로 라우팅 정책을 연합학습(federated learning)으로 공동 학습 | 파라메트릭 MLP router 또는 비파라메트릭 K-means router 중 선택 | 경량 MLP 분류기 / K-means 비모수 라우터(연합학습) | 사용(RouterBench 기반 "RouterBench-Data") |
| VDAR-Router (2607.18098) | LLM으로 질의 난이도를 언어화(verbalize)한 뒤 유사 난이도 과거 사례를 검색, 보상함수로 모델 선택. 학습 없는(training-free) 방식 | LLM 기반 난이도 verbalizer + 검색(retrieval) 인덱스 + 보상함수 | 학습 없는 LLM-as-judge + 검색 결합형 | **사용(RouterBench, LLMRouterBench)** |
| OrcaRouter (2605.30736) | 어휘·문장임베딩 특징 기반 LinUCB 컨텍스추얼 밴딧. 오프라인에서 보상행렬·arm별 릿지회귀로 초기화 후 온라인에서 선택된 arm만 갱신 | LinUCB 밴딧 + arm(모델)별 릿지회귀 | 컨텍스추얼 밴딧 정책(경량 선형모델) | **사용(RouterArena 리더보드 2위)** |
| Routing, Cascades, and User Choice (2602.09902) | 표준/추론 모델 간 라우팅을 제공자-사용자 슈타켈베르크 게임으로 분석, 난이도·지연시간 기반 최적 정적 정책을 이론적으로 도출 | 없음(학습된 라우터 미구현, 순수 이론 모델) | 이론적 프레임워크(구현체 아님) | 미사용 |
| Cluster, Route, Escalate (2606.27457) | 1단계: 질의를 클러스터링해 최적 비용 모델에 배정 → 2단계: 품질 추정 기반 캐스케이드 에스컬레이션 | 클러스터링 + 정확도 예측 소형 분류기(품질 추정기) | 클러스터링 + 경량 분류기 결합형 | 미사용 |
| UCCI (2605.18796) | 토큰 단위 마진 불확실성을 등위회귀(isotonic regression)로 질의별 오류확률에 보정, 비용 제약 하 에스컬레이션 임계값 선택 | Isotonic regression 보정기(신경망 라우터 아님) | 통계적 보정 모델(비신경망) | 미사용 |
| ProbeDirichlet / Towards Fair Evaluation (2602.11877) | 답변 생성 전 내부 hidden state의 불확실성을 포착해 라우팅 | 레이어 간 hidden state를 학습 가능한 Dirichlet 분포로 집계하는 경량 probe | hidden state 기반 확률적 probe(경량) | 미사용(자체 RouterXBench) |
| RouterBench 자체 baseline (2403.12031) | KNN/MLP/유사도 기반 predictive router + oracle 비교 | KNN, MLP, 유사도 기반 predictor | 분류/회귀에 최적화된 경량 모델(비교군) | 데이터셋 원 논문 |
| RouterArena 자체 baseline 비교(2510.00202) | 기존 6개 기법(RouteLLM 등)을 RouterArena로 재평가 | 각 원 논문 아키텍처 재사용 | 위 각 항목과 동일 | **사용(RouterArena)** |

---

## 7. 서베이 / 메타분석 논문 (참고용, 6장 표에서 제외)

개별 라우팅 기법을 제안하기보다 라우팅 분야 전체 또는 다수 기법을 종합 비교한 논문이다.

- **Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in LLM-Based Systems** (2025-02) — [arXiv:2502.00409](https://arxiv.org/abs/2502.00409)
- **Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey** (2026-03) — [arXiv:2603.04445](https://arxiv.org/abs/2603.04445)
- **Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques** (2025-06) — [arXiv:2506.06579](https://arxiv.org/html/2506.06579v1)
- **The Routing Plateau: Understanding and Breaking the Accuracy Limits of LLM Routers** (2026-06) — 21개 라우팅 기법을 RouterBench·R2-Bench·EmbedLLM-bench·CARROT/SPROUT·Nine-by-30k(신규 구성) 5개 벤치마크로 비교. 기존 라우터가 "질의별 세부 신호보다 모델 평균 성능 경향을 학습"하는 데 그친다는 정체(plateau) 현상을 지적 — [arXiv:2606.07587](https://arxiv.org/abs/2606.07587)
- **Awesome-Routing-LLMs (GitHub 큐레이션 리스트)** — [github.com/MilkThink-Lab/Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs)

> The Routing Plateau에서 확인된 R2-Bench, EmbedLLM-bench, CARROT/SPROUT은 RouterBench/RouterArena 외에 존재하는 또 다른 라우팅 벤치마크다. 이번 조사 범위(RouterBench/RouterArena 사용 여부)에는 해당하지 않아 3장에는 넣지 않았지만, 후속 조사 시 참고할 것.

## 8. 전체 링크 모음 (arXiv ID 순)

- FrugalGPT(참고, 범위 밖) — [2305.05176](https://arxiv.org/abs/2305.05176)
- Tryage(참고, 범위 밖) — [2308.11601](https://arxiv.org/abs/2308.11601)
- Routing to the Expert / Zooter(참고, 범위 밖) — [2311.08692](https://arxiv.org/abs/2311.08692)
- AutoMix(참고, 범위 밖) — [2310.12963](https://arxiv.org/abs/2310.12963)
- RouterBench — [2403.12031](https://arxiv.org/abs/2403.12031)
- Hybrid LLM — [2404.14618](https://huggingface.co/papers/2404.14618)
- RouteLLM — [2406.18665](https://arxiv.org/abs/2406.18665)
- MetaLLM — [2407.10834](https://arxiv.org/pdf/2407.10834)
- Eagle — [2409.15518](https://arxiv.org/abs/2409.15518)
- RouterDC — [2409.19886](https://arxiv.org/abs/2409.19886)
- GraphRouter — [2410.03834](https://github.com/ulab-uiuc/GraphRouter)
- EmbedLLM — [2410.02223](https://arxiv.org/abs/2410.02223)
- **A Unified Approach to Routing and Cascading for LLMs** — [2410.10347](https://arxiv.org/html/2410.10347v1)
- PickLLM — [2412.12170](https://arxiv.org/abs/2412.12170)
- RouterEval — [2503.10657](https://arxiv.org/abs/2503.10657)
- MixLLM — [2502.18482](https://arxiv.org/abs/2502.18482)
- Doing More with Less(서베이) — [2502.00409](https://arxiv.org/abs/2502.00409)
- Causal LLM Routing — [2505.16037](https://arxiv.org/abs/2505.16037)
- IRT-Router — [2506.01048](https://arxiv.org/abs/2506.01048)
- Arch-Router — [2506.16655](https://arxiv.org/abs/2506.16655)
- BEST-Route — [2506.22716](https://github.com/microsoft/best-route-llm)
- Towards Efficient Multi-LLM Inference(서베이) — [2506.06579](https://arxiv.org/html/2506.06579v1)
- CP-Router — [2505.19970](https://arxiv.org/abs/2505.19970)
- RADAR — [2509.25426](https://arxiv.org/abs/2509.25426)
- RouterArena — [2510.00202](https://arxiv.org/abs/2510.00202)
- LLM Routing with Dueling Feedback — [2510.00841](https://arxiv.org/pdf/2510.00841)
- LLMRank — [2510.01234](https://arxiv.org/pdf/2510.01234)
- ICL-Router — [2510.09719](https://arxiv.org/abs/2510.09719)
- Optimizing Reasoning Efficiency through Prompt Difficulty Prediction — [2511.03808](https://arxiv.org/abs/2511.03808)
- **Federate the Router** — [2601.22318](https://arxiv.org/abs/2601.22318)
- LLMRouterBench — [2601.07206](https://arxiv.org/abs/2601.07206)
- **Routing, Cascades, and User Choice for LLMs** — [2602.09902](https://arxiv.org/abs/2602.09902)
- **Towards Fair and Comprehensive Evaluation of Routers (ProbeDirichlet / RouterXBench)** — [2602.11877](https://arxiv.org/html/2602.11877v1)
- Dynamic Model Routing and Cascading(서베이) — [2603.04445](https://arxiv.org/abs/2603.04445)
- LLM Router: Prefill Activations — [2603.20895](https://arxiv.org/abs/2603.20895)
- vLLM Semantic Router — [2603.04444](https://arxiv.org/html/2603.04444v4)
- **UCCI** — [2605.18796](https://arxiv.org/abs/2605.18796)
- **OrcaRouter** — [2605.30736](https://arxiv.org/html/2605.30736v1)
- **Cluster, Route, Escalate** — [2606.27457](https://arxiv.org/abs/2606.27457)
- **The Routing Plateau**(메타분석, 7장) — [2606.07587](https://arxiv.org/abs/2606.07587)
- **VDAR-Router** — [2607.18098](https://arxiv.org/abs/2607.18098)

---

## 9. 참고: 2024년 이전 선행 연구

조사 범위(2024–2026) 밖이지만 난이도 기반 routing/cascading의 직접적 기반이 되는 2023년 논문이다.

| 논문 | 라우팅 기법 | 라우팅 모델 | 모델 종류 |
| --- | --- | --- | --- |
| FrugalGPT (2305.05176) | 순차적 캐스케이드, 품질 임계값 도달 시 정지 | 프롬프트 적응 + LLM 근사 + 캐스케이드 스코어러 | 경량 스코어러(회귀) |
| Tryage (2308.11601) | 실시간 프롬프트 라우팅 | 라우팅 함수(세부 아키텍처는 원문 확인 필요) | 확인 필요 |
| Zooter / Routing to the Expert (2311.08692) | 보상모델 증류 기반 라우팅 함수 학습 | 태그 기반 라벨 보강 라우팅 함수 | 경량 분류기 |
| AutoMix (2310.12963) | 소형 모델 생성 → 자가검증 → POMDP 기반 대형 모델 에스컬레이션 | few-shot 자가검증 + POMDP 라우터 | 경량 정책 모델 |

---

## 10. 조사 방법 메모

- arXiv/OpenReview/HuggingFace Papers/GitHub 검색을 통해 1차로 논문을 수집하고, 가능한 경우 원문 초록·본문 인용으로 라우팅 기법과 데이터셋 사용 여부를 교차 확인했다.
- RouterBench/RouterArena 사용 여부는 검색 스니펫만으로 확정하기 어려운 경우 `(확인 필요)`로 표시했다. 정밀한 인용 검증이 필요하면 각 논문 PDF의 "Experiments/Datasets" 절을 직접 대조해야 한다.
- 이 문서는 스냅샷이며, 2026-08-17 기준 검색 결과를 반영한다. 이후 새 논문(특히 ICLR/ACL/NeurIPS 2026 사이클)이 계속 나올 수 있으므로 주기적 갱신이 필요하다.

### 검증 로그 (2026-08-17)

최초 작성 시 시간 제약으로 표에서 제외했던 논문 9건을 각 arXiv abs/html/pdf 원문을 직접 대조해 검증하고 4/5/6/7장에 편입했다.

| 논문 | 판정 결과 |
| --- | --- |
| A Unified Approach to Routing and Cascading for LLMs | RouterBench 사용 확인 → 4·6장 |
| Federate the Router | RouterBench 기반 데이터 사용 확인 → 4·6장 |
| VDAR-Router | RouterBench+LLMRouterBench 사용 확인 → 4·6장 |
| OrcaRouter | RouterArena 리더보드 등재 확인 → 4·6장 |
| Routing, Cascades, and User Choice for LLMs | 이론 전용, 데이터셋 미사용 확인 → 5·6장 |
| Cluster, Route, Escalate | 자체 데이터, RB/RA 언급 없음 확인 → 5·6장 |
| UCCI | 실서비스 워크로드 사용, RB/RA 미사용 확인 → 5·6장 |
| Towards Fair and Comprehensive Evaluation of Routers | 자체 RouterXBench 제안, RB/RA 미사용 확인 → 5·6장 |
| The Routing Plateau | RouterBench 포함 5개 벤치마크 비교, 단일 기법이 아닌 메타분석이라 6장 표 대신 7장에 배치 |
