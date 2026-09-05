# LLM 라우팅 모델 스펙·기능·성능 조사 보고서

기준일: 2026-09-05
관련 문서: `docs/ml/llm-routing-literature-review.md`(논문 목록·분류),
`docs/ml/llm-routing-benchmark-dataset-analysis.md`(벤치마크 데이터셋 분석),
`docs/ml/integrated-routing-data-and-model-comparison-plan.md`(최종 비교 모델군·데이터·평가 계약)

> 2026-09-16 실행 경계: 이 문서에서 비교 대상으로 선정한 여러 모델과 ModernBERT
> 운영 초기학습은 Colab에서 실행한다. GCP KFP/Vertex GPU 학습은 ModernBERT feedback
> 재학습으로 한정하며 비교 결과를 다른 family의 자동 배포로 연결하지 않는다.

## 0. 문서 성격

`llm-routing-literature-review.md`에서 정리한 30여 편의 라우팅 논문과 자체 채택 모델(ModernBERT)에 대해, 각 라우팅 모델의 **아키텍처 스펙 · 핵심 기능 · 정량적 성능 · 라우터 학습/검증/평가 데이터의 출처와 규모**를 arXiv 원문(abs/html) 대조로 조사했다. RouterBench/RouterArena/RouterEval/LLMRouterBench 4개 벤치마크 자체의 구성·라벨링 방법은 이 문서가 아니라 `llm-routing-benchmark-dataset-analysis.md`에서 다룬다.

신뢰도 표기: 각 항목은 arXiv abs 또는 html 원문에서 직접 확인한 수치만 기재했다. 확인하지 못한 항목은 "원문에서 확인 불가"로 표시했다. 2026년 발표 논문 일부는 매우 최근이라 html 렌더링이 불완전해 일부 세부 항목이 누락될 수 있다.

---

## 1. 자체 채택 모델 — ModernBERT (LLM_Router 프로젝트)

다른 논문들과 달리, 이 프로젝트가 실제로 학습·배포하는 모델이므로 스펙과 실측 성능을 상세히 기록한다. 출처: `docs/ml/modernbert-training-spec.md`, `docs/ml/training-reports/modernbert-v2-balanced-training-report.md`.

### 1-1. 모델 스펙

| 항목 | 값 |
| --- | --- |
| base model | `answerdotai/ModernBERT-base` (149M 파라미터, revision `8949b909…f0c8`) |
| 태스크 | 단일 라벨 3-클래스 시퀀스 분류 (easy=0, medium=1, hard=2) |
| 최대 컨텍스트 | 8,192 토큰 (공식 모델카드 기준, smoke는 512로 제한) |
| precision / device | FP16 / CUDA T4 |
| train batch / gradient accumulation | 1 / 8 (effective batch 8) |
| learning rate / warmup / weight decay | 2e-05 / 703 step(10%) / 0.01 |
| gradient checkpointing | true |
| 학습 데이터(v2 balanced) | train 18,768행(클래스당 6,256) / validation·test 각 3,218행(자연분포) |
| 학습 인프라 | Google Colab T4 GPU, 3 epoch·7,038 optimizer step, 약 5시간 48분 |

### 1-2. 핵심 기능

- 질의 텍스트를 입력받아 easy/medium/hard 3단계 난이도로 분류해 LLM 라우팅 tier를 결정.
- RouterBench 11개 모델의 tier별 성공률을 "최소 충분 tier" 개념으로 라벨링(`minimum-sufficient-tier-v2`).
- v2에서는 클래스 불균형을 해소하기 위해 optimizer 단위(micro-batch×accumulation)마다 클래스 구성을 3/3/2로 순환시키는 balanced sampling 사용.

### 1-3. 실측 성능 (v2 balanced 학습 결과, 2026-08-15 보고서 기준)

| 지표 | 값 | 비교 기준 |
| --- | --- | --- |
| RouterBench validation accuracy (최종 epoch 3) | 45.15% | majority-class baseline 46.3% — **기준선을 못 넘음** |
| RouterBench validation accuracy (최고점, epoch 2) | 46.86% | baseline과 사실상 동률 |
| RouterBench validation macro F1 (최종) | 0.4352 | epoch 2 대비 −0.0153 (epoch 3 과적합) |
| RouterArena transfer accuracy (최고점, epoch 2) | 35.82% | 균등추측(33.3%)보다 근소 우위 |
| RouterArena transfer macro F1 (epoch 3) | 0.3212 | 균등추측 기대값 0.326과 사실상 동일 |
| TF-IDF+LogisticRegression baseline accuracy | 43.44% | ModernBERT 최종(45.15%)과 macro F1 차이 0.03에 불과 |
| easy/hard/medium 이진분리 AUC | 0.63 / 0.615 / 0.589 | 0.5=신호 없음 기준 — 약한 신호 |

### 1-4. 주요 진단 결과 (한계)

1. **성능 상한은 모델이 아니라 라벨에 있다** — 149M 파라미터 트랜스포머가 bag-of-words(TF-IDF) 대비 macro F1 +0.03만 개선. 텍스트 자체의 신호 한계로 판단.
2. **모델이 실제로 학습한 것은 출처 문체다** — 출처 내부(within-dataset) 상관 r=0.114로, 같은 데이터셋 안에서는 난이도를 거의 구분하지 못함. 예측 분산의 32.1%가 출처(source dataset)로 설명됨. "숫자·수식이 있으면 hard", "평범한 산문이면 easy" 같은 표층 휴리스틱을 학습.
3. **전역 클래스 균형화가 오히려 math 도메인의 easy 신호를 인위적으로 훼손** — 자연분포 대비 균형화 후 math easy recall이 0.327→0.084로 급락. 균형화 제거(자연분포 전량 학습)가 accuracy·macro F1·kappa 4개 지표 모두에서 우월한 것으로 probe 실험(TF-IDF 대체모델) 확인.
4. **medium 클래스가 구조적 잔여 구간** — 어떤 threshold 정의에서도 F1 0.15~0.33으로 easy/hard의 절반 수준.

### 1-5. 권고 (보고서 원문 기준)

1. epoch 2 checkpoint로 교체하거나 best-checkpoint 선택 도입 (재학습 없이 즉시 macro F1 +0.015 회수).
2. `balanced_train.enabled`를 끄고 자연분포 전량(25,724행)으로 재학습.
3. 라벨 정의를 solve-rate 기반으로 바꾸는 실험(probe kappa +31% 개선, 데이터 손실 없음).
4. 형식논리·코드 도메인은 실제 데이터 보강 검토(단, RouterBench와 동일한 라벨 의미를 가진 데이터만 병합 가능 — 외부 데이터셋의 자체 난이도 라벨 직접 병합은 금지, RouterBench의 tier 정의 모델 다수가 서비스 종료되어 재현 불가).

---

## 2. 라우팅 논문별 모델 스펙·기능·성능

발표순으로 정리했다. 표기 형식: **스펙** → **핵심 기능** → **성능** → **평가 데이터셋**.

### 2-1. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing (2404.14618, ICLR 2024)

- **스펙**: DeBERTa-v3-large(약 300M). 결정론적 라우터(BCE), 확률적 라우터(모델당 10개 응답 샘플 소프트라벨), 데이터 변환 라우터(불균형 완화) 3단계. 입력 쿼리 텍스트 → 출력 [0,1] 확률.
- **기능**: 품질 간격 H(x)=q(S(x))−q(L(x))로 쉬운 쿼리를 소형 모델로 라우팅.
- **성능**: 최대 40% 비용 절감(품질 무손실). 모델 쌍별 20~40% 절감 시 품질 하락 0~10.3%. 라우터 지연 0.036초(FLAN-T5-800m 대비 약 13배 빠름).
- **평가**: MixInstruct(20K), FLAN-T5(800m/11b)·Llama-2(7b/13b)·GPT-3.5-turbo 쌍, BART Score.

### 2-2. RouteLLM: Learning to Route LLMs with Preference Data (2406.18665, ICLR 2025)

- **스펙**: SW Ranking(Bradley-Terry+임베딩), Matrix Factorization, BERT Classifier(BERT-base), Causal LLM Classifier(Llama-3-8B) 4종. 출력: 강모델 승률 P(wins|q).
- **기능**: Chatbot Arena 선호 데이터 + GPT-4 판정 증강으로 승패 예측 후 비용 임계값 α로 라우팅.
- **성능**: MT-Bench에서 Matrix Factorization이 Random 대비 APGR +60.4%, 비용 최대 3.66배 절감(95% GPT-4 품질). MMLU 1.41배, GSM8K 1.49배 절감. 모델 교체 시 재학습 없이 APGR +56.6%/+49.8% 유지. 라우터 오버헤드는 GPT-4 생성비용의 0.4% 이하.
- **평가**: Chatbot Arena(80K), MMLU(14,042), MT-Bench(160), GSM8K(1,000+).

### 2-3. MetaLLM (2407.10834)

- **스펙**: 선형 리워드 모델 Q_j(x;θ_j)=xᵀθ_j (Sentence-BERT 임베딩), UCB 멀티암드 밴딧, Ridge regression 초기화 후 온라인 갱신.
- **기능**: 리워드 r(x,i)=a_i(x)−p·c_i를 UCB로 최적화해 쿼리별 최적 LLM 온라인 선택.
- **성능**: SST-2에서 비용 약 58% 절감하며 정확도 91.51%→92.55%. MMLU(Together AI)에서 비용 10% 절감, 정확도 80.24%→80.90%.
- **평가**: SST-2(67,439/872), MMLU(14,042). API: OpenAI, Together AI.

### 2-4. Eagle: Efficient Training-Free Router for Multi-LLM Inference (2409.15518, NeurIPS 2024 workshop)

- **스펙**: 학습 없는(training-free) ELO 레이팅. Eagle-Global(전체 성능)+Eagle-Local(쿼리 유사 이웃 성능, N=20, K=32)을 P=0.5로 가중합. 임베딩 stella_en_1.5B_v5.
- **기능**: 쌍대 비교만으로 ELO 순위 구성, 재학습 없이 실시간 확장.
- **성능**: AUC 기준 SVM 대비 +23.52%, KNN 대비 +5.14%, MLP 대비 +4.73%. 구성 시간 baseline의 4.8%(8.0초 vs MLP 248.3초), 증분 업데이트는 100~200배 빠름.
- **평가**: **RouterBench 전체 7개 태스크(MMLU/MT-Bench/MBPP/HellaSwag/Winogrande/GSM8K/ARC-C)**.

### 2-5. RouterDC: Query-Based Router by Dual Contrastive Learning (2409.19886, NeurIPS 2024)

- **스펙**: mDeBERTaV3-base(86M) + LLM별 768차원 임베딩(전체 100M 미만). 이중 대조학습(Sample-LLM K+=3/K-=3, Sample-Sample 5클러스터), AdamW lr 5e-5.
- **기능**: 기존 ZOOTER의 softmax 정규화 약화 문제 해결을 위해 대조학습+유사 쿼리 클러스터링 결합.
- **성능**: In-distribution 평균 58.54%(최고 개별 LLM 대비 +3.98%p), OOD 평균 45.84%(+1.90%p). ZOOTER(ID 53.97%/OOD 40.16%) 대비 우위. Voting 대비 추론 6배 빠름.
- **평가**: ID 5개(MMLU, GSM8K, CMMLU, ARC-C, HumanEval), OOD 3개(PreAlgebra, MBPP, C-EVAL), LLM 7개.

### 2-6. GraphRouter: A Graph-based Router for LLM Selections (2410.03834, ICLR 2025)

- **스펙**: Heterogeneous GNN(2-layer GAT, hidden 32). Task/Query/LLM 3종 노드(BERT 임베딩), 엣지 예측 문제로 학습.
- **기능**: 태스크·쿼리·LLM 상호작용을 그래프로 표현하는 유도적(inductive) 학습, 신규 LLM에 재학습 없이 few-shot 대응.
- **성능**: FrugalGPT 대비 Reward 최소 12.28% 향상, Oracle 대비 최소 88.89% 달성. 신규 LLM few-shot 대응 시 C2MAB-V 대비 Reward +9.52%, 시간비용 99.45% 감소(3.00초 vs 542.25초).
- **평가**: 주 실험은 Alpaca/GSM8K/SQuAD/Multi-News 각 600건. 확장 실험은 여기에 HumanEval/HotpotQA를 **각 600건씩 표본화**해 추가(원 데이터셋 전체 크기 164/113K와 실험 사용량은 다름). LLM 10개.

### 2-7. EmbedLLM: Learning Compact Representations of Large Language Models (2410.02223)

- **스펙**: Matrix Factorization. 모델 임베딩 φ_m + 질문 임베딩(all-mpnet-base-v2) 하다마르 곱 기반 분류기, BCE 손실.
- **기능**: LLM별 task-specific 표현 반복 학습 비효율 해소 위해 단일 compact 임베딩으로 정확성 예측·라우팅·벤치마크 예측 동시 지원.
- **성능**: 정확성 예측 MF 0.7409 vs KNN 0.7152. 라우팅 속도 초당 750개(Causal LLM 라우터 대비 15배). GPU 메모리 1GB 미만(파인튜닝 대비 60배 저렴). MMLU 14,000문항 예측으로 약 200만 입력 토큰 절감.
- **평가**: 112개 오픈소스 모델 × 36,054문항(MMLU, TruthfulQA 등 10개 데이터셋).

### 2-8. A Unified Approach to Routing and Cascading for LLMs (2410.10347)

- **스펙**: 새 아키텍처 제안이 아닌 통합 최적화 프레임워크. 구체적 신경망 구조는 원문에서 확인 불가.
- **기능**: 라우팅(단일모델 선택)+캐스케이딩(순차 시퀀스)을 "비용 예산 제약 하 기대 품질 최대화"로 통합.
- **성능**: **RouterBench**(11모델)에서 캐스케이드 라우팅 AUC 87.24 vs 기존 캐스케이드 84.48 vs 라우팅 단독 83.25. SWE-Bench에서 54.12 vs 53.20(최대 14% 향상).
- **평가**: **RouterBench(주 실험)**, SWE-Bench, Minerva Math+LiveCodeBench, ARC-Challenge/MMLU-Pro/MixEval, GSM8k.

### 2-9. PickLLM: Context-Aware RL-Assisted Large Language Model Routing (2412.12170, AAAI 2025 workshop)

- **스펙**: 무상태 RL 2종 — LRI 학습 오토마톤(β=0.5), ε-greedy Q-learning(θ=0.7). Reward=[w_a·정확도−w_c·비용]/[w_l·log10(지연)].
- **기능**: 비용·지연·정확도를 가중 보상함수로 통합한 경량 RL로 세션 내 최적 LLM에 실시간 수렴.
- **성능**: WikiQA 500쿼리에서 SLA 방식 최대비용 대비 60% 절감, 지연 무작위 대비 52% 감소. HC3에서 LLM-Blender 대비 크게 우세(Open QA 0.95 vs 0.51).
- **평가**: HC3(Reddit ELI5, Open QA, Wiki CSAI, Medical, Finance 각 500쿼리).

### 2-10. MixLLM: Dynamic Routing in Mixed Large Language Models (2502.18482)

- **스펙**: BERT 쿼리 임베딩(+InsTag 20개 도메인) + LLM별 경량 회귀(Random Forest/MLP) + 메타 결정자(품질-비용 λ, 불확실성, 대기시간). 오프라인 경사하강+온라인 정책그래디언트.
- **기능**: 컨텍스트 밴딧으로 품질·비용·대기시간 동시 고려, 온라인 피드백으로 지속 학습.
- **성능**: GPT-4 품질의 97.25%를 비용 24.18%로 달성(λ=1.4). 최고 baseline OptLLM(96.39%/32.94%) 대비 우위. 온라인 학습으로 미세피드백 69.74%→71.29%.
- **평가**: **RouterBench 확장판**(36,497쿼리, 8개 데이터셋, 11+2개 LLM).

### 2-11. Causal LLM Routing: End-to-End Regret Minimization from Observational Data (2505.16037)

- **스펙**: 2-layer feedforward NN(GELU, hidden 200)+softmax. BERT(768) 또는 Llama-3.2-1B(2048) 임베딩. Doubly Robust 추정기(결과모델 NN+성향점수 XGBoost)로 인과 편향 보정.
- **기능**: 관측 데이터의 처리 편향을 인과추론으로 보정하며 예측-라우팅 결정을 분리하지 않고 end-to-end로 regret 직접 최소화.
- **성능**: SPROUT(BERT)에서 RM-Softmax가 λ=100일 때 Baseline 대비 64.62→77.53(+19.7%). **RouterBench**에서 λ=500일 때 53.85→56.48(+4.9%). AUC(SPROUT-BERT) R&C 대비 +152%.
- **평가**: **RouterBench**(35,712쌍, 11 LLM, 8벤치마크), SPROUT(44,241프롬프트, 13 LLM).

### 2-12. IRT-Router (2506.01048, ACL 2025)

- **스펙**: MIRT-Router(다차원 IRT)/NIRT-Router(신경망 NCDM). bert-base-uncased(768) 임베딩, 능력 차원 N=25. P̂=σ(−aᵀθ+b). 24,430쿼리×20LLM=488,600 상호작용 학습.
- **기능**: IRT로 쿼리 난이도·LLM 능력 명시적 모델링, 의미론적 유사도 기반 콜드스타트 워밍업.
- **성능**: ID(α=0.8,β=0.2)에서 정확도 80.67%/비용 $0.42 (RouterBench 자체 라우터 80.01%/$1.15, Large LLM 단독 77.53%/$12.93 대비). OOD 87.12%/$0.14.
- **평가**: LLM 20개, 데이터셋 12개(ID 8개: MMLU, CMMLU, ACLUE 등; OOD 4개). RouterBench 미사용(자체 데이터).

### 2-13. Arch-Router: Aligning LLM Routing with Human Preferences (2506.16655)

- **스펙**: **Qwen2.5-1.5B** 생성형(분류기 아님) SFT. 43,000샘플. 입력 쿼리+후보 라우트 정책 → 출력 라우트 식별자 생성.
- **기능**: 사용자 정의 도메인/액션에 쿼리 매핑("preference-aligned routing"), 신규 모델 추가 시 재학습 불필요.
- **성능**: 종합 정확도 93.17%로 Claude-sonnet-3.7(92.79%), GPT-4o(89.74%) 상회. 레이턴시 51±12ms(Claude-sonnet-3.7의 1450±385ms 대비 약 28배 빠름).
- **평가**: CLINC-150, MANtIS, SGD, LMSYS-1M.

### 2-14. BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute (2506.22716, ICML 2025)

- **스펙**: DeBERTa-v3-small(44M) + K×N 경량 분류 헤드. 라벨은 best-of-n 응답의 참조모델 동등 여부(armoRM 기반). BCE+pairwise ranking loss.
- **기능**: 쿼리별 모델과 sampling 횟수(best-of-n)를 동시 결정.
- **성능**: 60% 비용절감 시 품질저하 0.80%(N-label baseline 5.08% 대비 4.28%p 개선). Model Cascades 대비도 0.80% vs 7.26%.
- **평가**: MixInstruct(6,000), RewardBench+CodeUltraFeedback/BeaverTails 등 총 10,000건, LLM 8개. **RouterBench 사용 여부는 확인 불가(2차 자료 기준으로 표기됐던 항목, 이번 조사에서 직접 확인 못함)**.

### 2-15. CP-Router: An Uncertainty-Aware Router Between LLM and LRM (2505.19970, AAAI 2026)

- **스펙**: 학습 불필요(training-free), 모델 무관. Conformal Prediction 불일치 점수 S=1−f(y). 예측집합 크기 1이면 LLM, 아니면 LRM. FBE(엔트로피 기준)로 오류율 α* 자동 선택.
- **기능**: 불확실성 추정으로 커버리지 보장을 갖춘 LLM↔LRM 동적 라우팅.
- **성능**: Llama 페어링에서 Elementary Math 정확도 78.2%/TRR 14.2%, GPQA 63.3%/TRR 6.4%. 6개 벤치마크 중 Llama 6/6, Qwen 5/6에서 최고 토큰 효율.
- **평가**: MMLU-STEM, STEM-MCQA, GSM8K, GPQA, LogiQA, CN-Chemistry.

### 2-16. RADAR: Reasoning-Ability and Difficulty-Aware Routing (2509.25426, ICLR 2026)

- **스펙**: 2-Parameter Logistic IRT, p_ij=σ(aⱼ(θᵢ−bⱼ)). 50,139개 고유 질문, 175만 개 이진 응답 학습. Fisher 정보량 기반 적응형 테스팅으로 신규 모델 5k 쿼리(12%)만으로 능력값 추정.
- **기능**: 다양한 추론 budget에서의 응답으로 item response model을 학습해 난이도 높은 쿼리를 능력 높은 모델-budget 조합으로 라우팅.
- **성능**: 하이퍼볼륨(GPQA-Diamond) RADAR 0.7513 vs IRT-Router 0.6942. CPT(90%) GPQA-Diamond 13.21%(IRT-Router 53.99%).
- **평가**: AIME, MATH, GPQA-Diamond, LSAT, MMLU(-Redux/-Pro), FRAMES. o4-mini+Qwen3 계열 35개 모델-budget 설정.

### 2-17. LLM Routing with Dueling Feedback (2510.00841)

- **스펙**: Contrastive Fine-Tuning(CCFT, 오픈소스 임베딩 all-MiniLM-L6-v2 등 튜닝) + FGTS.CDB(Bradley-Terry-Luce 기반 contextual dueling bandit, 후회 상한 Õ(d√T)).
- **기능**: 절대 점수 대신 쌍별(pairwise) 선호 피드백만으로 학습하는 dueling bandit 라우팅, 샘플 효율적(카테고리당 5개).
- **성능**: fine-tuned 임베딩이 OpenAI 임베딩 baseline 대비 낮은 누적후회·빠른 수렴(그래프 형태, 구체 표 수치는 확인 불가). MixInstruct에서 최고모델 선택률 21.22%.
- **평가**: **RouterBench**(7개 벤치마크), MixInstruct(110K, 11 LLM).

### 2-18. LLMRank: Understanding LLM Strengths for Model Routing (2510.01234)

- **스펙**: 신경망 랭킹 모델(은닉층 [512,256], ReLU+dropout 0.1) + 텍스트 인코더. MSE+KL divergence(τ=0.5) 손실. U=Q−λC(λ=0/10³/10⁵).
- **기능**: 해석 가능한 특징(작업유형, 추론패턴, 복잡도 등) 기반 프롬프트 인식 라우팅.
- **성능**: LLMRank-Perf 품질 0.843/비용 $5.784(89.2% oracle utility). GPT-4(0.812/$17.237, 85.9%) 상회. Balanced(λ=10³)는 GPT-4 대비 1.8% 낮은 품질에 12배 저비용.
- **평가**: **RouterBench zero-shot**(36,497 프롬프트, 11벤치마크, 11 LLM).

### 2-19. ICL-Router: In-Context Learned Model Representations for LLM Routing (2510.09719)

- **스펙**: Projector(2층 MLP) + LLM 기반 Router(**Qwen2.5-7B-Instruct**). 임베딩 Qwen3-Embedding-8B(4096차원, 고정). 2단계 학습(Query Reconstruction+ICL Model Routing).
- **기능**: in-context vector로 모델 역량 표현, 신규 모델 추가 시 재학습 없이 프로파일만 추가.
- **성능**: ID 평균 76.30%(RouterDC 72.32%, EmbedLLM 74.16% 대비 우위), OOD 평균 66.47%. In-context 예제 500개가 최적점.
- **평가**: ID 5개(OlympiadBench, BBH, LogicBench 등), OOD 5개(AIME, KORBench 등), 후보 모델 8개.

### 2-20. Federate the Router (2601.22318)

- **스펙**: Parametric MLP-Router(공유 trunk+모델별 헤드) 또는 Nonparametric K-Means-Router(Voronoi 셀 가중평균). 연합학습(N=10 클라이언트, 참여율 60%, 로컬 SGD+서버 가중평균).
- **기능**: 클라이언트별 희소·불균일 평가 데이터 상황에서 연합학습으로 공유 라우팅 정책 학습, 신규 모델은 헤드만 재학습.
- **성능**: "연합학습이 로컬 라우터를 능가"(구체 수치는 그래프 형태로만 제시, 원문에서 표 수치 확인 불가). 저자들이 "첫 연합 라우터 연구"로 명시, 직접 비교 가능한 기존 baseline 없음.
- **평가**: **RouterBench-Data**(11 LLM, 8개 데이터셋, Dirichlet 분할), ProxRouter-Data(14 LLM, 10개 데이터셋).

### 2-21. Routing, Cascades, and User Choice for LLMs (2602.09902)

- **스펙**: 학습 모델 없음. Stackelberg 게임 이론 분석(공급자 리더/사용자 팔로어).
- **기능**: 라우팅+캐스케이딩+사용자 포기 선택을 하나의 게임으로 통합 분석, 정렬/미정렬 조건과 "스로틀링"(지연을 늘려 비용 절감) 유인 증명.
- **성능**: 순수 이론 분석, 수치 실험 없음. 주요 정리: 대부분 최적 정책은 정적(에스컬레이션 없음), 포기 페널티가 작을수록 공급자의 지연 증가 유인 커짐.
- **평가**: 실제 데이터셋 없음(이론적 파라미터 가정).

### 2-22. Towards Fair and Comprehensive Evaluation of Routers — ProbeDirichlet/RouterXBench (2602.11877)

- **스펙**: 각 레이어 hidden state mean pooling → 학습 시 Dirichlet(β) 레이어 가중치 샘플링, 추론 시 결정론적 기댓값 집계. 경량 선형 분류기(입력 4096, Llama-3.1-8B hidden state).
- **기능**: 외부 임베딩 대신 LLM 내부 hidden state 활용, Dirichlet 확률적 레이어 집계로 특정 레이어 과의존 방지. Router Ability/Scenario Alignment/Cross-Domain Robustness 3축 평가 프레임워크(RouterXBench) 제안.
- **성능**: AUROC 도메인내 68.70%/도메인외 65.46%(EmbeddingMLP 59.46/55.22 대비 최대 16.68% 상대개선). Dirichlet 집계(68.70) > Mean Pool(68.04) > Final Layer(53.91).
- **평가**: ID(Alpaca/MMLU/Big-Math), OOD(Magpie/MMLU-Pro/MATH). 자체 제안 RouterXBench, RouterBench/RouterArena 미사용.

### 2-23. LLM Router: Rethinking Routing with Prefill Activations (2603.20895)

- **스펙**: "SharedTrunkNet" — 여러 인코더의 prefill 활성화(PCA 축소) 연결(concat) 입력 joint multi-output MLP. Encoder-Target Decoupling. Fisher Separability로 상위 절반 층 선택, Qwen3.5-122B 인코더가 최고 AUC. 정확한 파라미터 수는 원문에서 확인 불가.
- **기능**: 공개 가중치 인코더의 내부 활성화만으로 폐쇄형 타깃 모델 성능을 예측·라우팅.
- **성능**: Frontier Pool 기준 Mean AUC 0.8560(Unified Multitask 0.8040, Matrix Factorization 0.7943 대비). 최고가 모델 대비 비용 74.3% 절감.
- **평가**: MMLU-Pro, Humanity's Last Exam(HLE), LiveCodeBench.

### 2-24. vLLM Semantic Router / When to Reason (2603.04444, 2510.08731)

- **스펙**: 13종 신호(휴리스틱+학습형) 병렬 평가 → 부울 의사결정 정책. 분류기는 **ModernBERT 32K 확장 컨텍스트 기반** + 태스크별 LoRA 어댑터(도메인/복잡도/사실성/모달리티 등).
- **기능**: 정보이론(엔트로피 감소)+부울대수 결합한 "신호-의사결정 분해". HaluGate(3단계 할루시네이션 탐지), 신경-기호 DSL.
- **성능**: 신호 추출 지연 휴리스틱 <1ms, 학습형 10-120ms. 관련 논문(When to Reason)은 MMLU-Pro에서 정확도 +10.2%p, 지연 47.1%↓, 토큰 48.5%↓.
- **평가**: MMLU 카테고리 기반, 구체적 엔드투엔드 라우팅 정확도 수치는 원문에서 확인 불가.

### 2-25. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing (2605.18796)

- **스펙**: 별도 신경망 라우터 아님. 4B 모델의 토큰 단위 마진 불확실성을 isotonic regression으로 캘리브레이션(ECE O(n^-1/3) 보장).
- **기능**: 쉬운 쿼리는 4B 모델, 캘리브레이션된 불확실성이 임계값 초과 시 12B로 에스컬레이션. 비용 제약 하 최적 임계값 선택.
- **성능**: micro-F1=0.91 유지하며 비용 31% 절감(95% CI [27%,35%]). ECE 0.12→0.03 개선.
- **평가**: 실서비스 워크로드(75,000건 개체명 인식, 4B/12B 모델, H100 서빙 로그).

### 2-26. OrcaRouter: A Production-Oriented LLM Router (2605.30736)

- **스펙**: LinUCB 컨텍스추얼 밴딧. 424차원 문맥(40차원 수공예+384차원 all-MiniLM-L6-v2 임베딩). 오프라인 워밍업(RouterBench 5,000프롬프트×10모델, 릿지회귀)+온라인 부분정보 학습.
- **기능**: 오프라인 전체정보 워밍업 후 온라인 지속 적응하는 하이브리드 학습, 프로덕션 비용/지연/오류패널티 보상 설계.
- **성능**: **RouterArena** 리더보드 기준 Arena Score 72.08(2위), 정확도 75.54%, 비용 $1.00/1,000쿼리(baseline Always-DSC 70.31/70.27%/$0.10). Oracle 상한 80.72.
- **평가**: **RouterArena(8,400)**, RouterBench(5,000, 중복제거), 모델 풀 10개.

### 2-27. Cluster, Route, Escalate (2606.27457)

- **스펙**: 2단계 — Stage1: all-MiniLM-L6-v2 임베딩 k-means 클러스터링(실루엣 점수로 k 선택), 클러스터별 비용효율 모델 매핑. Stage2: **ModernBERT-base** 이진분류(Accept/Escalate) 파인튜닝 QE 캐스케이드.
- **기능**: 클러스터링으로 비용효율 모델 사전 할당, 저품질 응답만 선택적으로 강한 모델에 에스컬레이션.
- **성능**: 최강 단일모델 대비 97-99% 정확도 유지. AIME 2024(λ=0.06): 88.4% vs baseline 89.1%(정확도 0.7%p 손실, 지연 18% 절감).
- **평가**: AIME(2024 테스트 30개), TeleQnA(9,000/1,000).

### 2-28. The Routing Plateau (2606.07587, 메타분석)

- **스펙**: 새 라우터 아키텍처 제안 아님. 21개 기존 방법을 5개 벤치마크(RouterBench, R2-Bench, EmbedLLM, CARROT/SPROUT, 자체 Nine-by-30k)로 평가한 메타분석.
- **기능**: "라우팅 플래토" 현상 규명 — 방법론이 상이해도 정확도가 좁은 범위로 수렴(상위 5개 spread 평균 0.22%p)하며 오라클 대비 10-30%p 격차 지속. 원인은 "예측 가능성 병목"(글로벌 평균 트렌드만 학습, 인스턴스별 신호 미포착).
- **성능**: 개선 실험(data scaling 30k→300k, ModernBERT-large 인코더 확장, 엔드투엔드 파인튜닝)으로 baseline 68.14%→69.38%(+1.24%p), 오라클 격차의 8.5% 해소.
- **평가**: **RouterBench 포함 5개 벤치마크**.

### 2-29. VDAR-Router (2607.18098)

- **스펙**: Training-free. 난이도 분석 생성 LLM(Gemma4-31B/Qwen3.5-4B) + Qwen3-Embedding-0.6B 검색. 리워드 R(m|q)=(1/k)Σ[α·p_m−β·c_m].
- **기능**: 추가 학습 없이 쿼리의 언어화(verbalized) 난이도 분석을 검색 키로 사용해 유사 난이도 과거 사례의 모델 성능 통계로 최적 모델 선택.
- **성능**: **RouterBench**(α=0.8/β=0.2) 리워드 48.61(ICL-Router 42.15, RouteLLM 34.34 대비 최고). **LLMRouterBench** 리워드 42.58(RouteLLM 41.11 대비 우위). ArenaExpert5K Pairwise Accuracy 60.28%(IRT-Router 53.22%).
- **평가**: **RouterBench(10,000), LLMRouterBench(12,732)**, ArenaExpert5K(2,678).

---

## 3. 논문별 학습·검증·평가 데이터 출처와 규모

### 3-1. 읽는 법과 집계 원칙

- 여기서 **학습**은 기반 LLM의 사전학습이 아니라, 논문이 제안한 라우터·점수모델·검색 DB·밴딧 초기상태를 만드는 데 쓴 데이터를 뜻한다.
- 기본 단위는 고유 **질의(query/instruction)** 수다. 논문이 `(질의, 후보 모델)` 쌍을 세면 **상호작용**으로 따로 표시했다. 예를 들어 24,430질의×20모델은 488,600상호작용이다.
- `≈`는 논문이 전체 건수와 비율만 제시해 계산한 값이다. 구현의 반올림 방식이 없으면 정확한 정수로 단정하지 않았다.
- `미기재`는 원문이 데이터 이름이나 분할 비율은 밝혔지만 절대 건수를 밝히지 않은 경우다. 벤치마크 원자료의 전체 크기를 해당 논문의 실제 사용량으로 대신 쓰지 않았다.
- training-free도 과거 사례, 보정 세트, 검색 DB가 필요할 수 있으므로 **파라미터 학습 없음**과 **데이터 불필요**를 구분했다.
- 각 행의 논문 제목 링크는 이번 보강에서 대조한 1차 출처(arXiv HTML)다.

조사 범위는 §2의 **29개 방법 전부**다. 이 중 28개 실증·시스템 연구는 사용 데이터와 공개된 규모를 아래에 기록했고, 실증 실험이 없는 순수 이론 연구 1개는 `없음`으로 구분했다.

### 3-2. 2024년 발표 방법

| 논문·라우터 | 학습·구성 데이터 | 검증·보정 데이터 | 최종 평가 데이터 | 단위·주의사항 |
| --- | --- | --- | --- | --- |
| [Hybrid LLM](https://arxiv.org/html/2404.14618) | MixInstruct train에서 균등 표본 10,000. 원천은 Alpaca-GPT4, Dolly-15K, GPT4All-LAION, ShareGPT이며 각 질의에 후보 LLM별 응답 10개를 생성 | MixInstruct validation 5,000; checkpoint 선택. 별도 임계값 실험은 이 중 무작위 500 | MixInstruct test 5,000 | 질의 기준 총 20K. 확률 라우터의 응답 생성량은 `10,000×후보모델 수×10회` |
| [RouteLLM](https://arxiv.org/html/2406.18665) | Chatbot Arena 약 80K 중 5K holdout, 16자 미만 제거 후 64모델의 **65K 선호 비교**. 증강: MMLU validation 약 1,500 또는 Nectar 기반 GPT-4 judge 약 120K | Chatbot Arena 5K로 모델 선택 | MMLU 14,042; MT-Bench 160; GSM8K 1,000+ | 학습 단위는 질의별 pairwise preference. 증강 종류에 따라 실제 train 규모가 달라짐 |
| [MetaLLM](https://arxiv.org/html/2407.10834) | SST-2 train 67,439 또는 MMLU auxiliary train 99,842로 ridge 초기화; 온라인 조건에서는 평가 스트림 피드백으로 갱신 | 비용 스케일 `p`를 validation에서 5회 탐색하나 validation 절대 건수는 미기재 | SST-2 test 872; MMLU test 14,042 | 오프라인·온라인·오프라인+온라인의 세 조건을 별도 평가 |
| [Eagle](https://arxiv.org/html/2409.15518) | RouterBench 7개 태스크의 historical feedback 중 70%; ELO/검색 DB 구성, 신경망 학습 없음. 온라인 모사는 70%→85%→100%로 15%씩 추가 | train과 validation을 합쳐 70%라고만 표기; 별도 validation 수 미기재 | RouterBench 나머지 30% | 논문 원문은 절대 질의 수를 적지 않아 비율만 확정 가능 |
| [RouterDC](https://arxiv.org/html/2409.19886) | GSM8K는 공식 train; MMLU·CMMLU·ARC-C·HumanEval은 각 데이터의 70%를 무작위 분할해 합침 | 별도 validation 미기재 | ID: GSM8K 공식 test, 나머지 30%. OOD: PreAlgebra·MBPP·C-EVAL | 절대 건수 미기재. 데이터 효율 ablation은 태스크당 30/100개 학습도 시험 |
| [GraphRouter](https://arxiv.org/html/2410.03834) | Alpaca·GSM8K·SQuAD·Multi-News 각 600, 총 2,400질의를 70% 사용 = 1,680질의; 각 질의를 10 LLM과 상호작용 | 10% = 240질의 | 20% = 480질의. 확장 실험은 HumanEval·HotpotQA도 원자료에서 각 600건 표본화 | 신규 LLM 실험의 보조 데이터는 train 질의 중 80건×신규 LLM 4개이며, 학습이 아니라 test-time few-shot으로만 사용 |
| [EmbedLLM](https://arxiv.org/html/2410.02223) | 10개 벤치마크의 **test split**에서 모은 36,054질의×112모델 correctness matrix를 질의 기준 80% 사용(≈28,843질의) | 10%(≈3,605) | 10%(≈3,606) | 원천: MMLU, TruthfulQA, SocialIQA, PIQA, MedMCQA, MathQA, LogiQA, GSM8K, GPQA, ASDiv. 비율만 명시되어 정수는 근삿값 |
| [Unified Routing & Cascading](https://arxiv.org/html/2410.10347) | RouterBench 주 실험은 별도 라우터 학습 대신 noisy quality/cost signal에 선형 추정기를 적합. 현실 벤치마크의 poor-estimator 조건은 각 데이터 50% 중 절반, 즉 전체의 25%로 품질·비용 추정기 학습 | RouterBench 5%(약 2,000)로 정책 hyperparameter 탐색. 현실 벤치마크 poor-estimator 조건은 전체의 25% validation | RouterBench 95%. 현실 벤치마크는 SWE-Bench, Minerva Math+LiveCodeBench, ARC-C/MMLU-Pro/MixEval, GSM8K의 50% test | GSM8K는 별도 공식 train을 쓰고 원 test 전체를 평가. 그 외 현실 벤치마크 절대 분할 건수는 미기재 |
| [PickLLM](https://arxiv.org/html/2412.12170) | 사전 offline train 없음. HC3 하위집합별 500질의 세션 안에서 LRI/Q-learning이 순차 피드백으로 갱신 | 별도 validation 없음 | Reddit ELI5·Open QA(WikiQA)·Wiki-CSAI·Medical·Finance 각각 500질의 세션 | 같은 온라인 스트림이 학습과 성능 측정 구간 역할을 하므로 일반적인 독립 train/test가 아님 |

### 3-3. 2025년 발표 방법

| 논문·라우터 | 학습·구성 데이터 | 검증·보정 데이터 | 최종 평가 데이터 | 단위·주의사항 |
| --- | --- | --- | --- | --- |
| [MixLLM](https://arxiv.org/html/2502.18482) | RouterBench 확장판 36,497질의(8데이터셋, 기존 11모델+Llama-3.1 8B/70B)의 80%(≈29,198) | 별도 validation 미기재 | 20%(≈7,299); OOD 실험은 train/test 도메인이 겹치지 않게 같은 80/20 | continual 실험은 위 train 부분을 다시 offline:online = 80:20, 50:50, 30:70으로 분할 |
| [Causal LLM Routing](https://arxiv.org/html/2505.16037) | RouterBench 35,712 또는 SPROUT 44,241의 80%. SPROUT 원천은 GPQA·MuSR·MMLU-Pro·MATH·OpenHermes·RAGBench | 10% | 10% | SPROUT은 ≈35,393/4,424/4,424. RouterBench에도 같은 비율을 prompt-level로 적용. 원문은 RouterBench를 “prompt-response pairs”라고 불러 고유 질의/응답쌍 단위가 다소 모호함 |
| [CP-Router](https://arxiv.org/html/2505.19970) | 파라미터 학습 없음. MCQA 정답과 LLM option logits로 conformal score를 계산 | calibration/test 절대 건수 미기재. 구현 설명상 FBE로 `α*`를 고를 때 **test set의 prediction-set 분포를 사용** | MMLU-STEM, STEM-MCQA, GSM8K, GPQA, LogiQA, CN-Chemistry; open-ended QA 추가 실험 | 원문은 “7 MCQA+1 QA”라고 쓰지만 주 결과 표는 6 MCQA를 보고하며, 분할·건수 공개가 불충분함 |
| [IRT-Router](https://arxiv.org/html/2506.01048) | ID 8종 train: ACLUE 1,400; ARC-C 1,400; CMMLU 7,000; HotpotQA 1,400; MATH 1,400; MBPP 630; MMLU 9,800; SQuAD 1,400 = **24,430질의×20 LLM = 488,600상호작용** | 별도 validation 미기재 | ID test 합계 10,470: 600/600/3,000/600/600/270/4,200/600. OOD: C-EVAL 1,000; CommonsenseQA 1,000; GSM8K 1,000; HumanEval 160 = 3,160 | 논문이 질의 수와 상호작용 수를 모두 명시한 모범 사례 |
| [Arch-Router](https://arxiv.org/html/2506.16655) | 자체 routing instruction 43,442 turns(5,194 conversations, 981 domains, 3,156 actions)의 90%(≈39,098) | 10%(≈4,344) | CLINC150 1,000 turns; MANtIS 1,000; SGD 5,141 turns(2,500 conv/2,000 spans); LMSYS-1M 821 turns(123 conv/662 spans) | train과 평가는 대화 turn/동작 route 단위. 지연 평가는 train에서 별도 2,000 prompts 표본 |
| [BEST-Route](https://arxiv.org/html/2506.22716) | 총 10,000 중 8,000. 원천: MixInstruct QA 6,000; coding 2,000(RewardBench 984+CodeUltraFeedback 1,016); safety 2,000(RewardBench 740+BeaverTails 1,260). 8 LLM×질의당 20응답 생성 | 1,000 | 동일 혼합 데이터 1,000; OOD MT-Bench(사용 건수 미기재) | 질의는 10K지만 생성 응답은 최대 `10K×8×20=1.6M` |
| [RADAR](https://arxiv.org/html/2509.25426) | 8벤치마크의 train split에서 35 model-budget 설정을 평가해 IRT 학습; 신규 설정은 train 질의의 12%(약 5K)로 능력 추정 | 별도 validation 미기재 | ID는 각 benchmark test; OOD는 목표 benchmark를 빼고 나머지 train으로 학습 후 목표 test 평가 | 원문은 총 **50,139 고유질의/1.75M 이진응답**이라 보고하지만 dataset 표의 합은 43,336(AIME 1,035; MATH 8,000; GPQA 448; LSAT 2,025; MMLU 13,937; Redux 5,298; Pro 12,032; FRAMES 561)으로 **6,803건 불일치**. 개별 train/test 절대치도 미기재 |
| [Dueling Feedback](https://arxiv.org/html/2510.00841) | RouterBench에서는 source category/benchmark 기반 query pair로 MiniLM/mpnet/e5-base를 contrastive fine-tune; 카테고리 대표 임베딩은 각 benchmark에서 5질의. MixInstruct는 110K(100K/5K/5K) metadata·선호 사용 | 독립 validation 크기 미기재 | RouterBench의 남은 질의를 무작위 online sequence로 평가; MixInstruct online 평가. OOD 모사는 300질의 뒤 420질의(ARC 120+기타 300) | MMLU 통제실험만 offline 50(5 topic×10), online test 595를 명시. 주 실험의 embedding fine-tune pair 수와 online stream 총길이는 미기재 |
| [LLMRank](https://arxiv.org/html/2510.01234) | RouterBench zero-shot 36,497에서 저자원(<50)·중국어·전 모델 실패 질의를 제거한 34,623의 70%(≈24,236) | 15%(≈5,193) | 15%(≈5,194) | benchmark category 기준 stratified split. 정수는 70/15/15 비율에서 계산한 근삿값 |
| [ICL-Router](https://arxiv.org/html/2510.09719) | ID 5종 총 5,228(OlympiadBench 674, BBH 1,080, LogicBench 1,000, MMLU-Pro 1,500, MBPP 974)의 70%(≈3,660). 2단계에는 이 train 안의 challenging 500 사용 | 별도 validation 미기재 | ID 30%(≈1,568). OOD test-only: AIME 60, KORBench 1,250, MMLU-CF 1,000, AGIEval 1,576, HumanEval 164 = **4,050** | 500 exemplars는 추가 데이터가 아니라 전체 train의 부분집합 |

### 3-4. 2026년 발표 방법

| 논문·라우터 | 학습·구성 데이터 | 검증·보정 데이터 | 최종 평가 데이터 | 단위·주의사항 |
| --- | --- | --- | --- | --- |
| [Federate the Router](https://arxiv.org/html/2601.22318) | RouterBench-Data(11 LLM×8 datasets) 또는 ProxRouter-Data(14 LLM×10 datasets)를 10 clients에 분배; client별 75% local train | 별도 validation 없음 | client별 25% local test의 합집합을 global test로 사용 | 원문은 전체 고유질의 절대 건수를 명시하지 않음. RouterBench-Data는 Dirichlet `α=0.45` 비IID 분할 |
| [Routing, Cascades, and User Choice](https://arxiv.org/html/2602.09902) | 없음 | 없음 | 없음 | 순수 Stackelberg 게임 이론 논문으로 실증 데이터·모델 학습 없음 |
| [ProbeDirichlet / RouterXBench](https://arxiv.org/html/2602.11877) | 총 12K: Alpaca 4K, BigMath 4K, MMLU 4K. 각 3.2K train+0.8K validation | 합계 2.4K(각 0.8K) | ID: Alpaca 1K, BigMath 1K, MMLU 10K. OOD: Magpie 10K, MATH 5K, MMLU-Pro 12K | scaling ablation: 단일 domain 1K/4K/8K, mixed 3K/12K/24K |
| [Prefill Activations](https://arxiv.org/html/2603.20895) | MMLU-Pro·LiveCodeBench·HLE로 만든 mixed pool 14,469질의/모델의 85%(≈12,299). 후보 target 9개 | 기본 실험은 없음; calibration 실험은 75/10/15이므로 ≈1,447 calibration | 15%(≈2,170). OOD: STEM train 4,722→humanities/social test 3,503; HLE 전체 제외 후 test 1,225 | Table 1의 mixed pool은 all-correct 7,596+all-fail 1,853+disagreement 5,020=14,469 |
| [vLLM Semantic Router](https://arxiv.org/html/2603.04444) / [When to Reason](https://arxiv.org/html/2510.08731) | When to Reason의 ModernBERT pipeline: MMLU-Pro 약 12K, Microsoft Presidio 약 50K token-level PII, jailbreak security datasets(건수 미기재) | train/validation 분할과 건수 미기재 | MMLU-Pro 14 domains; 평가 표본 수 미기재 | 시스템 논문은 13개 신호별 데이터가 달라 통합 train set이 없음. MMLU-Pro가 학습·평가 양쪽에 등장하지만 분리 방법이 설명되지 않음 |
| [UCCI](https://arxiv.org/html/2605.18796) | 별도 신경망 학습 없음. 비공개 production photography NER 75,000 중 calibration 22,500(30%)으로 isotonic map 적합 | validation 15,000(20%)으로 threshold 선택 | test 37,500(50%) | 수동 라벨 6 entity types. 원자료는 개인정보 때문에 비공개 |
| [OrcaRouter](https://arxiv.org/html/2605.30736) | RouterArena와 유사도 0.85 초과인 RouterBench 1,135건 제거 후 5,000질의×10모델 reward matrix로 ridge warm start | in-distribution full-information diagnostic는 held-out RouterArena fold로 tuning하나 크기 미기재 | RouterArena 전체 8,400질의 | 주 제출은 RouterBench warm start 후 RouterArena에서 online partial feedback; train/eval benchmark를 분리 |
| [Cluster, Route, Escalate](https://arxiv.org/html/2606.27457) | Stage 1: AIME 1983–2023 921, TeleQnA 9,000. Stage 2 ModernBERT QE: AIME 921×2모델×1회=1,842; TeleQnA 9,000×1모델×5회=45,000 examples | early stopping으로 best macro-F1를 고르지만 별도 validation 분할·건수 미기재 | AIME 2024 30; TeleQnA 1,000 | Stage 1의 단위는 질의, Stage 2는 `(질의, 모델출력)` example |
| [The Routing Plateau](https://arxiv.org/html/2606.07587) | Nine-by-30k 23,027 / Nine-by-300k 302,371 / RouterBench 21,898 / R2-Bench 18,580 / EmbedLLM 119,964 / CARROT-SPROUT 152,193 | 각각 3,000 / 3,000 / 3,649 / 3,097 / 14,897 / 35,528 | 각각 5,000 / 5,000 / 10,950 / 9,291 / 15,210 / 36,214 | 모두 **고유질의 수**. 기존 4벤치마크는 공개 원 분할 유지; 동일 split을 모든 방법에 고정 |
| [VDAR-Router](https://arxiv.org/html/2607.18098) | RouterBench 10,000, LLMRouterBench 12,732, ArenaExpert5K 2,678을 각각 80% 사용해 Difficulty Analysis DB/검색 pool 구성 | 별도 validation 없음 | 각 20% | 비율상 train/test는 각각 RB 8,000/2,000; LLMRB ≈10,186/2,546; ArenaExpert ≈2,142/536. 파라미터 학습은 없지만 reference DB가 필요 |

### 3-5. 이 프로젝트 실험에 직접 주는 시사점

1. **현재 18,768 train은 작은 편이지만 비정상적으로 작은 규모는 아니다.** Hybrid LLM은 10K, BEST-Route는 8K, ProbeDirichlet은 12K로 학습했고 RouterDC는 태스크당 30~100건에서도 성능 포화 가능성을 보고했다. 반대로 300K까지 늘린 Routing Plateau도 +1.24%p 개선에 그쳐, 단순 증량만으로 해결되지는 않는다.
2. **현재 가장 큰 차이는 데이터량보다 학습 목표다.** 이 프로젝트는 질문 텍스트→easy/medium/hard 한 라벨을 배우지만, 다수 논문은 `질의×모델 correctness/preference`를 직접 감독한다. 같은 24,430질의도 IRT-Router에서는 20모델과 결합되어 488,600개의 감독 신호가 된다.
3. **출처별 문체 누출을 막는 평가가 필요하다.** MixLLM·ICL-Router·ProbeDirichlet은 OOD domain을 분리하고, OrcaRouter는 benchmark 간 중복 1,135건을 제거했다. 따라서 다음 비교 실험은 random split뿐 아니라 source-held-out 또는 domain-held-out split을 함께 써야 한다.
4. **무학습 후보도 데이터 비용이 0은 아니다.** Eagle·VDAR·OrcaRouter는 모델 파인튜닝 대신 historical feedback DB나 warm-start matrix를 요구한다. 후보 비교에는 학습 시간뿐 아니라 reference set 구축을 위한 `질의×후보모델 호출 수`를 포함해야 한다.
5. **검증 세트가 없는 논문은 그대로 모방하지 않는다.** 29개 중 여러 연구가 별도 validation을 공개하지 않았고 CP-Router는 원문 구현 설명상 test 분포로 `α*`를 고른다. 새 통합 실험에서는 threshold를 train-only group OOF로 먼저 동결하고, validation은 checkpoint·모델 선택에만, test와 RouterArena는 최종 1회 평가에만 사용한다.

---

## 4. 서베이 논문 요약 (개별 라우터 스펙 없음)

| 논문 | 핵심 분류체계·결론 |
| --- | --- |
| **Doing More with Less** (2502.00409) | 시간축(생성 전/후 라우팅) × 구현방식(유사성 기반/지도학습/RL/생성형) 4분류. 표준화·비금전적 비용(에너지) 반영·적응형 전략 부재를 실무 문제로 지적. |
| **Towards Efficient Multi-LLM Inference** (2506.06579) | 라우팅 8개(Tryage/ZOOTER/FORC/Routoo/HybridLLM/OptLLM/MetaLLM/RouteLLM) vs 계층적 추론 6개(FrugalGPT/EcoAssistant 등) 비교. IES(추론 효율성 점수) 지표 제안. FrugalGPT 최대 98% 비용 절감 인용. |
| **Dynamic Model Routing and Cascading** (2603.04445) | 7개 패러다임(난이도/선호도/클러스터링/불확실성/RL/멀티모달/캐스케이딩) × 3차원 프레임워크(When/What/How). "잘 설계된 라우팅은 최강 단일 모델도 능가 가능" 주장. |

---

## 5. 종합 관찰

1. **분류기 아키텍처는 소형 인코더(BERT/DeBERTa/mDeBERTa, 100M 이하)가 압도적으로 많다** — RouteLLM, RouterDC, Hybrid LLM, IRT-Router, BEST-Route 등. ModernBERT를 명시적으로 채택한 사례는 vLLM Semantic Router, Cluster/Route/Escalate, The Routing Plateau(확장 실험)로, 이 프로젝트의 선택과 일치한다.
2. **생성형(소형 LLM) 라우터도 등장** — Arch-Router(Qwen2.5-1.5B), ICL-Router(Qwen2.5-7B-Instruct), Causal LLM Routing(Llama-3.2-1B 옵션). 해석성·유연성 강점이 있으나 레이턴시가 인코더 분류기보다 훨씬 큼(Arch-Router 51ms vs Claude-sonnet-3.7 1,450ms 비교에서도 LLM 자체는 소형이어야 함을 시사).
3. **RouterBench를 baseline으로 쓴 논문들의 공통 패턴**: 자체 라우터가 RouterBench 자체 predictive router(80.01%/$1.15 등)보다 근소 우위를 보이는 경우가 많지만, ModernBERT v2 balanced 실측 결과(§1-3)처럼 majority-class baseline조차 못 넘는 사례도 실제 프로젝트에서는 나타난다 — 논문의 보고 성능과 실제 재현 성능 사이 간극에 유의해야 한다.
4. **Training-free 계열(Eagle, CP-Router, VDAR-Router)** 은 재학습 비용 없이도 학습 기반 라우터와 경쟁력 있는 성능을 보고하며, 데이터가 부족한 상황에서 대안이 될 수 있다.
