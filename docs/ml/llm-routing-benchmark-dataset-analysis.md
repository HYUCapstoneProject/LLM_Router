# LLM 라우팅 벤치마크 데이터셋 분석 보고서

기준일: 2026-09-06
관련 문서: `docs/ml/llm-routing-literature-review.md`(논문 목록),
`docs/ml/llm-routing-model-specs-report.md`(모델 스펙·성능),
`docs/ml/routerbench-data-labeling-spec.md`,
`docs/ml/routerarena-routing-evaluation-spec.md`(현행 사용 계약),
`docs/ml/integrated-routing-data-and-model-comparison-plan.md`(xRouteBench 통합·최종 모델군 계약)

## 1. 조사 대상과 방법

LLM 난이도 기반 라우팅 연구에서 사용되는 **RouterBench, RouterArena, RouterEval,
LLMRouterBench, xRouteBench**를 arXiv abs/html 원문과 공식 공개 저장소·데이터 카드까지
대조해 구성·라벨링 방법·평가지표·강점/한계를 정리했다. 확인하지 못한 세부 항목은
"원문에서 확인 불가"로 표시했다.

이 프로젝트는 RouterBench로 초기 학습, RouterArena로 독립 외부평가(학습 피드백 금지)를
수행한다(`docs/ml/routerbench-data-labeling-spec.md`,
`docs/ml/routerarena-routing-evaluation-spec.md`). xRouteBench `llmrouter_generic` train은
다음 통합 버전의 학습 원천으로 확정했지만 pivot·라벨·중복 제거 구현은 아직 남아 있다.
아래 §2~3은 현행 두 벤치마크를 특히 상세히 다룬다.

---

## 2. RouterBench (arXiv:2403.12031, ICML 2024 Workshop)

링크: [arXiv](https://arxiv.org/abs/2403.12031) · [GitHub](https://github.com/withmartian/routerbench)

### 구성

| 항목 | 값 |
| --- | --- |
| 규모 | 405,467 query-model 평가 레코드. 공식 파일의 canonical query 36,497개, 이 프로젝트 핵심 범위 34,778개 |
| LLM 후보 | 11개 메인(오픈소스 6 + 독점 5). You.com·Perplexity는 RAG 비교 시스템이며 11개 LLM tier에는 포함하지 않음 |
| 소스 데이터셋 | 8개, 64개 세부 태스크 |

### 원천 데이터셋(태스크) 설명

| 데이터셋 | 측정 대상 |
| --- | --- |
| MMLU | 57개 하위과목에 걸친 사전학습 지식·다분야 추론 |
| MT-Bench | 다중턴 대화 전문가 쌍별 선호도(80문항) |
| MBPP | 입문 수준 프로그래밍 문제 해결 |
| HellaSwag | 문장 완성에서 모델-인간 상식 판단 격차 |
| Winogrande | 대명사 해석 기반 상식적 문맥 이해 |
| GSM8K | 다단계 수학 응용문제 추론 |
| ARC Challenge | 단순 검색을 넘어선 복잡한 QA 이해 |
| RAG(800쿼리) | 스포츠·역사·미디어·정치 등 검색 기반 답변 |

### 라벨링 방법

- MMLU/HellaSwag/GSM8K/ARC/Winogrande: 정답과의 exact-match(규칙 기반).
- MBPP/MT-Bench/RAG: GPT-4를 LLM-as-judge로 채점.
- 결과를 True/False → 0/1, 등급형은 [0,1]로 정규화해 통일.
- **난이도를 별도 지표로 정의하지 않음** — "적절한 난이도의 업계 응용 과제 선택"이라는 선정 원칙만 명시.

### 평가 지표

- **AIQ (Average Improvement in Quality)**: 비용-성능 평면에서 라우터의 비감소 볼록 껍질(non-decreasing convex hull) R̃_θ(c)을 그린 뒤, 비교 대상 라우터 간 공유 비용구간 [c_min, c_max]에서 곡선 아래 면적을 정규화.
- 보조: 정확도, 총 비용($), WTP(Willingness-To-Pay) 파라미터 λ.
- Random 라우터가 기저선.

### 0~1 점수에서 프로젝트 라벨까지

RouterBench의 질문×모델 `performance`는 0~1이다. exact-match 계열은 사실상 0/1이고,
GSM8K partial credit와 judge 계열은 중간값을 가질 수 있다. 이 프로젝트는 이 원점수를
곧바로 `easy/medium/hard`로 해석하지 않고 다음 순서로 변환한다.

1. exact-match는 `performance == 1`, 연속 점수는 `performance >= 0.70`을 성공으로
   이진화한다.
2. 모델을 RouterBench 관측 성능으로 low/medium/high capability tier에 나눈다.
3. 질문마다 tier별 `success_rate = 성공 모델 수 / 유효 모델 수`를 0~1로 계산한다.
4. 현행 v2는 low `>=0.67`이면 easy, 아니면서 medium `>=0.67`이면 medium,
   아니면서 high `>=1/3`이면 hard, 그 외는 unresolved로 둔다.

별도 v3 probe는 tier 경계를 없애고 11개 모델 전체 solve rate가 `>=0.64`면 easy,
`>=0.27`이면 medium, 나머지는 hard로 실험한다. v2는 "최소 충분 모델 tier", v3는
"후보 모델군이 관측한 상대 난이도"이므로 같은 숫자 라벨이어도 의미가 다르다. 정확한
계약과 실측 분포는 `routerbench-data-labeling-spec.md` §14~15가 소유한다.

### 강점/한계 (논문 자체 언급)

- 지연시간·처리량 미포함.
- 예측형/캐스케이딩 라우터 위주 평가, 더 고급 라우터 설계 미탐색.
- RAG 평가는 내재적 검색 능력 보유 모델에 국한, 검색기-LLM 2단계 라우팅 미해결.
- 도메인 특화·장문맥 등 현실 시나리오 반영 부족.
- (후속 연구가 지적) 후보 모델 풀이 구형·소규모, 난이도 구분 부재.

### 이 프로젝트에서의 현황 (참고)

`docs/ml/training-reports/modernbert-v2-balanced-training-report.md`의 실측에 따르면, RouterBench tier 라벨을 구성하는 11개 모델 중 `claude-instant-v1`/`claude-v1`/`claude-v2`/`gpt-3.5-turbo-1106`/`gpt-4-1106-preview`는 모두 서비스 종료·폐기된 스냅샷이라 **동일 라벨 의미로 신규 데이터를 확충하는 것이 현재 불가능**하다는 한계가 실무적으로 확인됐다.

---

## 3. RouterArena (arXiv:2510.00202, ICLR 2026)

링크: [arXiv](https://arxiv.org/abs/2510.00202) · [GitHub](https://github.com/RouteWorks/RouterArena) · [리더보드](https://routeworks.github.io/)

### 구성

| 항목 | 값 |
| --- | --- |
| 최종 평가 질의 수 | 8,400개 (원시 62,000개 → 약 30,000개 필터링 후 재구성) |
| 원천 데이터셋 | 23개 오픈소스 데이터셋(대형 데이터셋은 각 ~1,000개로 제한) |
| 난이도 라벨링용 모델 | 42개(GPT-5 74.04%, Claude-3.5-Sonnet 67.73%, DeepSeek-Chat 66.86% 등 정확도 순 확인) |
| 비교 대상 라우터 | 12개(상용 NotDiamond/Azure 포함) |

### 라벨링/구축 방법

- 정답은 원 데이터셋 라벨을 재사용(별도 재채점 없음).
- 카테고리 분류(Bloom's Taxonomy 기반)는 DeepSeek-V3.1을 LLM-as-judge로 사용, 인간 검증 450샘플에서 정확 일치율 54.9%, ±1레벨 허용 시 76.7%.
- **난이도 정의(경험적, 42개 모델 정답 수 기준)**: Hard ≤4/42, Medium 5~19/42, Easy ≥20/42.
- 도메인 구조는 듀이십진분류법(DDC) 기반, 종교 영역 제외, 과학:인문학 비율 약 2:1. **9개 도메인×44개 카테고리 개별 명칭 전체 목록은 원문 도표(Figure 3)만 제시되고 텍스트 리스트로는 확인 불가.**

### 평가 지표 (5개 관점 + 종합)

1. **Query-answer Accuracy**: 전체 정확도 평균.
2. **Query-answer Cost**: C = 토큰당 단가 × 토큰 수(공식 API 가격, MoE 아키텍처 반영).
3. **Routing Optimality**(3개 하위): 최적 선택 비율(정답+최저가), 최적 정확도 비율, 최적 비용 비율.
4. **Routing Robustness**: 패러프레이징/문법변경/동의어치환/오타 노이즈 입력에서 동일 모델 선택 비율(5%, 420샘플).
5. **Routing Latency**: A100 기준 TTFT+종료시간.
6. **Arena Score**(종합): 가중조화평균 S_{i,β}=(1+β)·A_i·C_i/(β·A_i+C_i), 정규화 비용 C_i=[log₂(c_max)−log₂(c_i)]/[log₂(c_max)−log₂(c_min)], c_min=$0.0044, c_max=$200, 기본 β=0.1(정확도 우선).

정확도·optimality 비율·robustness와 정규화 비용 `C_i`, Arena Score는 0~1 범위다.
원시 비용(달러)과 latency(시간)는 0~1 점수가 아니다. 이 프로젝트가 보고하는
RouterArena accuracy·macro-F1도 0~1이지만, 이는 공식 Arena Score가 아니라
**RouterBench 학습 라벨과 RouterArena Difficulty의 이름 일치도**다.

### 강점/한계

- 자체 언급: 대부분 라우터의 Robustness 낮음(BERT 계열의 표면적 변화 민감성), 라우터 자체 레이턴시 오버헤드 무시 못할 수준, 모든 라우터가 오라클 미달.
- 개선 주장: RouterBench(24 카테고리) 대비 44 카테고리로 세분화, 3단계 난이도 도입, 단일 지표 아닌 5개 관점, 상용 라우터 포함.
- 12개 라우터 비교 결과 **어떤 라우터도 모든 지표에서 1위를 차지하지 못함**(정확도-비용 트레이드오프 명시적 존재).
- (LLMRouterBench가 지적) 통합 모델 풀 부재(라우터마다 다른 모델 풀), 코드/데이터 완전 공개 지연.

### 이 프로젝트에서의 현황 (참고)

RouterArena는 이 프로젝트에서 **inference-only 관찰(진단) 전용**이며 학습 피드백에 사용하지 않는다(`docs/ml/routerarena-routing-evaluation-spec.md` 금지 사항). 실측 결과 macro F1이 균등추측과 거의 동일해 현재 ModernBERT v2 checkpoint는 RouterArena 기준으로는 유효한 신호를 보이지 못했다(`docs/ml/training-reports/modernbert-v2-balanced-training-report.md` §5).

---

## 4. RouterEval (arXiv:2503.10657, EMNLP 2025 Findings)

링크: [arXiv](https://arxiv.org/abs/2503.10657) · [GitHub](https://github.com/MilkThink-Lab/RouterEval)

### 구성

| 항목 | 값 |
| --- | --- |
| 성능 기록 | 2억 건 이상 |
| 후보 LLM | 8,500개 이상(대다수 오픈소스, 로컬 배포 가능한 소규모 모델 중심) |
| 통합 소스 벤치마크(12개) | ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K, IFEval, BBH, GPQA, MUSR, MATH Level 5, MMLU-PRO |

개별 모델 8,500개 전수 명단은 부록 B 수록으로 본문 발췌 조사로는 확인 불가.

### 라벨링/구축 방법

- 기존 공개 벤치마크의 성능 기록(정답 여부)을 그대로 재사용, 별도 채점 없음.
- Selection Vector: 이진 메트릭(정답 시 1) / 연속 메트릭(최고 성능의 95% 이내 시 1).
- **난이도는 후보 모델 수 m으로 정의**: Easy m∈{3,5}, Hard m∈{10,100,1000}.

### 평가 지표

- **μ_o**(Original Metric): 라우터가 선택한 LLM들의 실제 정확도.
- **V_R**(Reference Value) = μ_o(r_θ)/Perf.(ref.) — 참조 모델(예: GPT-4) 대비 상대 성능.
- **V_B**(Best Single Model Value) = μ_o(r_θ)/Perf.(BSM) — 후보 풀 내 최고 단일 모델 대비 상대 성능.
- **E_p**(분류 편향/엔트로피): 라우터 예측 분포의 다양성(낮을수록 특정 모델 편중).

### 핵심 발견 — "라우팅의 스케일링 법칙"

라우터 능력 p≥0.5일 때 후보 모델 수 증가에 따라 성능이 신경망 스케일링 법칙과 유사하게 급격히 개선됨을 모델 레벨에서 실증. 약한 모델(개별 성능≤0.3)들도 유능한 라우터 하에서 상호보완해 "all-weak" 그룹의 오라클 성능이 MMLU에서 0.95까지 도달.

### 강점/한계

- 자체 언급: 데이터량도 "우수한 라우터 구축엔 불충분", 대규모 후보 모델 배포 시 계산비용 증가(3~10개 구성이 가장 비용-효과적), 대부분 벤치마크의 원 성능 기록이 비공개.
- (LLMRouterBench가 지적) 성능-비용 트레이드오프 미다룸(성능중심 평가만), 독점(상용) 모델 미포함.

---

## 5. LLMRouterBench (arXiv:2601.07206, ACL 2026 Findings)

링크: [arXiv](https://arxiv.org/abs/2601.07206) · [GitHub](https://github.com/ynulihao/LLMRouterBench)

> 2026년 1월 발표로 매우 최근 논문이라 사전 검색만으로는 존재 확인이 어려웠으나, WebFetch(html)·WebSearch·공식 GitHub 교차 확인으로 실존 확인됨.

### 구성

| 항목 | 값 |
| --- | --- |
| 규모 | 400K개 이상 인스턴스(프롬프트 23,945개), 데이터셋 21개, 모델 33개, 라우팅 baseline 10개 |
| 성능중심(Performance-oriented) 설정 | 15개 데이터셋, 경량 모델(~7~9B) 20개 — 수학(AIME/MATH500/MATHBench), 코드(HumanEval/MBPP/LiveCodeBench), 논리(BBH/KORBench/Knights&Knaves), 지식(MMLU-Pro/GPQA/FinQA/MedQA), 감정(EmoryNLP/MELD) |
| 성능-비용(Performance-Cost) 설정 | 10개 데이터셋, 대형/독점 모델 13개(GPT-5, Claude-sonnet-4, Gemini-2.5-Pro, Qwen3-235B, DeepSeek-V3, GLM-4.6, Kimi-K2 등 8개 제공업체, 입력가 $0.09~$3.00/1M토큰) — 수학(AIME/LiveMathBench), 코드(LiveCodeBench/SWE-Bench), 지식(GPQA/HLE/MMLU-Pro/SimpleQA), 명령어 준수(ArenaHard), 도구 사용(τ²-Bench) |

### 라벨링/구축 방법

- 데이터셋별 맞춤 평가기(정규표현식 매칭 등 규칙기반 + 일부 LLM judge, 약 $500 규모의 LLM 기반 채점 언급). 21개 데이터셋 전체의 채점 방식 상세 매핑은 원문에서 확인 불가.
- **난이도(query hardness)는 정답을 맞힌 모델 개수로 정의**(RouterEval·RouterArena와 유사 접근). 예: 최대 3개 모델만 정답을 맞힌 초고난도 질의가 410개(11.9%)이며 이 구간에서 라우터 정확도 급락(Avengers 24.6%, EmbedLLM 23.2%).

### 평가 지표

**성능중심**:
- AvgAcc = 전체 데이터셋 평균 정확도.
- Gain@b(b∈{Random, Best-Single}) = 기준 대비 상대 이득.
- Gap@O = 오라클(질의별 정답 생성 모델 중 최소비용 선택) 대비 격차.

**성능-비용**:
- PerfGain = 최고 성능 시점 Best-Single 대비 이득(최대 4%로 보고).
- CostSave = 동일 정확도 유지 시 비용 절감률(최대 31.7%로 보고).
- ParetoDist = 정규화 좌표에서 파레토 프론티어까지 평균 L1 거리.
- Latency-aware 분석: 토큰 사용량×OpenRouter 처리량 통계로 근사(저자들이 "특정 provider 설정에 한정된 참고용"이라 명시).

### 강점/한계 (Limitations 섹션 원문 인용 요약)

1. 모든 기존 라우팅 방법을 포괄하지 못함(공개 구현 있는 최신 접근법 위주). 모듈형 설계로 경량 어댑터 추가 통합 가능.
2. 현대 경량/플래그십 LLM 평가체계 반영 21개 데이터셋 구성 — 도메인 특화 버티컬, 초장문맥, 멀티모달 미포함.
3. 지연시간 분석은 근사치(OpenRouter 처리량 통계 기반, 확정적 값 아님).

### 4대 벤치마크 상호 비교 (LLMRouterBench 논문 Table 1 인용)

| 벤치마크 | 성능중심 | 성능-비용 | 데이터셋 수 | 독점모델 | 통합 모델풀 | baseline 수 | 완전공개 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RouterBench | ✗ | ✓ | 8 | ✓(GPT-4/3.5, Claude v1/v2) | ✓ | 3 | ✓ |
| RouterEval | ✓ | ✗ | 12 | ✗ | ✓ | 4 | ✓ |
| RouterArena | ✓ | ✓ | 23 | ✓ | ✗ | ✗ | 부분공개 |
| LLMRouterBench | ✓ | ✓ | 21 | ✓(최신) | ✓ | 10 | ✓ |

데이터 규모는 약 1.8B 토큰(FusionFactory 104M 대비 약 17배), 프롬프트 23,945개(RouterArena 8,400 대비 약 3배)라고 주장.

### 핵심 발견

- 어떤 단일 모델도 모든 도메인을 지배하지 못함(model complementarity 확인).
- 상위 라우팅 방법(EmbedLLM, GraphRouter, MODEL-SAT, Avengers)의 성능이 서로 유사하게 수렴 — `llm-routing-model-specs-report.md`의 "The Routing Plateau" 논문과 동일한 관찰.
- 오라클과의 격차는 주로 "모델 회수 실패"(정답 가능 모델이 1~3개뿐인 고난도 질의에서 검출 실패)에서 기인.
- 백본 임베딩 모델 종류는 라우팅 성능에 큰 영향 없음. 모델 풀은 무작정 키우기보다 신중한 큐레이션이 효과적(수확체감).

---

## 6. xRouteBench (arXiv:2608.06867)

링크: [arXiv](https://arxiv.org/abs/2608.06867) ·
[GitHub](https://github.com/ulab-uiuc/LLMRouter) ·
[Hugging Face](https://huggingface.co/datasets/ulab-ai/xRouteBench)

xRouteBench는 2026-08 공개된 LLMRouter 프로젝트의 query×candidate-model replay
데이터다. 각 질문을 동일한 18개 후보 모델에 실행해 응답, 0~1 `performance`, 토큰,
비용과 latency를 기록한다. 단일 텍스트뿐 아니라 memory/RAG, 시계열, 영상,
멀티모달 수학, 개인화 시나리오를 함께 제공한다.

| config | train 질의 | test 질의 | 이 프로젝트 처리 |
| --- | ---: | ---: | --- |
| `llmrouter_generic` | 4,487 | 3,729 | 텍스트 통합 후보 |
| `memory_locomo` | 885 | 314 | RAG이므로 제외 |
| `memory_longmemeval` | 277 | 101 | RAG이므로 제외 |
| timeseries/video/multimodal/personalized | 시나리오별 상이 | 시나리오별 상이 | 현행 text-only 난이도 분류 범위 밖 |

`llmrouter_generic`은 MMLU, GSM8K, MATH, MBPP, ARC 등을 포함하며 routing split은
train 80,802행, test 67,122행이다. 한 질문이 최대 18개 모델 행으로 반복되므로 모델
학습 전에는 `task_name + task_id + query` 기준으로 pivot해 **질문 한 건**으로 복원해야
한다. 공식 test는 통합 학습에 넣지 않고 외부 holdout으로 보존한다.

RouterBench와 달리 이미 세 capability tier가 있는 것이 아니므로 공개
`performance`를 이용해 우리 목적의 라벨을 다시 만들어야 한다. exact 계열의 0/1과
연속 F1을 섞어 평균내지 않고, metric별 acceptance rule로 성공 여부를 만든 다음 18개
모델의 solve rate를 계산한다. 세부 통합·중복 제거·threshold 선택 계약은
`integrated-routing-data-and-model-comparison-plan.md`가 소유한다.

공식 cost-aware 목적은 정규화한 성능과 비용으로
`reward = alpha * performance - beta * price_cost`를 사용한다. 이는 이 프로젝트의
3단계 난이도 라벨 목적함수와 같지 않으므로 그대로 라벨로 복사하지 않는다.

## 7. 5개 벤치마크 종합 비교

| 항목 | RouterBench | RouterArena | RouterEval | LLMRouterBench | xRouteBench |
| --- | --- | --- | --- | --- | --- |
| 발표 | 2024-03 | 2025-10 | 2025-03 | 2026-01 | 2026-08 |
| 규모(질의) | 36,497 query(405,467 평가 레코드) | 8,400 | 모델×문항 2억+ | 23,945 | generic 8,216 + 기타 시나리오 |
| 모델 후보 수 | 11 | 42(라벨링용)+평가 다수 | 8,500+ | 33 | 18 |
| 난이도 라벨 | 공식값 없음 | 3단계(정답 모델 수) | 후보 수 m으로 난이도 설정 | 정답 모델 수 기반 | 공식 3단계 없음 |
| 채점 방식 | exact-match+GPT-4 judge | 원 데이터셋 라벨 재사용+LLM 분류 | 원 벤치마크 재사용 | 규칙기반+일부 LLM judge | task별 metric, 0~1 performance |
| 핵심 지표 | AIQ | Arena Score | V_R/V_B+E_p | Gain/Gap/CostSave/ParetoDist | performance-cost reward |
| 이 프로젝트 사용처 | 학습 라벨 원천 | 독립 외부평가 | 미사용 | 미사용 | generic train 통합 확정, test 외부 holdout |

### 공통 한계 패턴

1. **후보 모델의 시간 경과 문제**: RouterBench의 구형 모델(Claude-instant-v1 등)이 서비스 종료되어 라벨 의미를 신규 데이터로 재현 불가 — 이는 이 프로젝트에서 실측으로 확인된 제약이다(§2 참고).
2. **난이도 정의가 대부분 "모델 정답 수/성공률" 기반의 상대적 지표**이며, 절대적 난이도 개념이 아니다. RouterArena·RouterEval·LLMRouterBench 모두 유사한 접근을 독립적으로 채택했다.
3. **텍스트만으로 라우팅하는 접근의 근본 한계**가 여러 논문에서 반복 관찰된다 — The Routing Plateau(§4, 모델 스펙 보고서)의 "예측 가능성 병목"과 이 프로젝트 ModernBERT v2 보고서의 "출처 문체 학습" 진단이 같은 현상을 서로 다른 벤치마크에서 재확인한 사례다.
4. **동일한 0~1도 의미가 다르다.** exact correctness, F1, judge score, 정규화 비용과
   라우터 종합점수를 한 열에 합치면 안 된다. 통합 데이터는 먼저 metric별 성공 계약을
   고정하고 나서 공통 solve rate를 만들어야 한다.
