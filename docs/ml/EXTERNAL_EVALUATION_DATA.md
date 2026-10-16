# 추가 외부 평가 데이터 선정과 준비 (external-evaluation-v1)

실험 코드: **DT-03** · [실험 목록](../../experiments/REGISTRY.md)

작성일: 2026-10-01. [README 4.9절](README.md#external-evaluation-candidates)에서 조사한 후보 중 실제로 쓸 데이터를 고르고, 평가에 바로 쓸 수 있게 정리한 기록이다. **평가 전용**이며 학습·임계값 조정·모델 선택에 쓰지 않는다. 이 문서는 데이터 준비까지이고 모델 평가 결과는 없다.

## 1. 선정 기준

| # | 기준 | 이유 |
|---|---|---|
| C1 | **문항마다 여러 모델의 채점 결과**가 있다 | 우리 난이도 라벨(전체 모델 중 몇 %가 풀었나)을 같은 규칙으로 만들고, 라우팅한 모델의 품질을 잴 수 있어야 한다 |
| C2 | **실제 비용 기록**이 있다 | 품질만이 아니라 품질–비용을 함께 평가하려면 필요하다. 없으면 보조 세트로만 쓴다 |
| C3 | **우리가 이미 쓴 문항과 겹치지 않는다** | 학습·개발·보호·봉인 test·RouterArena와 같은 문항은 독립 평가가 아니다 |
| C4 | **단일 턴 텍스트 질문**이다 | 라우터 입력 계약과 같아야 한다. 에이전트·도구·다중 턴·그림 과제는 제외한다 |
| C5 | **고정 가능한 공개 revision**이 있다 | 파일 SHA로 같은 데이터를 다시 만들 수 있어야 한다 |
| C6 | **서비스 후보와 가까운 모델 세대**다 | 2025년 모델(Claude-sonnet-4, GPT-5, Gemini-2.5 등)이 있으면 서비스 상황에 더 가깝다 |

## 2. 후보 판정

| 후보 | 판정 | 근거 |
|---|---|---|
| **LLMRouterBench** | **주 세트** | C1·C2·C5·C6 충족. 2025년 flagship 13개 모델의 문항별 점수·토큰·**실제 비용**(공개 단가와 비율 1.0으로 일치 확인). 단, 일부 데이터셋은 C3·C4 때문에 제외 |
| **SPROUT** | **보조 세트** (품질·라벨만) | C1·C4·C5 충족. 2024년 13개 모델(Claude-3.5-sonnet, GPT-4o 등)의 판정 점수와 토큰은 있지만 **달러 비용 기록이 없어** C2 미충족. 일반 대화형 질문(OpenHermes)이 많아 서비스 질문 분포에 가깝다 |
| LiveBench | 보류 | 답변·점수 snapshot의 모델·문항 범위가 달라 교집합 검증이 먼저 필요. 비용 없음 |
| HELM Lite | 보류 | release·adapter별 수집이 필요하고 과거 모델 위주 |
| R2-Bench | 보류 | 모델 수·예산 설정이 카드와 코드에서 다르고 SPROUT과 문항이 겹칠 위험 |
| KMMLU-Pro | 보류 | 모델별 결과·비용이 없어 직접 호출이 필요. CC-BY-NC-ND 조건 확인 필요 |

**이용 조건.** 두 선정 데이터 모두 dataset card와 코드 저장소에 명시적 LICENSE가 없다(LLMRouterBench README에는 MIT 배지만 있음). 그래서 **원문·응답은 Git에 올리지 않고 로컬 연구 평가에만 쓴다.** 레포에는 설정·코드·집계 manifest·제외 목록(ID만)만 둔다. 외부 공개·재배포 전에는 저자 확인이 필요하다.

## 3. 가려서 쓴 데이터

### 3.1. 데이터셋 단위 제외

| 원천 | 제외 | 이유 |
|---|---|---|
| LLMRouterBench | swe-bench, tau2 | 저장소 패치·다중 턴 도구 사용 과제(C4). 풀 모델 하나씩 결측 |
| LLMRouterBench | arc-agi | 격자 퍼즐을 숫자 배열로 직렬화한 입력. 자연어 질문이 아님(C4) |
| LLMRouterBench | mmlupro/test_1000, hle·simpleqa subset_500, arenahard_* | 더 큰 세트의 부분집합·중복 보기 |
| LLMRouterBench | bbh, math500, mbpp, humaneval, medqa 등 13개 | 7~8B 소형 모델 결과만 있고 flagship 풀·비용 비교가 불가(C2·C6) |
| SPROUT | MMLU-Pro, MATH | 우리 개발 데이터(xRouteBench mmlu_pro·math)와 같은 계열이고, MMLU-Pro는 비용이 있는 주 세트 사본을 우선 |
| SPROUT | RAGBench 12종 | 검색 문서가 붙은 긴 입력. 프로젝트 데이터 계약이 RAG를 제외함 |
| SPROUT | MuSR 3종 (748) | 여러 문단의 이야기 지문을 주고 묻는 문서 기반 질문(RAG형 입력) |

### 3.2. 문항 단위 제외 (실제 건수)

| 원천·계열 | 사유 | 건수 |
|---|---|---:|
| LLMRouterBench mmlupro | **우리 데이터와 중복** (같은 질문 본문, 선택지만 확장) | 1,728 |
| LLMRouterBench aime | 우리 데이터와 중복 (xRouteBench AIME 2020–2024) | 18 |
| LLMRouterBench arenahard | 우리 데이터와 중복 | 1 |
| LLMRouterBench 전체 | 유효 모델 결과 12개 미만 (호출 실패 결측) | 206 |
| LLMRouterBench hle·arenahard | 같은 index인데 모델마다 질문이 다름 | 2 |
| SPROUT gpqa_extended | 주 세트 GPQA와 같은 문항 (주 세트 사본 유지) | 198 |
| SPROUT openhermes | 우리 데이터와 중복 | 1 |
| SPROUT openhermes·gpqa_extended | **RAG형**: `BEGININPUT/BEGINCONTEXT` 문서 블록, 앞머리 `Context:` 지문, "다음 지문을 읽고 답하라" | 338 |
| LLMRouterBench arenahard·hle, SPROUT gpqa_extended | ModernBERT 토큰 8,192 초과 | 4 |

MMLU-Pro 중복이 많은 이유는 MMLU-Pro가 **원래 MMLU 문항에 선택지를 10개로 늘려 만든 데이터**이기 때문이다. RouterBench의 MMLU와 질문 본문이 같다. 샘플을 직접 대조해 같은 질문임을 확인했다.

### 3.3. 문항 처리 규칙

- **입력:** 벤치마크가 붙인 답변 형식 지시문만 데이터셋별 고정 패턴으로 제거한다(예: "The last line of your response should be…", "\boxed{}", "Let's think step by step."). 질문·선택지·코드 조건은 남긴다. 패턴이 맞지 않으면 추측하지 않고 제외한다. mmlupro·gpqa는 `origin_query`에 선택지가 없어서 `prompt`에서 입력을 만든다.
- **성공 판정:** LLMRouterBench는 점수 1.0이 성공(정답형은 0/1). arenahard는 기준 답변 대비 판정(0/0.5/1)이라 1.0(명확한 승)만 성공으로 보고 **별도 진단 slice**로 분리한다. SPROUT은 판정 점수 0.70 이상이 성공(우리 부분점수 규칙).
- **결측:** 비용 0·출력 토큰 0인 기록(호출 실패), SPROUT의 빈 응답은 **오답이 아니라 결측**이다. 유효 결과가 12개 미만이면 제외한다.
- **난이도 라벨:** 우리와 같은 규칙 `overall-solve-rate-grid-v1:h0.27-e0.64`를 외부 모델 풀에 적용한다. 풀이 다르므로 "같은 정의, 다른 풀"의 라벨이다. 현재 기준인 v1 경계를 쓰며, 검증 데이터셋 v2(T1 0.36/0.73, [2.5.1절](README.md#outcome-audit))로 바꿀 때는 config의 경계만 교체한다. 외부 원천의 기록 점수는 이번 채점 오류 점검 대상이 아니었다.
- **중복 검사:** 프로젝트의 중복 규칙(정확 일치 + token 3-gram SimHash Jaccard ≥ 0.90)에 더해, **선택지 앞 질문 본문(10단어 이상)이 같은 경우**도 중복으로 본다. 선택지를 섞거나 형식을 바꾼 같은 문항을 잡기 위한 규칙이다.
- **RAG형 제외:** 문서를 붙여 주고 그 문서로 답하게 하는 형식은 뺀다. 질문 안에 인용문이 있어도 그 인용문 자체가 묻는 대상이면(예: 문장 읽기 실험 문제) 남긴다.
- **토큰 한계:** 라우터(ModernBERT) 토큰이 8,192를 넘으면 제외한다. 캐시 임베딩(BGE) 토큰 512 초과는 제외하지 않고 `within_cache_embed_limit=false`로 표시한다. 판단 근거는 5절에 있다.
- **분할:** 중복 그룹·같은 질문 본문·같은 긴 지문(1,000자 이상에서 앞 300자 동일)을 한 단위로 묶어 **calibration 30% / test 70%**로 고정한다. 한 지문의 여러 질문이 양쪽에 나뉘지 않는다.

## 4. 준비 결과

| 세트 | slice | 계열 | 문항 | 두 한계 모두 이내 |
|---|---|---|---:|---:|
| **LLMRouterBench** | 주 평가 (새 계열) | SimpleQA 4,326 · HLE 2,030 · LiveCodeBench 1,002 · GPQA 198 · LiveMathBench 118 | **7,674** | 7,055 |
| | 같은 계열 (참고) | MMLU-Pro 1,256 · AIME 40 | 1,296 | 1,266 |
| | 판정 선호 진단 | ArenaHard 740 | 740 | 625 |
| **SPROUT** | 일반 대화 | OpenHermes 19,117 | 19,117 | 19,117 |
| | 새 계열 | GPQA extended 341 | 341 | 330 |
| | | 합계 | **29,168** | **28,393** |

"두 한계 모두 이내"는 ModernBERT 8,192 토큰과 캐시 BGE 512 토큰을 모두 지키는 문항 수다. 캐시 한계를 넘는 775문항은 LiveCodeBench 396, HLE 216, ArenaHard 115 등 **길고 어려운 문항**에 몰려 있다. 난이도 라벨 분포(calibration + test):

| slice | easy | medium | hard |
|---|---:|---:|---:|
| LLMRouterBench 주 평가 | 1,659 | 1,920 | 4,095 |
| LLMRouterBench 같은 계열 | 1,104 | 97 | 95 |
| LLMRouterBench 판정 진단 | 304 | 305 | 131 |
| SPROUT 일반 대화 | 16,029 | 2,278 | 810 |
| SPROUT 새 계열 | 68 | 152 | 121 |

- 주 평가 세트는 HLE 때문에 **hard가 많다**(53%). SPROUT 일반 대화는 2024년 모델도 대부분 풀어서 **easy가 84%**다. 두 세트의 성격이 반대라 함께 보면 라우터가 양쪽에서 어떻게 움직이는지 확인할 수 있다.
- 입력 길이 중앙값은 ModernBERT 35 token, BGE 37 token이다. 남은 문항의 최대는 ModernBERT 7,488 token이다.
- LLMRouterBench 풀의 문항당 평균 기록 비용은 $0.00057(deepseek-v3-0324)부터 $0.0587(gemini-2.5-pro)까지 약 100배 차이다. 세 단계 후보는 **calibration 세트로만** 정하고 test에서 바꾸지 않는다.

## 5. 토큰 한계: 어느 쪽에 맞추나

| 단계 | 모델 | 한계 | 넘으면 |
|---|---|---:|---|
| 라우터 | ModernBERT (운영 1.0) | 8,192 | 뒤쪽을 잘라 분류 |
| 캐시 임베딩 | `BAAI/bge-large-en-v1.5` | 512 | 뒤쪽을 잘라 임베딩 |
| 캐시 reranker | cross-encoder (운영 비활성) | 512 (질문+후보 쌍) | 뒤쪽을 잘라 비교 |

**외부 라우터 평가는 라우터 한계(8,192)에 맞춘다.** 평가 대상이 라우터이고, 캐시는 긴 질문에서 실패하지 않고 앞 512 토큰만 보고 동작한다. 512로 자르면 길고 어려운 문항(LiveCodeBench의 40%, HLE의 11%)이 빠져서 세트가 쉬운 쪽으로 치우친다. 그러면 라우터가 가장 중요하게 판단해야 하는 문항을 평가하지 못한다.

**캐시 실험과 "서비스 전체 경로" 보고에는 512 이내 표시를 쓴다.** 앞 512 토큰이 같고 뒤가 다른 두 긴 질문은 같은 임베딩이 되어 다른 질문의 답을 재사용할 위험이 있다. 그래서 캐시 실험은 `within_cache_embed_limit=true`인 문항(28,393)만 쓰거나, 초과 문항을 따로 집계해야 한다. 이 위험은 캐시 설계의 한계이며 외부 데이터 선정으로 해결할 문제가 아니다.

## 6. 사용 규칙

> **2026-10-16 변경:** LLMRouterBench·SPROUT를 EV-10의 학습에도 쓰기로 했다([라우터 최종 실험 계획서](ROUTER_FINAL_EXPERIMENT_PLAN.md) 2.4절). 기존 calibration 30% / test 70%는 test를 연 적이 없으므로 버리고, 중복 묶음 단위로 train 50 / dev 15 / test 35로 다시 나눈다. RouterArena와 겹치는 700문항과 LLMRouterBench 판정 선호 진단(arenahard)은 학습에 넣지 않고 test에 둔다. RouterArena는 평가 전용으로 유지한다. 아래 규칙 1의 "calibration"은 새 dev로 읽고, 규칙 2·4는 이 변경에 맞춰 고쳤다.

1. test 세트는 평가 방식·후보 세 개·판정 기준을 calibration에서 모두 고정한 뒤 **한 번만** 평가한다.
2. 새 풀의 결과를 **기존 RB/xRB 결과의 외부 성능**이라고 보고하지 않는다. 10-16부터 단계는 원천마다 모델 집단(최종 계획서 2.1절)으로 정의하며, "A만 학습한 버전"의 결과만 "다른 모델 풀로의 전이"로 부른다.
3. 주 평가(새 계열)와 같은 계열·판정 진단·SPROUT 결과를 합산하지 않고 따로 보고한다.
4. 평가 비용은 모든 원천에서 서비스 가격 비율(Haiku:Sonnet:Opus = 1:3:5)로 통일한다(최종 계획서 2.3절). SPROUT의 데이터 저자 단가표 비용([SPROUT 비용 추산](SPROUT_COST_ESTIMATION.md))은 모델을 가격대 집단으로 나눌 때만 쓴다.
5. 라우터 평가는 전체(8,192 이내)로 하고, 캐시와 함께 보는 서비스 경로 결과는 `within_cache_embed_limit=true` 부분집합으로 따로 보고한다.
6. 공개 모델 결과이므로 Claude 서비스의 실제 성공률로 해석하지 않는다.

## 7. 재현

원본 파일은 `ml/artifacts/external-eval-raw/`(Git 제외)에 고정 revision으로 내려받아 SHA를 확인했다. 준비 코드는 [`external_data.py`](../../ml/src/llm_router_labeling/evaluation/external_data.py), 설정은 [`external_evaluation_v1.json`](../../ml/config/external_evaluation_v1.json)이다.

```bash
PYTHONPATH=ml/src python ml/scripts/prepare-external-evaluation.py \
  --config ml/config/external_evaluation_v1.json --repo-root . \
  --llmrouterbench-root ml/artifacts/external-eval-raw/llmrouterbench/bench-release \
  --sprout-dir ml/artifacts/external-eval-raw/sprout \
  --tokenizer ml/artifacts/r0-release-20260927/approved \
  --output-dir ml/artifacts/external-evaluation-v1
```

준비된 Parquet(원문 포함)은 `ml/artifacts/external-evaluation-v1/`에 두고, 원문이 없는 manifest와 제외 목록은 [`experiments/results/external-evaluation-data-v1-20261001/`](../../experiments/results/external-evaluation-data-v1-20261001/manifest.json)에 보존한다.
