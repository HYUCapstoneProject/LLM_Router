# 결정 및 평가 계획 — cross-encoder 리랭커

대상: FastAPI Router 서버 팀

최초 작성: 2026-08-16
결정 갱신: 2026-08-30

## 결정 상태

- **채택:** pgvector 후보를 cross-encoder로 재검증하는 2단계 캐시 판정 구조
- **평가 후 확정:** 모델, 점수 임계값, 실행 장치(CPU/GPU)
- **운영 활성화 조건:** 오프라인 평가와 shadow mode에서 정확도·지연·장애 폴백 통과
- **비채택:** 캐시 판정마다 생성형 LLM을 호출하는 방식

단일 임베딩 임계값만으로는 아래에서 확인한 false hit과 paraphrase 누락을 동시에
제어할 수 없으므로 재검증 계층 자체는 채택한다. 다만 공개 cross-encoder는 일반적인
검색 관련성을 학습했을 수 있어, 특정 모델을 운영에 사용하기 전 자체 데이터 평가가
반드시 필요하다.

---

## 1. 배경 — 임베딩만으로는 캐시 오탐을 못 막습니다

실서버(`34.182.108.1:8000`)에 92문항을 `/router/embed` 로 던져 측정했습니다.
**Claude 는 호출하지 않았고 DB 도 건드리지 않았습니다.**

### 계약은 정상입니다

```
차원      1024
L2 norm   1.000000 (전 문항)
```

정규화가 정확해서 pgvector 에서 `유사도 = 1 - 거리` 가 그대로 성립합니다. 이건 문제없습니다.

### 문제는 클래스가 겹친다는 점입니다

```
paraphrase(같은 질문, 캐시가 잡아야 함)    최소  0.8621
hard_negative(다른 질문, 잡으면 안 됨)     최대  0.9371
분리 여유                                 -0.0750
```

**어떤 임계값을 골라도 오탐과 누락 중 하나는 발생합니다.**

| threshold | recall | false hit |
|---|---|---|
| 0.80 (현재 설정) | 100% | 37 / 4163 |
| 0.90 (F1 최적) | 86% | 2 / 4163 |
| 0.94 | 38% | 0 / 4163 |

### 어떤 방식으로 틀리는가

| 유형 | n | 최대 유사도 | 평균 |
|---|---|---|---|
| `entity_swap` | 12 | **0.9371** | 0.8470 |
| `inverted` | 4 | 0.8947 | **0.8758** |
| `scope_shift` | 3 | 0.8920 | 0.8060 |
| `adjacent_concept` | 12 | 0.8977 | 0.7894 |
| `same_domain` | 6 | 0.7728 | 0.7090 |

가장 위험한 실제 사례:

```
0.9371  "Write a Python function ... longest common prefix"
     ↔  "Write a Java function ... longest common prefix"

0.9026  "boiling point of water in Celsius"
     ↔  "boiling point of water in Fahrenheit"

0.8947  "red-black tree deletion 복잡도"
     ↔  "red-black tree insertion 복잡도"

0.8814  "왜 인덱스가 쿼리를 빠르게 하나"
     ↔  "언제 인덱스가 성능을 악화시키나"
```

마지막 두 개는 **답이 정반대**인데 유사도가 0.88을 넘습니다.

### 원인

bi-encoder 는 두 문장을 **각각 따로** 벡터로 압축합니다. `Python` 하나가 `Java` 로
바뀌어도 1024차원 중 극히 일부만 움직입니다. 구조적 한계라 임베딩 모델을 바꿔도
같은 문제가 남습니다.

cross-encoder 는 두 문장을 **함께 넣고** 토큰 간 attention 을 걸기 때문에
이 차이를 직접 볼 수 있습니다. 우리가 관측한 최악의 두 유형(`entity_swap`, `inverted`)이
정확히 cross-encoder 가 고치는 케이스입니다.

공개 벤치마크 수치를 우리 캐시의 성능으로 간주하지 않는다. 검색 관련성과
"같은 답변을 재사용해도 되는가"는 다른 목표이므로 아래 자체 평가셋으로 직접 판단한다.

---

## 2. 1차 screening 평가

`experiments/rerank-eval-pairs.csv` 의 **58쌍**에 리랭커 점수를 매겨
`reranker_score` 칸을 채워주세요.

```csv
pair_id,label,neg_type,tier,query,candidate,bi_encoder_similarity,reranker_score
g01~g01p1,positive,,easy,What is the capital of France?,Which city is the capital of France?,0.9746,
g01~g01n1,negative,entity_swap,easy,What is the capital of France?,What is the capital of Germany?,0.7009,
```

- `label=positive` (21쌍) — 같은 질문. **점수가 높아야 합니다**
- `label=negative` (37쌍) — 다른 질문. **점수가 낮아야 합니다**

### 평가한 후보 모델

| 모델 | 크기 | 비고 |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 약 22M | 가장 빠른 후보 |
| `BAAI/bge-reranker-base` | 약 278M | 임베딩과 같은 계열 |
| `cross-encoder/quora-distilroberta-base` | 약 82M | 중복 질문 판별 계열 |
| `GPTCache/albert-duplicate-onnx` | ALBERT ONNX | GPTCache 공식 pair evaluator |

**두 개 이상을 같은 환경에서 실행해 비교합니다.** 크기가 작을수록 좋습니다 —
게이트웨이가 캐시 히트마다 호출하는 경로라 지연이 그대로 사용자에게 갑니다.

2026-08-30 CPU screening에서 네 모델 모두 false hit 0건과 recall 90%를 동시에
만족하지 못했다. 현재 최선은 BGE cosine과 `quora-distilroberta-base`를 함께 적용한
조합이지만 운영 확정값이 아니다. 상세 수치와 오류 쌍은
`results/reranker-20260830/REPORT.md`에 기록했다.
GPTCache의 실제 기본 흐름과 모델 비교는 `GPTCACHE_COMPARISON.md`에 정리했다.

---

## 3. 평가 산출물과 통과 기준

1. 모델별 점수가 채워진 CSV
2. 모델명·revision·실행 장치
3. 로딩 시간, 전체 추론 시간, pair당 평균 및 P95 latency
4. positive/negative 점수 분포와 유형별 오류
5. 임계값별 precision, recall, false-hit rate, F1
6. 프로세스 최대 메모리와 모델 캐시 크기

58쌍은 모델 선별용 screening 데이터다. 1차 기준은 negative 37쌍 false hit 0개,
positive 21쌍 recall 90% 이상, 그리고 고정 임계값 대비 동일 false-hit 조건에서 더 높은
recall이다. 통과 모델은 더 큰 독립 test set과 shadow mode에서 다시 검증한다. 0/37을
운영 오류율 0%로 해석하지 않는다.

---

## 4. 연동 계약 (운영 모델 선정 후 확정)

```
POST /router/rerank
```

```json
// 요청 — 게이트웨이가 pgvector top-k 를 뽑은 뒤 한 번에 보낸다
{
  "query": "질문 원문",
  "candidates": ["후보1", "후보2", "후보3", "후보4", "후보5"]
}

// 응답
{
  "scores": [0.94, 0.31, 0.28, 0.12, 0.05],
  "modelVersion": "ms-marco-MiniLM-L-6-v2"
}
```

- `scores` 는 `candidates` 와 같은 순서·같은 길이여야 합니다.
- 점수 범위(0~1 정규화 여부, logit 그대로 여부)를 명시해주세요. 임계값을 잡으려면
  범위를 알아야 합니다.
- `modelVersion` 은 `request_log` 에 남깁니다. 모델이 바뀌면 임계값을 다시 잡아야 하고,
  이전 측정치와 비교하면 안 되기 때문입니다.

호출 흐름은 이렇게 됩니다.

```
exact-match 조회      ~5ms     ← 완전 동일 질문이면 여기서 끝 (embed 호출 없음)
    ↓ miss
/router/embed        ~150ms
    ↓
pgvector top-5         ~5ms
    ↓
/router/rerank       측정 필요 ← 후보 5개를 한 번에
    ↓
캐시 히트 확정
```

reranker 오류, timeout, 응답 길이 불일치, NaN 점수는 모두 cache miss로 처리한다.
비용은 증가해도 검증되지 않은 캐시 답변을 반환하지 않는 fail-closed 정책이다.

운영 반영 전에는 기존 임계값이 실제 응답을 결정하고 reranker는 점수와 latency만
기록하는 shadow mode로 실행한다. 검증을 통과한 뒤 feature flag로 최종 판정을 전환한다.

---

## 5. 왜 LLM 판정기가 아니라 리랭커인가

Haiku 같은 소형 LLM 으로 "두 질문이 같은가"를 묻는 방법도 검토했지만 리랭커가 낫다고
판단했습니다.

| | Haiku 판정기 | cross-encoder |
|---|---|---|
| 비용 | 약 \$0.0001/히트 | 0 |
| 지연 | 외부 API 왕복 포함 | 로컬 장치에서 실측 |
| 결정성 | 샘플링 변동 | **결정적** |
| 외부 의존 | Anthropic API | 없음 |

결정성이 특히 중요합니다. M7(9/30) 결과 freeze 가 있는데 판정기가 매번 다른 답을 내면
실험이 재현되지 않습니다.

그리고 **"LLM 호출을 줄이려고 만든 캐시인데 검증하려고 LLM 을 부른다"** 는 구성은
심사에서 지적받기 쉽습니다.

---

## 6. 참고 — 재현 방법

```bash
python experiments/probe_embeddings.py \
    --questions experiments/questions-v2.json \
    --csv experiments/rerank-eval-pairs.csv
```

`ROUTER_API_TOKEN` 환경변수가 필요합니다. Claude 호출도 DB 쓰기도 없습니다.
