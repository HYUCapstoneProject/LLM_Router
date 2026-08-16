# 요청 — cross-encoder 리랭커 평가

대상: FastAPI Router 서버 팀

기준일: 2026-08-16

배포 전에 **오프라인 평가만** 부탁드립니다. 반나절이면 되고 비용은 0입니다.
결과에 따라 리랭커를 실제로 붙일지 결정합니다.

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

공개된 벤치마크 기준으로 bi-encoder 단독 precision@1 은 약 85%, cross-encoder 리랭킹을
얹으면 96~98% 이고, 시맨틱 캐시 워크로드에서 오탐률이 60~80% 감소합니다.

---

## 2. 요청 내용

`experiments/rerank-eval-pairs.csv` 의 **58쌍**에 리랭커 점수를 매겨
`reranker_score` 칸을 채워주세요.

```csv
pair_id,label,neg_type,tier,query,candidate,bi_encoder_similarity,reranker_score
g01~g01p1,positive,,easy,What is the capital of France?,Which city is the capital of France?,0.9746,
g01~g01n1,negative,entity_swap,easy,What is the capital of France?,What is the capital of Germany?,0.7009,
```

- `label=positive` (21쌍) — 같은 질문. **점수가 높아야 합니다**
- `label=negative` (37쌍) — 다른 질문. **점수가 낮아야 합니다**

### 후보 모델

| 모델 | 크기 | 비고 |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 약 22M | 가장 가벼움. top-5 에 15~30ms |
| `BAAI/bge-reranker-base` | 약 278M | 임베딩과 같은 계열 |
| `BAAI/bge-reranker-v2-m3` | 더 큼 | 정확도 우선일 때 |

**두 개 이상 돌려주시면 비교해서 고르겠습니다.** 크기가 작을수록 좋습니다 —
게이트웨이가 캐시 히트마다 호출하는 경로라 지연이 그대로 사용자에게 갑니다.

---

## 3. 회신받고 싶은 것

1. 채워진 CSV
2. 사용한 모델명과 버전
3. 58쌍 추론에 걸린 시간 (CPU/GPU 여부 포함)
4. 서빙 VM 에 올렸을 때 예상 메모리 증가분

우리 쪽에서 positive/negative 분리도와 임계값을 계산해 붙일지 판단하겠습니다.

---

## 4. 붙이기로 하면 필요한 계약 (초안)

지금 만들지 마시고, 위 평가 결과가 좋을 때만 진행합니다.

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
/router/rerank       ~140ms    ← 후보 5개를 한 번에
    ↓
캐시 히트 확정
```

---

## 5. 왜 LLM 판정기가 아니라 리랭커인가

Haiku 같은 소형 LLM 으로 "두 질문이 같은가"를 묻는 방법도 검토했지만 리랭커가 낫다고
판단했습니다.

| | Haiku 판정기 | cross-encoder |
|---|---|---|
| 비용 | 약 \$0.0001/히트 | 0 |
| 지연 | 300~800ms | 140ms |
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
