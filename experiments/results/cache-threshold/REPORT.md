# 임베딩 계약과 캐시 임계값 실측

실험 코드: **CA-01** · [실험 목록](../../REGISTRY.md)

측정일: 2026-08-16
대상: 배포된 FastAPI `/router/embed` (`bge-large-en-v1.5`)
질문셋: `experiments/questions-v2.json` 92문항
스크립트: `experiments/probe_embeddings.py`

**Claude 호출 0회, DB 접근 0회.** 임베딩만 받아 계산했다.

이 문서는 **cross-encoder 리랭커를 도입하기로 한 근거**다.
결론부터: 어떤 코사인 임계값을 골라도 오탐과 누락 중 하나가 남는다.

## 1. 계약은 정상이다

```
차원      1024        (기대 1024)
L2 norm   1.000000    전 문항, 최소·최대 모두
```

정규화가 정확하므로 pgvector 에서 `유사도 = 1 - (a <=> b)` 가 그대로 성립한다.
**이 성질이 이후 모든 계산의 전제**다 — 오프라인 검증에서 pgvector 판정을
numpy 내적으로 재현할 수 있는 것도 이 때문이다.

정규화가 깨지면 `1 - 거리` 가 코사인 유사도가 아니게 되고 임계값이 통째로
무의미해진다. 그래서 계약 검증을 매 측정의 첫 단계로 둔다.

## 2. 핵심 문제 — 두 분포가 겹친다

```
paraphrase   (같은 질문, 캐시가 잡아야 함)   최소  0.8621
hard_negative(다른 질문, 막아야 함)          최대  0.9371
                                            분리 여유  -0.0750
```

**최악의 hard negative 가 최선의 paraphrase 보다 유사도가 높다.**
겹치는 구간이 0.075 나 되므로 단일 임계값으로는 두 클래스를 가를 수 없다.

## 3. 임계값 스윕 (92문항에서 만든 4,163쌍)

| threshold | recall | false hit | 비고 |
|---|---:|---:|---|
| 0.80 | 100% | 37 / 4163 | 당시 운영값 |
| 0.90 | 86% | 2 / 4163 | F1 최적 (0.878) |
| 0.94 | 38% | 0 / 4163 | 오탐 0 이지만 캐시가 사실상 무력화 |

**어느 지점도 만족스럽지 않다.** 0.80 은 오탐을 방치하고, 0.94 는 recall 38% 로
캐시를 쓰는 의미가 사라진다.

## 4. 어떤 방식으로 틀리는가

| 유형 | n | 최대 유사도 | 평균 |
|---|---:|---:|---:|
| `entity_swap` | 12 | **0.9371** | 0.8470 |
| `inverted` | 4 | 0.8947 | **0.8758** |
| `scope_shift` | 3 | 0.8920 | 0.8060 |
| `adjacent_concept` | 12 | 0.8977 | 0.7894 |
| `same_domain` | 6 | 0.7728 | 0.7090 |

두 가지를 봐야 한다.

**`entity_swap` 이 최대값을 만든다(0.9371).** 단어 하나만 바뀐 경우다.

**`inverted` 는 평균이 가장 높다(0.8758).** 표본은 4건뿐이지만 *전부* 높다.
답이 정반대인데 임베딩이 거의 같다는 뜻이라 가장 위험한 유형이다.

실제 사례:

```
0.9371  "Write a Python function ... longest common prefix"
     ↔  "Write a Java   function ... longest common prefix"

0.9026  "boiling point of water in Celsius"
     ↔  "boiling point of water in Fahrenheit"

0.8947  "red-black tree deletion 복잡도"
     ↔  "red-black tree insertion 복잡도"

0.8814  "왜 인덱스가 쿼리를 빠르게 하나"
     ↔  "언제 인덱스가 성능을 악화시키나"
```

마지막 두 개는 **답이 정반대**인데 유사도가 0.88 을 넘는다.

## 5. 원인 — 구조적 한계다

bi-encoder 는 두 문장을 **각각 따로** 벡터로 압축한다. `Python` 하나가 `Java` 로
바뀌어도 1024차원 중 극히 일부만 움직인다. 임베딩 모델을 더 좋은 것으로 바꿔도
같은 문제가 남는다 — 인코딩 방식 자체의 성질이기 때문이다.

cross-encoder 는 두 문장을 **함께 넣고** 토큰 간 attention 을 걸어 이 차이를
직접 본다. 관측된 최악의 두 유형(`entity_swap`, `inverted`)이 정확히
cross-encoder 가 고치는 케이스다.

→ 이 판단으로 리랭커를 도입했다. 후속 문서:
`../reranker-20260830/REPORT.md`(모델 선정), `../rerank-endtoend/REPORT.md`(실적용 검증).

## 6. 특별히 봐야 할 사항

**(1) 4,163쌍은 92문항의 모든 조합이다.** 서로 무관한 쌍이 대부분이라
"오탐 37/4163 = 0.9%" 를 실사용 오탐률로 읽으면 안 된다. 분모가 다르다.
실적용 검증(`../rerank-endtoend/`)은 캐시 조회 상황을 재현해 다시 쟀다.

**(2) `inverted` 표본이 4건뿐이다.** 평균이 가장 높다는 관찰은 방향으로만 쓰고
수치로 인용하지 않는다.

**(3) 여기서 정한 0.90 은 운영에 반영되지 않았다.**
운영 캐시 조회는 여전히 행별 `threshold` 0.80 이고, 대신 리랭커 게이트가
`코사인 >= 0.8774 AND 점수 >= 0.03582` 로 다시 거른다. 임계값이 두 곳에
있다는 점을 혼동하지 말 것.

**(4) 티어별 임계값 가설은 이 데이터로 반증됐다.**
초기 v1 질문셋(hard negative n=2)에서는 티어별로 임계값을 다르게 두면 좋아
보였으나, v2(n=37)에서는 최악 사례가 세 티어에 고르게 퍼져 있었다.
`cache_entry.threshold` 를 행별 컬럼으로 둔 구조는 유지하되 현재 세 값은 같다.

**(5) 전부 영어 기술 질문이다.** 다른 도메인·언어에서 같은 분포가 나오지 않는다.

## 7. 재현

```bash
set ROUTER_API_TOKEN=<토큰>
python experiments/probe_embeddings.py \
    --questions experiments/questions-v2.json \
    --csv experiments/rerank-eval-pairs.csv
```

`--csv` 는 리랭커 평가용 58쌍(positive 21, negative 37)을 뽑아낸다.
그 파일이 `../reranker-20260830/` 모델 선정의 입력이 됐다.
