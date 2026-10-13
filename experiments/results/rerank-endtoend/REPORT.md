# 리랭커 실적용 검증 — 오탐이 실제로 줄어드는가

실험 코드: **CA-03** · [실험 목록](../../REGISTRY.md)

측정일: 2026-08-30
대상: **배포된 운영 스택** (FastAPI `10.138.0.2:8000`, bge-large-en-v1.5, quora-distilroberta-base@f62e7a4)
스크립트: `experiments/validate_reranker_endtoend.py`

이 문서는 **캐시 판정에 리랭커를 실제로 켠(`mode=enforce`) 근거**다.
그 전까지는 오프라인 스크리닝(`../reranker-20260830/REPORT.md`) 결과만 있었고,
사전 기준(오탐 0건 + recall 90%)을 통과한 모델이 없어 판정에 쓰지 못하고 있었다.

## 1. 왜 이 측정을 했나

원래 계획은 `shadow` 모드로 실제 트래픽에서 점수를 모으는 것이었다.
**그런데 이건 실사용자가 없는 실험용 서비스다.** 며칠을 기다려도 시맨틱 히트가
5건 쌓이는 수준이라 판단 근거가 만들어지지 않는다.

그래서 데이터를 직접 넣어 측정했다.

## 2. 방법

`experiments/questions-v2.json` 92문항을 쓴다. 이 질문셋은 캐시 오탐을 재려고
설계된 것이라 목적에 정확히 맞는다.

| kind | 개수 | 역할 |
|---|---:|---|
| `base` | 19 | 캐시에 심는다 |
| `paraphrase` | 21 | **히트해야 한다** (정탐) |
| `hard_negative` | 37 | **히트하면 안 된다** (오탐) |
| `filler` | 15 | 히트하면 안 된다 (무관 질문) |

**Claude 호출 0회, DB 쓰기 0회.** 배포된 `/router/embed` 와 `/router/rerank` 만 부르고,
pgvector 의 판정을 numpy 로 재현했다. 재현이 정당한 근거는 두 가지다.

- `1 - (embedding <=> v)` 는 코사인 유사도이고, 반환 벡터가 L2 정규화되어 있음을
  이미 확인했다(전 문항 norm 1.000000, `RERANKER_REQUEST.md`, 2026-10-13 삭제·`git show 3c10d04:experiments/RERANKER_REQUEST.md`). 따라서 내적과 같다.
- 조회 SQL(top-k → 행별 threshold → 유사도 내림차순)을 그대로 옮겼다.

즉 **임베딩과 리랭커는 실제 배포본이 계산했고**, 그 위의 판정 로직만 재현했다.

## 3. 결과

```
임계값  A: 코사인 >= 0.80                                    ← 당시 운영값
        B: 코사인 >= 0.8774 AND 리랭커 >= 0.03581769         ← 스크리닝이 고른 조합
```

| | recall | 정탐 | 오탐 | 오탐률 |
|---|---:|---:|---:|---:|
| **A) 코사인 단독** | 100.0% | 21 / 21 | **19** | **36.5%** |
| **B) 코사인 AND 리랭커** | 90.5% | 19 / 21 | **1** | **1.9%** |

**오탐 19건 → 1건, 정탐 2건 손실.**

## 4. 어떤 오탐이 막혔나

```
유형                n    코사인 오탐    리랭커 통과
entity_swap        12        10             0
adjacent_concept   12         4             0
inverted            4         4             1
scope_shift         3         1             0
same_domain         6         0             0
filler             15         0             0
```

`same_domain` 과 `filler` 는 코사인만으로도 전부 걸러졌다. 문제는 **표면이 거의 같고
핵심 단어 하나만 다른** 유형에 몰려 있다.

코사인이 놓친 최악 5건:

```
cos=0.9371  score=0.000517  [entity_swap]
  질문: Write a Java   function that returns the longest common prefix ...
  후보: Write a Python function that returns the longest common prefix ...

cos=0.9026  score=0.000932  [entity_swap]
  질문: What is the boiling point of water in Fahrenheit?
  후보: What is the boiling point of water in Celsius?

cos=0.8977  score=0.010018  [entity_swap]
  질문: Derive the gradient update rule ... using mean squared error
  후보: Derive the backpropagation update rule ... with cross-entropy

cos=0.8977  score=0.029348  [adjacent_concept]
  질문: How does a hash function avoid producing collisions?
  후보: How does a hash table  handle  collisions?

cos=0.8947  score=0.113275  [inverted]
  질문: ... red-black tree insertion and its rebalancing cases
  후보: ... red-black tree deletion,  and why the rebalancing cases are exhaustive
```

전부 **답이 실제로 다른** 질문이다. 캐시가 히트하면 사용자는 틀린 답을 받는다.
Python 답을 Java 질문에 내보내는 것이 그 예다.

## 5. ⚠️ 특별히 봐야 할 값

### (1) 리랭커도 못 막은 오탐 1건

```
cos=0.8947  score=0.113275  [inverted]
  질문: Analyze the time and space complexity of a red-black tree insertion
        and its rebalancing cases.
  후보: Analyze the time and space complexity of a red-black tree deletion,
        and explain why the rebalancing cases are exhaustive.
```

점수 0.1133 은 임계값 0.0358 의 3배가 넘는다. 즉 **임계값을 조금 올리는 정도로는
못 막는다.** insertion↔deletion 처럼 구조가 같고 한 단어만 반대인 경우가
리랭커의 남은 약점이다. `inverted` 4건 중 1건이 통과했다.

### (2) 잃은 정탐 2건 — 둘 다 진짜 같은 질문이다

```
cos=0.8621  score=0.020212
  질문: What is the full form of the acronym HTTP?
  후보: What does HTTP stand for?

cos=0.9146  score=0.012978
  질문: At what temperature in Celsius does water boil?
  후보: What is the boiling point of water in Celsius?
```

두 번째는 코사인 0.9146 으로 충분히 높은데 리랭커 점수가 0.0130 이라 막혔다.
**질문 형태가 크게 다르면(평서형↔의문형, 어순 도치) 리랭커가 같은 질문임을 놓친다.**

다만 손실의 성격이 다르다. 이 경우 **캐시 미스가 되어 Claude 가 정상 답변을 만든다.**
사용자는 옳은 답을 받고 비용만 한 번 더 든다. 반대로 막지 못한 오탐은 **틀린 답이
그대로 나간다.** 이 비대칭이 enforce 를 택한 이유다.

### (3) 오탐률 36.5% 를 그대로 인용하면 안 된다

`hard_negative` 37문항은 **리랭커·임베딩의 약점을 노려 일부러 만든 것**이다.
실제 사용자 질문 분포에서 이런 쌍이 절반 가까이 나올 리 없다.

이 숫자가 말하는 것은 "우리 캐시가 실사용에서 36% 틀린다"가 아니라
**"이런 유형이 들어오면 코사인 단독으로는 막지 못한다"** 이다.
발표에서 절대 수치로 쓰지 말고 비교값(19건→1건)으로만 쓴다.

### (4) 임계값 두 개가 서로 다르다

운영 캐시 조회는 행별 `threshold`(현재 0.80)로 후보를 뽑고,
게이트는 그보다 높은 0.8774 를 AND 조건으로 다시 건다.
**0.80~0.8774 구간의 후보는 조회에는 걸리지만 게이트에서 떨어진다.**
의도된 동작이며, 스크리닝이 AND 조합으로 임계값을 잡았기 때문이다.

## 6. 결정

`llm.cache.rerank.mode` 를 **`enforce`** 로 두고 배포했다(`badc528`).

구현에서 함께 정한 것:

- **fail-closed** — 리랭커 장애 시 캐시 미스로 떨어뜨린다. 검증할 수 없는 캐시 답을
  내보내는 것보다 Claude 를 한 번 더 부르는 편이 싸다. `shadow` 는 반대로 예외를
  삼킨다(관측이 서비스를 막으면 안 된다).
- **모드 오타는 기동 실패** — `off` 로 조용히 떨어지면 리랭커가 안 도는 걸 아무도 모른다.
- **레거시 캐시 행은 통과** — 05 마이그레이션 이전 행은 `question` 이 비어 있어 채점할
  수 없다. 이것 때문에 캐시를 통째로 무력화하면 손해가 더 크다.

## 7. 배포 후 실측 확인

운영에서 같은 사례를 재현했다(`request_log`).

```
질문                            hit  cosine   score      rejected  reason
Java 함수 (entity swap)          f      —    0.000362      t      rerank_rejected
Python 패러프레이즈               t   0.9945  0.984290      f          —
Python 원본 (첫 요청)             f      —       —          —          —
```

배포 전이었다면 Java 질문에 Python 답이 나갔다.

## 8. 한계

- **19개 base 로 만든 캐시다.** 실제 캐시는 훨씬 크고, 커질수록 후보가 늘어
  오탐 기회도 는다. 이 측정은 작은 캐시 기준이다.
- **질문이 전부 영어 기술 질문이다.** 다른 도메인·언어에서 같은 비율이 나오지 않는다.
- **리랭커 지연을 재지 않았다.** 별도 측정에서 후보 5개 배치에 424~446ms 였고
  (교차 리전 왕복 약 120ms 포함), 운영 시맨틱 히트가 약 200ms → 1,178ms 로 늘었다.
  exact 히트(5ms)는 영향이 없다.
- **정탐 손실 2건은 표본이 작다.** recall 90.5% 의 신뢰구간은 계산하지 않았다.

## 9. 재현

```bash
set ROUTER_API_TOKEN=<토큰>
python experiments/validate_reranker_endtoend.py
```

원본 결과(문항별 유사도·점수·판정)는 `results.json` 에 있다.
