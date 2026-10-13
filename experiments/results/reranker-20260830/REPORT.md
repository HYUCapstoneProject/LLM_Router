# Cross-encoder reranker screening 결과

실험 코드: **CA-02** · [실험 목록](../../REGISTRY.md)

평가일: 2026-08-30
평가 장치: CPU, batch size 8
평가셋: `rerank-eval-pairs.csv` 58쌍 (positive 21, hard negative 37)

## 결론

cross-encoder 재검증 계층은 유지한다. GPTCache 공식 ONNX evaluator를 포함한 네 후보 중
`cross-encoder/quora-distilroberta-base`와 기존 BGE cosine 조건을 함께 적용한 조합이
가장 좋았다. 그러나 사전에 정한 통과 기준인 **false hit 0건과 recall 90% 이상**을
동시에 만족한 후보는 없었다.

따라서 아래 값은 운영 확정값이 아니라 다음 검증을 위한 **shadow mode 후보값**이다.

```text
BGE cosine similarity >= 0.8774
AND quora-distilroberta score >= 0.03581769
```

이 조합은 TP 20, FP 2, TN 35, FN 1로 precision 90.91%, recall 95.24%,
false-hit rate 5.41%, F1 93.02%였다. 같은 recall 90% 조건의 BGE 단독 최선
(TP 19, FP 7, F1 80.85%)보다 개선됐지만, 58쌍의 작은 screening 결과를 운영
오류율로 해석하면 안 된다.

## 모델 단독 비교

각 모델에서 recall 90% 이상을 유지하면서 false hit을 최소화한 임계값이다.

| 모델 | revision | threshold | TP | FP | FN | precision | recall | F1 | 평균 / P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms-marco-MiniLM-L6-v2` | `233902d` | 1.775188 | 20 | 17 | 1 | 54.05% | 95.24% | 68.97% | 6.54 / 8.85 ms |
| `BAAI/bge-reranker-base` | `2cfc18c` | 0.970950 | 19 | 4 | 2 | 82.61% | 90.48% | 86.36% | 53.33 / 64.53 ms |
| `quora-distilroberta-base` | `f62e7a4` | 0.035818 | 20 | 3 | 1 | 86.96% | 95.24% | 90.91% | 21.83 / 29.45 ms |
| `GPTCache/albert-duplicate-onnx` | `4298cbb` | 0.004277 | 20 | 17 | 1 | 54.05% | 95.24% | 68.97% | 30.36 / 48.10 ms |

CPU 프로세스 RSS는 추론 후 MiniLM, BGE, Quora, GPTCache ALBERT 순서로 약 486MB,
1,089MB, 640MB, 478MB였다. 내려받은 모델 캐시는 약 88MB, 1,082MB, 318MB,
47MB였다. 로딩 시간은 각각 18.4초, 126.2초, 45.3초, 2.4초로 측정됐다.
이 수치는 현재 개발 PC에서 후보 간 상대 비교용이며 GCE의
실제 요청 지연으로 간주하지 않는다.

## 기존 BGE와 결합한 결과

캐시 히트는 다음 두 조건을 모두 만족하도록 모든 임계값 조합을 탐색했다.

```text
bi_encoder_similarity >= similarity_threshold
AND reranker_score >= reranker_threshold
```

| 모델 | similarity threshold | reranker threshold | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | 0.8824 | -0.074757 | 19 | 7 | 2 | 73.08% | 90.48% | 80.85% |
| BGE reranker base | 0.8621 | 0.970950 | 19 | 3 | 2 | 86.36% | 90.48% | 88.37% |
| Quora DistilRoBERTa | **0.8774** | **0.035818** | **20** | **2** | **1** | **90.91%** | **95.24%** | **93.02%** |
| GPTCache ALBERT ONNX | 0.8738 | 0.002322 | 19 | 7 | 2 | 73.08% | 90.48% | 80.85% |

## 최선 조합의 오류

False hit 2건:

- red-black tree **삭제** 복잡도 ↔ **삽입** 복잡도
- **quicksort** 평균 복잡도 ↔ **mergesort** 전체 복잡도

Miss 1건:

- `HTTP가 무엇의 약자인가` ↔ `HTTP acronym의 full form은 무엇인가`

남은 오류는 일반적인 문장 관련성·중복질문 학습 목표와 “동일한 답을 안전하게
재사용할 수 있는가”가 다르다는 것을 보여준다. tier별 임계값도 확인했으나 tier당
positive가 6~9개뿐이라 확정 근거로 사용하지 않았다.

## 결정과 다음 검증

1. `quora-distilroberta-base`를 1순위 후보로 유지하되 운영 모델로 확정하지 않는다.
2. entity swap, 부정·반전, 인접 알고리즘/개념 쌍을 늘린 독립 평가셋을 group 단위로
   분리한다.
3. 해당 데이터로 reranker를 fine-tuning하거나 캐시 동일성 전용 모델을 추가 비교한다.
4. 독립 평가에서 false hit 제한과 recall 기준을 통과하면 임계값을 다시 고정한다.
5. 그 전까지 Spring 연동은 shadow mode와 fail-closed 경로만 구현한다.

원본 점수와 실행 메타데이터는 각 모델 하위 디렉터리의 CSV/JSON에 보존했다.
GPTCache 구현과의 구조·모델 비교는 `../../GPTCACHE_COMPARISON.md`에 정리했다.
