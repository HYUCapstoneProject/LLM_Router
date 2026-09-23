# GPTCache 방식과 현재 설계 비교

확인일: 2026-08-30

## GPTCache가 실제로 사용하는 흐름

GPTCache의 기본 `Cache.init()`은 문자열 passthrough와 exact match다. semantic cache를
빠르게 구성하는 `init_similar_cache()`는 다음 조합을 기본으로 만든다.

1. `GPTCache/paraphrase-albert-onnx`로 query embedding 생성
2. SQLite에 질문·답변을, FAISS에 embedding을 저장
3. vector search로 후보 조회
4. 기본 `SearchDistanceEvaluation`으로 각 후보를 평가
5. evaluator 점수가 설정 임계값 이상인 후보만 남기고 가장 높은 후보의 답을 반환

이는 [공식 `api.py`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/api.py)와
[판정 흐름](https://github.com/zilliztech/GPTCache/blob/main/gptcache/adapter/adapter.py)에서
확인했다. 기본 `Config.similarity_threshold`는 0.8이지만 evaluator의 점수 범위를
곱해 실제 rank threshold로 변환하므로, 우리 cosine 0.8과 같은 의미로 비교하면 안 된다.

## 제공하는 2차 판정 모델

GPTCache는 vector distance 외에 다음 두 로컬 pair evaluator를 제공한다.

- [`GPTCache/albert-duplicate-onnx`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/onnx.py):
  ALBERT tokenizer로 두 문장을 함께 입력하고 duplicate class 확률을 반환한다.
- [`cross-encoder/quora-distilroberta-base`](https://github.com/zilliztech/GPTCache/blob/main/gptcache/similarity_evaluation/sbert_crossencoder.py):
  `SbertCrossencoderEvaluation`의 기본 모델이다.

즉 GPTCache도 “embedding으로 후보를 찾고 pair model로 다시 판단”하는 구성을 지원한다.
우리의 pgvector → cross-encoder 구조와 원리는 같다. 다만 GPTCache 기본 quick-start는
2차 모델이 아니라 vector distance 평가이며, 특정 evaluator 사용과 임계값 튜닝은
사용자가 선택해야 한다.

## 자체 58쌍에서 직접 비교

GPTCache 소스와 동일하게 `albert-base-v2` tokenizer와
`GPTCache/albert-duplicate-onnx`의 class-1 출력을 사용했다. 모델 revision은
`4298cbbdba96523eb34a4c5bbcf5e935848793bc`다.

| 방식 | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| GPTCache ALBERT 단독 | 20 | 17 | 1 | 54.05% | 95.24% | 68.97% |
| BGE + GPTCache ALBERT | 19 | 7 | 2 | 73.08% | 90.48% | 80.85% |
| Quora cross-encoder 단독 | 20 | 3 | 1 | 86.96% | 95.24% | 90.91% |
| **BGE + Quora cross-encoder** | **20** | **2** | **1** | **90.91%** | **95.24%** | **93.02%** |

GPTCache ALBERT의 모델 단독 임계값은 0.00427679, joint 조건은 BGE 0.8738과
ALBERT 0.00232199였다. CPU pair당 평균/P95는 30.36/48.10ms, 추론 후 프로세스
RSS는 약 478MB, 모델과 tokenizer 캐시는 약 47MB였다.

## 적용 판단

- GPTCache 라이브러리 자체를 도입할 필요는 없다. 현재 PostgreSQL/pgvector와 Spring
  로그·비용 집계 구조를 유지하는 편이 책임과 실험 통제가 명확하다.
- GPTCache와 같은 2단계 구조는 타당하며, 그 기본 cross-encoder인 Quora 모델이 이번
  screening에서도 1순위였다.
- 하지만 GPTCache 모델도 엄격한 false-hit 기준을 통과하지 못했다. 공개 duplicate
  모델을 운영에 바로 고정하지 않고 캐시 동일성 데이터로 추가 검증·fine-tuning한다.
- exact match → BGE top-k → reranker → threshold → cache hit/miss 구조와 장애 시
  fail-closed 정책을 유지한다.

GPTCache 논문도 embedding과 similarity evaluator를 교체 가능한 모듈로 설명하며,
실험에서는 `paraphrase-albert-small-v2` 기반 ONNX embedding을 사용했다.
[공식 논문](https://aclanthology.org/2023.nlposs-1.24/)의 약 0.3초 cache-hit 수치는 해당
환경 전체 경로 측정값이지 우리 서버의 예상 latency가 아니다.
