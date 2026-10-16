# GPTCache 실제 실행 기준선 (2026-10-12, 실행 전 고정)

목적: 보고서의 비교 대상을 "GPTCache와 같은 방식을 우리 임베딩으로 재현"이 아니라 GPTCache 라이브러리 자체로 만든다.

## 실행 대상

`gptcache==0.1.44`(PyPI 최신)를 Cloud Build에서 설치해 `gptcache.adapter.api.put/get`으로 캐시에 넣고 조회한다.
캐시에 저장하는 값은 원질문 ID라, `get`이 돌려준 값으로 어느 항목을 재사용했는지 안다.

| 설정 | 임베딩 | 판정 (similarity_evaluation) | 기준 |
|---|---|---|---|
| G1 기본 | Onnx (GPTCache/paraphrase-albert-onnx) | SearchDistanceEvaluation (거리) | 0.8 |
| G2 리랭커 | 같음 | SbertCrossencoderEvaluation (cross-encoder/quora-distilroberta-base) | 0.8 |
| G3 중복질문 분류기 | 같음 | OnnxModelEvaluation (GPTCache/albert-duplicate-onnx) | 0.8 |

나머지는 `init_similar_cache` 기본값(sqlite + faiss, 후보 top_k=1, 후처리 temperature_softmax, temperature 0)을 쓴다.
**바꾼 기본값은 하나:** 퇴출 크기(`eviction_params["max_size"]`)를 1,000 → 10,000. 기본값이면 D2 캐시(5,286개)에서 LRU 퇴출이 일어나
GPTCache에 불리하다.

## 캐시와 질문

- 대화형: 캐시 = 원질문 300, 질문 = 변형 900(같은 뜻 600, 함정 300).
- D2 재현본: 캐시 = 원질문 300 + 배경 4,986, 질문 = 변형 600(같은 뜻 300, 함정 300).
- 지금까지와 같은 calibration/test 분할. 보고는 test.

## 지표

1. 기본 기준(0.8)의 결과 — 라이브러리를 그대로 쓴 결과.
2. 맞춘 기준: 같은 설정을 기준 0으로 한 번 더 돌려 `hit_callback`으로 1위 후보 점수를 받고, calibration 함정 통과 ≤ 5%에서
   같은 뜻 재사용이 최대인 기준을 고른 뒤 test에 적용한다. 기준 0.8에서 이 점수로 낸 판정이 1의 실제 결과와 모두 같은지
   확인한다(일치하지 않으면 2는 보고하지 않는다).
3. 예외: 임베딩·판정에서 예외가 나면 그 질문은 miss로 세고 개수를 보고한다(Onnx 임베딩은 512토큰을 넘으면 실패할 수 있다).

## 한계

- 데이터는 Claude 작성. 대화형은 모두 99자 이하.
- GPTCache 0.1.44는 2024년 이후 갱신이 없다.
