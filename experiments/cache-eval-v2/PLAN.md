# 캐시 재평가 v2 — 배포한 NLI 구성의 동결 기준 검증 (2026-10-13, 실행 전 고정)

## 목적

10/13에 배포하는 캐시 구성(C4)이 **기준을 고른 적 없는 새 데이터**에서도 통하는지 본다. 기준값은 하나도 다시 고르지
않는다. Notion 13번(D1 반복 재생)과 14번 2(외부 도구 비교)의 질문을 이 구성으로 다시 잰다.

## 비교 구성 (모두 고정값)

| 이름 | 검색 | 판정 | 비고 |
|---|---|---|---|
| C1 | – | 공백 정리 후 문자열 완전 일치 | 운영 exact 캐시 |
| C2 | exact → BGE 1위 | 코사인 ≥ 0.80 | GPTCache 기본과 같은 방식(임베딩은 BGE) |
| C3 | exact → BGE 상위 5 | 코사인 ≥ 0.8774 AND quora ≥ 0.03581769, 통과 중 quora 최대 | 10/13 이전 운영 |
| **C4** | exact → BGE 1위(코사인 ≥ 0.80) | 차이 문장 양방향 NLI 최소 ≥ 0.28154424 | **10/13 배포 구성** |

점수는 배포한 서빙 코드(`LLM_Router-serving` `feat/nli-verifier`, transformers 4.48.3)의 엔진으로 낸다:
BGE(`TransformersBgeEmbedder`와 같은 CLS+L2, 512 토큰), quora(`TransformersCrossEncoderReranker`), NLI
(`TransformersNliReranker`, 1,280 토큰). 검색은 정확한 내적 검색(운영 HNSW와 4,000/4,000 일치가 13번에서 확인됨).

## 데이터 (원문은 git에 올리지 않는다)

학습 노출: NLI 모델(tasksource/deberta-base-long-nli)은 GLUE(QQP·MRPC), PAWS, SICK, PARADE, APT, HellaSwag, ARC 등으로
학습했다. 노출된 세트는 **[노출]**로 따로 보고하고 결론의 근거로 쓰지 않는다.

**흐름형 (요청을 순서대로 넣어 캐시가 쌓이는 운영 방식 재생).** 미스일 때만 저장, 히트는 저장하지 않음, 용량 제한 없음.
1. **LmArena** (vCache/SemBenchmarkLmArena @c6f3622, Apache-2.0): 실제 챗봇 사용자 질문 + GPT-4.1-nano 같은 뜻 변형.
   클래스 700개를 seed 20261013으로 뽑아 그 클래스의 질문 전부를 seed 순서로 섞는다. 같은 클래스면 정답 재사용.
2. **SearchQueries** (vCache/SemBenchmarkSearchQueries @5159bff, Apache-2.0): ORCAS 검색어. k-means 군집 안을 LLM 판정
   union-find로 나눈 `id_set`이 같으면 정답. 군집 단위로 seed 뽑기해 약 10,000개(같은 군집의 다른 `id_set` = 자연 함정).
3. **Classification** (vCache/SemBenchmarkClassification @373719c, Apache-2.0) **[일부 노출: CommonsenseQA←cos_e]**:
   8,000개 무작위. 정답 = 같은 데이터셋 안에서 저장된 응답 라벨이 같음(vCache 정의).
4. **D1 재생** (D_stream, 13번과 같은 원천): `query_text` 중 BGE 512토큰 이하. 요청 4,000개 흐름, 반복률
   r ∈ {0, 10, 30, 50%} × seed 3개. 반복은 같은 문자열. 다른 질문의 답을 재사용하면 오재사용 **[일부 노출: HellaSwag·ARC]**.

**쌍형 (D2 방식: 캐시 = 원문 쪽 전부, 조회 = 짝, 라벨 1이면 재사용해야 하고 0이면 막아야 함).**
5. **서비스형 신규 세트** (Claude 작성, 이번에 처음 만듦): 100~800자 서비스형 요청 그룹. 원질문, 가벼운 바꿔쓰기,
   크게 바꾼 바꿔쓰기, 함정 1~2개. 배경으로 LmArena 다른 클래스 첫 질문 3,000개를 캐시에 함께 넣는다.
   작성 후 채점 전에 해시를 고정한다.
6. **LangCache 문장쌍** (redis/langcache-sentencepairs-v1 @fcf7322, Apache-2.0) test: PIT2015 전부,
   **[노출]** QQP 4,000쌍·PAWS 4,000쌍·MRPC 전부(seed 표본).

## 지표

- 흐름형: 히트율(= 호출 절감), 히트 중 오재사용, 요청당 오재사용, 재사용 가능했던 요청 중 정답 히트(재현율).
- 쌍형: 같은 뜻 재사용률, 함정(라벨 0) 통과율, 같은 뜻이 다른 항목에 재사용된 비율. Clopper–Pearson 95%.
- C4가 C3보다 낫다고 말하는 조건(미리 정함): 노출 없는 세트(LmArena, SearchQueries, 신규 세트)에서
  오재사용(흐름형: 히트 중 오재사용, 쌍형: 함정 통과)이 C3보다 낮고, 그 대가로 줄어든 정답 재사용을 함께 보고한다.
- 운영 실측: 배포 후 신규 세트 일부를 운영 FastAPI에 직접 보내 점수 일치와 지연을 잰다.

## 한계 (미리 기록)

- vCache 세트의 같은 뜻 변형은 GPT-4.1-nano가 만들었고, SearchQueries 라벨도 LLM 판정이다.
- 신규 세트는 다시 Claude가 만든다. 기준을 고를 때 쓴 세트와 작성자가 같다.
