# 문서 제공형(RAG형) 평가 문항 제외 목록 (EV-10)

작성일: 2026-10-18. 파일: [document-grounded-eval-exclusions.csv](document-grounded-eval-exclusions.csv) (sample_id와 출처만 담았고 질문 원문은 없다).

## 왜

프로젝트 데이터 규약은 RAG·문서 제공형 질문(지문을 붙여 주고 그 지문으로 답하게 하는 형식)을 뺀다. 외부 데이터(LLMRouterBench·SPROUT)는 준비 단계에서 이 규칙으로 1,086건을 뺐고([외부 평가 데이터](../../../docs/ml/EXTERNAL_EVALUATION_DATA.md)), RouterBench는 RAG 문항(`needs_retrieval`)을 뺐다. 그런데 A 분할을 만들 때의 규약([ML 명세](../../../docs/ml/README.md) 2.2절)은 xRouteBench의 SQuAD·BoolQ를 RAG로 보지 않아 남겨 두었다. 두 문항 모두 "Answer the question using the provided context. Question: … Passage: …" 형식(지문 중앙값 약 735자)이다. 외부 데이터의 자동 규칙은 글 맨 앞의 `Context:`만 찾기 때문에 이 형식을 잡지 못했다.

사용자 결정(10-18): **평가에서만 뺀다.** 학습(train 472건, 1.4%)은 다시 하지 않는다.

## 규칙과 결과

- 규칙: xRouteBench의 카테고리가 `squad` 또는 `boolq`인 dev·test 문항 전부(문항을 하나씩 고르지 않는다).
- 결과: dev 80건(boolq 41, squad 39), test 100건(boolq 50, squad 50). xRouteBench dev는 459건에서 379건이 된다.
- 07b가 dev와 모든 보정 출력에서, 08이 평가 행에서 뺀다. `selection-lock.json`의 `exclusions.document_grounded`에 SHA와 건수가 남는다.

## 같은 검사에서 남긴 것

넓은 패턴(`provided context`, `Passage:`, `Context:`, `read the following …` 등)으로 train·dev·test 전체를 다시 훑었고, SQuAD·BoolQ 밖에서 걸린 것은 아래뿐이며 모두 남긴다.

| 걸린 것 | 판단 |
|---|---|
| MMLU 역사 문제 "This question refers to the following information. Read the following excerpt/quote/letter …" | 인용문 자체가 묻는 대상이다. 규약이 남기라고 한 경우다 |
| HellaSwag(wikiHow) 문장 안의 "article:", "paragraph :" | 단어가 우연히 맞았다 |
| HLE 문제 설명 안의 "Context:", 참고 링크 | 문제 설명의 일부다 |
| ArenaHard의 "Context:"·"Background Context:" 단락 | 사용자가 쓴 과제 배경이다. ArenaHard는 test 진단 구간이다 |

## RouterArena (진단)

RouterArena 8,400건 중 1,603건은 `Context` 열에 지문·전제·코드가 있다(PubMedQA 466, NarrativeQA 383, LiveCodeBench 183 등). 라우터에는 `Question`만 넣었으므로 이 문항들은 불완전한 질문이다. 08의 RouterArena 진단(H3의 Bloom 라벨 상관, R0-집단 일치도)은 `Context`가 빈 6,797건만 쓴다.
