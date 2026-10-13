# cache-natural-v1 / 2단계: 판정기 개선 비교 계획

작성: 2026-10-10. 새 점수를 계산하기 전에 고정한다.

## 1. 배경 조사 (시중 의미 캐시는 무엇을 하는가)

| 도구·연구 | 판정 방식 | 우리와의 관계 |
|---|---|---|
| GPTCache | 기본 `similarity_threshold=0.8` 거리 판정. 선택 판정기 `SbertCrossencoderEvaluation` 기본 모델 `cross-encoder/quora-distilroberta-base`, `OnnxModelEvaluation` 기본 `GPTCache/albert-duplicate-onnx` | 우리 C2·C3와 같은 구성. ALBERT ONNX는 8/30 스크리닝 최하위 |
| Redis LangCache | 캐시 전용으로 미세조정한 임베딩 `redis/langcache-embed-v3-small`(MiniLM 384차원, 약 800만 쌍) | 임베딩 자체를 바꾸는 접근 |
| vCache (ICLR 2026) | 캐시 항목마다 임계값을 온라인 학습. 정답 여부를 exact match 또는 LLM judge로 받아 오류율 상한을 보장 | "맞는 hit와 틀린 hit의 유사도 분포가 크게 겹친다"고 명시. 우리 진단과 같음 |
| Krites (2026) | 회색 구간을 LLM judge가 비동기로 검증. 의도와 개체·조건(이름, 장소, 시간, 수량) 일치를 확인 | 조건 비교를 LLM이 수행 |
| NLI 검증 | DeBERTa-v3 NLI cross-encoder로 양방향 함의 확인 | 판정기를 "같은 뜻인가"에 더 가까운 과제로 교체 |

## 2. 분할

300그룹을 `sha256(id)`의 첫 바이트를 3으로 나눈 나머지가 0이면 calibration, 아니면 test로 나눈다.
기준·규칙·조합은 calibration에서만 고른다. test는 각 방법을 고정한 뒤 한 번만 평가한다.

## 3. 방법

검색은 모두 운영과 같다(BGE 상위 5개, 코사인 ≥ 0.80). 판정 단계만 바꾼다. 후보가 여러 개면 판정 점수가 가장 높은
통과 후보를 쓴다.

| 이름 | 판정 |
|---|---|
| M0 | 운영 C3 그대로(코사인 ≥ 0.8774, quora ≥ 0.0358). 튜닝 없음 |
| M1 | C3 두 기준을 calibration에서 다시 고름 |
| M2 | NLI 양방향 함의 `min(P(a⇒b), P(b⇒a))` ≥ t. 모델 `cross-encoder/nli-deberta-v3-base` |
| M3 | LangCache 임베딩 코사인 ≥ t. 모델 `redis/langcache-embed-v3-small` |
| G | 조건 비교 규칙(숫자, 기술 개체, 부정, 반의어, 변환 방향, 출력 형식, 시간, 범위 한정어). 하나라도 어긋나면 막음 |
| M4~M6 | G + M1 / G + M2 / G + M3 |
| M7 | 위 신호 전체를 calibration에서 로지스틱 회귀로 결합 |

조건 규칙은 일반 범주로 작성하고, 규칙별 사용 여부만 calibration에서 정한다. calibration에서 함정을 막은 수가
같은 뜻 질문을 막은 수의 3배 이상인 규칙만 켠다.

## 4. 선택 규칙과 지표

- 각 방법의 기준은 calibration에서 **함정 통과율 ≤ 5%를 만족하면서 같은 뜻 재사용이 최대**인 값으로 정한다.
- test에서 보고: 같은 뜻 재사용(자기 원질문), 같은 뜻의 오재사용(다른 원질문), 함정 통과율, Clopper–Pearson 95% 구간,
  함정 비중 10%일 때 hit 중 잘못된 재사용.
- 주 비교: M0 대비 같은 함정 통과율 수준에서 같은 뜻을 얼마나 더 살리는가.

## 5. 한계 (사전)

- 데이터와 조건 규칙을 같은 작성자(Claude)가 만들었다. 규칙이 이 데이터의 표현 습관에 맞춰질 수 있어 결과가 낙관적일 수 있다.
  최종 후보는 팀원의 D2(PR #53 하네스)처럼 독립 데이터에서 다시 확인해야 한다.
- LLM judge(Krites, vCache 방식)는 API 비용이 들어 이번 단계에서 실행하지 않는다.
