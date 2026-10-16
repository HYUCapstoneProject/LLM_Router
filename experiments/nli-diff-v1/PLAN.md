# 차이 문장 NLI + 다른 NLI 모델 비교 (2026-10-12, 채점 전 고정)

## 질문

1. 긴 입력에서 공통 문장을 빼고 달라진 문장만 NLI에 넣으면, 점수 포화가 풀려 기준값이 정상 범위로 내려오는가?
2. 학습 방식이 다른 NLI 모델(적대적 데이터, 긴 입력, 이진 함의)이 지금 모델보다 나은가?

## 데이터·분할

`nli-adoption-v1`과 같다. 대화형(`cache-natural-v1`) + D2 재현본(`d2-replica-v1`), 기존 calibration/test 분할, 기존 검색 후보.
차이 추출은 `diffpair.py`(채점 전에 고정, 문장 단위 difflib). 원문 텍스트는 git에 올리지 않는다.

## 모델 (리비전 고정)

| 모델 | 학습 | 출력 | 최대 길이 |
|---|---|---|---:|
| cross-encoder/nli-deberta-v3-base, -xsmall (현재 후보) | SNLI+MNLI | 3분류 | 512 |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli @6f5cf0a | MNLI+FEVER+ANLI(적대적) | 3분류 | 512 |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 @8e7e5af | 여러 분류 과제를 함의로 변환 | 이진 | 512 |
| MoritzLaurer/ModernBERT-base-zeroshot-v2.0 @d421c45 | 위와 같음, ModernBERT | 이진 | 1280 |
| tasksource/deberta-base-long-nli @04dcf11 | 긴 입력 NLI 포함 다과제 | 3분류 | 1280 |
| tasksource/deberta-small-long-nli @9a77395 | 위와 같음, small | 3분류 | 1280 |

점수: 양방향 함의 확률의 최소값. 형태: 원문(full), 차이 문장(diff). 후보는 코사인 1위만.

## 비교 방식 (모델 × 형태마다)

- A 기준 하나: 합친 calibration 함정 ≤ 5%에서 같은 뜻 재사용 최대.
- B 길이별 기준: `nli-adoption-v1`과 같은 절차(L 40~800, 구간별 함정 ≤ 5%).
- B-diff: 짧은 구간(≤ L)은 원문, 긴 구간은 차이 문장 점수. L과 기준은 같은 절차로 고른다.
- 고정 0.5: 관례적 기준(선택 없음).

## 판단 기준 (미리 정함)

- 질문 1 "포화 해소": diff 형태의 A(기준 하나)가 두 test 모두에서 함정 ≤ 5%를 지키고, 기준값이 0.99 미만이며,
  대화형 같은 뜻 재사용이 B-full base(72.7%)보다 5%p 이상 낮지 않다.
- 질문 2: 같은 방식(B 또는 A)에서 두 test의 같은 뜻 재사용 합이 현재 base보다 높고 두 test 함정 ≤ 5%이면 "더 낫다".
- 사후에 추가하는 비교는 사후 분석으로 따로 표시한다.

## 알려진 한계

- 두 데이터 모두 Claude 작성. 대화형은 모두 99자 이하.
- 차이 추출은 문장 단위라, 한 문장 안의 일부만 바뀌어도 그 문장 전체가 남는다. 대명사가 앞 문장을 가리키면 맥락이 빠진다.
