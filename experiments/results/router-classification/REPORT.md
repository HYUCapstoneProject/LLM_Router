# 라우터 분류 성능 실측

측정일: 2026-08-16 (개방형 질문), 2026-08-30 (RouterBench held-out)
대상: 배포된 FastAPI `/router/classify`
모델: `router-modernbert-base-routerbench-20260814-45dcc6d`
스크립트: `experiments/probe_router.py`, `experiments/fetch_router_predictions.py`

**Claude 호출 0회.** `/router/classify` 만 부른다.

## 1. 요약 — 결론이 데이터에 따라 갈린다

**같은 모델인데 두 데이터에서 전혀 다르게 행동한다.** 이게 이 문서의 핵심이다.

| 평가셋 | n | 라벨 일치율 | 예측 분포 |
|---|---:|---:|---|
| RouterBench held-out test (학습 도메인) | 3,218 | **47.5%** | easy 1157 / medium 801 / hard 1260 — 균형 |
| 직접 만든 개방형 영어 질문 (도메인 밖) | 35 | **31.4%** | easy 29 / medium 2 / hard 4 — **easy 로 붕괴** |
| 위 질문의 한국어본 | 19 | 42.1% | hard 적중 **0/6** |

즉 **"라우터가 easy 로 붕괴한다"는 서술은 개방형 질문에 한정된 것**이고,
학습 도메인 안에서는 붕괴하지 않는다. 이 구분 없이 인용하면 틀린 주장이 된다.

## 2. RouterBench held-out test (n=3,218)

학습셋과 같은 분포에서 뽑은 held-out split 전량이다.

```
predicted →     easy   medium   hard
true easy        730      336    424
true medium      217      265    302
true hard        210      200    534

라벨 일치율 47.5%   (3-class 무작위 = 33.3%)
```

티어별 recall:

| 참 라벨 | recall | n |
|---|---:|---:|
| easy | 49.0% | 1,490 |
| medium | **33.8%** | 784 |
| hard | 56.6% | 944 |

**medium 이 가장 약하다(33.8%).** 3-class 무작위와 같은 수준이라 사실상 구분하지
못하고 있다. medium 오답 519건 중 302건이 hard 로 갔다 — 어중간한 질문을
과대평가해 비싼 모델로 보낸다는 뜻이다.

validation split(n=1,000)에서도 같은 패턴이다: 일치율 44.2%, medium recall 31.1%.

### ⚠️ confidence 가 낮다

```
중앙값 0.535 · 평균 0.567
```

3-class 최대 확신이 1.0, 무작위가 0.333 인데 중앙값이 0.535 다.
**모델이 스스로도 확신하지 못하고 있다.** confidence 를 게이트로 쓸 계획이 있다면
(예: 낮으면 상위 티어로) 이 분포를 먼저 봐야 한다.

## 3. 직접 만든 개방형 질문 (n=35, 영어)

시연 질문을 고르려고 만든 질문셋이다
(`experiments/questions-demo-en.json`, `questions-demo-en2.json`).

```
의도한 티어 →  실제 예측
easy    6/7 적중
medium  0/6 적중   ← 전부 easy 로
hard    3/6 적중

2차 후보 16문항 추가 투입 → medium 후보 18개 중 1개만 medium (confidence 0.37)
```

합계 35문항의 예측 분포: **easy 29 / medium 2 / hard 4**

특히 눈에 띄는 오분류:

```
"Explain the CAP theorem and give a concrete example ..."   → easy, confidence 0.88
"Explain how garbage collection works in the JVM ..."       → easy, confidence 0.76
"Write a function that detects a cycle in a directed graph" → easy, confidence 0.91
```

**틀리면서 확신한다.** confidence 0.88 로 easy 를 고른다.

### 원인 추정

학습 데이터가 객관식·상식 문제에 치우쳐 있다. RouterBench 36,497건 중
hellaswag 10,042 + grade-school-math 7,450 으로 절반 가까이다.
개방형 서술 질문은 학습 분포 밖이고, 그런 입력에서 easy 쪽으로 쏠린다.

기획서 3-3 이 예측한 클래스 불균형 위험이 실제로 관측된 것이다.

## 4. 한국어 (n=19)

같은 질문의 한국어 번역본이다(`questions-demo-ko.json`).

| | easy 적중 | medium 적중 | hard 적중 | 전체 |
|---|---|---|---|---|
| 영어 | 6/7 | 0/6 | 3/6 | 47% |
| 한국어 | 2/7 | **6/6** | **0/6** | 42% |

**한국어는 hard 6문항을 전부 easy 로 보낸다.** 시연의 핵심 장면인
"어려운 질문 → Opus" 가 한국어로는 절대 나오지 않는다.

confidence 도 0.37~0.60 에 몰려 무작위(0.333)와 큰 차이가 없다.

### ⚠️ 그래서 시연·평가는 영어로 한다

임베딩 모델이 `bge-large-**en**-v1.5` 이고 RouterBench 도 영어 전용이다.
한국어 입력은 두 모델 모두의 학습 분포 밖이다.

## 5. 특별히 봐야 할 사항

**(1) 일치율 47.5% 를 "정확도 47.5%" 로 읽으면 안 된다.**
`difficulty_label` 자체가 배포 모델과 다른 기준으로 만들어진 라벨이다
(`../pareto/REPORT.md` §5, 일치율 45.1%). 이 라벨을 100% 맞혀도 얻는 이득에
천장이 있다.

**(2) 개방형 질문 결과(31.4%)의 표본은 35건뿐이다.** 방향은 분명하지만
정밀한 수치로 인용하기엔 작다.

**(3) 두 결과가 모순이 아니다.** RouterBench 는 in-domain, 개방형 질문은
out-of-domain 이다. 같이 보면 **"학습 분포를 벗어나면 무너진다"** 는 하나의
이야기가 된다. 재학습 시 개방형 질문 데이터를 넣어야 한다는 근거다.

**(4) 시연에서 즉흥 질문을 쓰면 안 된다.** 검증된 질문셋이
`PLAN.md` §5-0 에 있다(easy 는 confidence 0.93~0.97 로 안정적, hard 는
0.52~0.64 로 맞지만 확신이 낮다).

## 6. 재현

```bash
set ROUTER_API_TOKEN=<토큰>

# 개방형 질문 (즉시)
python experiments/probe_router.py --questions experiments/questions-demo-en.json
python experiments/probe_router.py --questions experiments/questions-demo-ko.json
python experiments/probe_router.py --questions experiments/questions-demo-en2.json

# RouterBench held-out (test 전량 약 42분)
python experiments/fetch_router_predictions.py --split test --limit 0
```

RouterBench 예측 원본은 `../pareto/router_predictions_test.csv`,
`router_predictions_validation.csv` 에 있다.
