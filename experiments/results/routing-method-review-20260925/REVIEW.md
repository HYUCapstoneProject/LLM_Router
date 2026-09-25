# R0·R1·R2·R3 종합 판단과 natural/balanced의 영향

2026-09-25. 기존 A와 G1의 검증된 결과를 합쳐 해석한 후속 분석이다. 학습·추론·선정 lock·운영 모델을 변경하지 않았다. 비용 감점 없는 비교는 저장 logits에서 λ=0 행동을 재계산한 사후 진단이며 새 독립 시험 또는 재선정이 아니다.

## 판단

**종합 개발 후보는 R1-natural을 우선하고, R3-natural을 일반화·비용 측면의 대안으로 유지한다. 현 checkpoint의 즉시 운영 배포는 권고하지 않는다.** 비용을 무시한 G1 미관측 과제의 답변 성공률은 R1-balanced가 가장 높지만 비용 정책과 기본 검증에서 불리했다. 난이도 라벨을 맞히는 목적에서는 R0가 가장 높았다. 이들은 서로 다른 목표이므로 하나의 정확도 순위로 합치지 않는다.

| 방법 | 강점과 실측 근거 | 종합 판단 |
|---|---|---|
| R0 | Arena 난이도 라벨 일치율 45.48%, macro-F1 .3961로 측정한 모델들 중 최고 | 난이도 분류·회귀 기준선으로 보존. A의 실제 후보 선택 품질·비용은 약함 |
| R1 | A의 비용–품질 nAUC 1위, G1 selection도 natural 1위. G1 λ=0 holdout은 balanced 최고 | 서비스 후보별 성공 예측을 학습하는 주력 개발 방식으로 추천 |
| R2 | A에서 R1 다음 nAUC. 일부 운영점은 R1보다 저렴하고 xRouteBench D_dev 기본점에서 품질도 높음 | 유효한 대안이나 미관측 task의 G1 결과가 없어 R3와 직접 순위를 정할 수 없음 |
| R3 | G1 고정 기본 정책의 holdout에서 R1보다 높은 평균 품질, 두 source의 예산 통과 | 일반화·비용 대안. 비용을 없앤 정확도에서도 우수하다는 결론은 성립하지 않음 |

A는 R0/R1/R2를, G1은 R1/R3×natural/balanced를 비교했다. 학습 분할·후보 pool·지표가 다르고 xRouteBench high 후보도 Cogito에서 Maverick으로 바뀌었다. 따라서 A와 G1의 nAUC를 한 열에 합쳐 R0~R3의 통제된 최종 순위를 만들 수 없다. 아래 추천은 현재 증거에 근거한 개발 우선순위이며 학술적 전체 우승 선언이나 사용자 최종 채택 결정이 아니다.

## 비용·품질을 함께 본 결과

A의 D_dev nAUC는 R1 .863179 > R2 .833874 > R0 .166115였다. A의 고정 test 기본 정책은 다음과 같았다.

| 방법 | source-macro 품질 | source-macro 상대비용 |
|---|---:|---:|
| R0 | .656886 | .112373 |
| R1 | .789141 | .513201 |
| R2 | .762016 | .340807 |

R1의 높은 품질에는 더 높은 비용이 동반됐다. R1/R2는 xRouteBench test에서 기본 예산 .5를 넘었다. R2를 모든 비용 조건에서 열등하다고 볼 수는 없다. R0는 A test의 두 출처 모두 always-medium보다 품질이 낮고 비용이 높았다.

G1 selection의 고정 정책 AUC는 R1-natural .814739, R1-balanced .809377, R3-natural .463653, R3-balanced .470166이었다. R1-balanced는 xRouteBench 예산 .5를 초과해 선정 대상에서 제외됐고, 잠긴 선정은 R1-natural이다. G1 holdout에서는 xRouteBench high 품질이 low보다 낮아 네 macro AUC가 모두 N/A다. 실제 품질·비용으로 해석한다.

## 정확도: 무엇을 맞히는가

### 실제 선택 후보의 답변 품질

G1 holdout 1,807개는 quality=success인 0/1 관측을 사용하므로 아래 λ=0 품질을 source-macro 성공률로 읽을 수 있다. 각 source를 같은 비중으로 평균한 값이며 전체 문항의 단순 정답률은 아니다. Selection에는 연속 quality도 있어 quality와 성공률을 구분한다.

| 방법 | 비용 감점 0의 holdout 성공률 | 기존 기본 정책의 holdout 품질 | 기본 상대비용 | 모든 source 예산 .5 |
|---|---:|---:|---:|---|
| R1-natural | 68.5145% | .652622 | .412801 | 실패 |
| R1-balanced | **69.9084%** | .669544 | .567213 | 실패 |
| R3-natural | 68.4976% | **.684976** | .374278 | 통과 |
| R3-balanced | 68.3398% | .682972 | **.359348** | 통과 |

추론 비용 감점을 없애면 R1-balanced가 가장 높고 R1-natural과 R3-natural은 0.0169%p 차이로 거의 같다. 따라서 기본 비용 정책 아래 R3가 좋았던 결과를 순수 정확도의 절대적 우위로 일반화할 수 없다. R3는 학습 목표 자체에 비용이 들어 있으므로 λ=0 진단이 비용의 학습 영향을 제거하는 실험은 아니다.

A의 D_dev에서도 λ=0 관측 품질은 R1 .779951 > R2 .777264 > R0 .638046이었다. R1–R2 차이는 .002687로 작으며 통계적 우월성을 주장하지 않는다. A와 G1의 절대 점수 차이는 개선률로 해석하지 않는다.

### 난이도 라벨 일치

같은 Arena prepared manifest와 8,379개 질문에 대한 관측이다. R0는 난이도 분류, 다른 방법은 비용 없는 후보 action을 난이도 슬롯으로 투영했으므로 알고리즘의 공정한 동일 목표 순위는 아니다.

| 학습 모델 | Arena 라벨 일치율 | Macro-F1 |
|---|---:|---:|
| R0, A | **45.48%** | **.3961** |
| R1, A | 22.80% | .1754 |
| R2, A | 21.36% | .1927 |
| R1-natural, G1 | 23.57% | .1953 |
| R1-balanced, G1 | 21.04% | .1726 |
| R3-natural, G1 | 31.58% | .3149 |
| R3-balanced, G1 | 31.35% | .3094 |

R0가 이 목적에는 가장 잘 맞지만 always-easy 일치율 47.49%보다 낮고 hard 재현율은 약 15.02%다. R0의 macro-F1은 그 상수 기준선보다 높으나 충분한 난이도 인식 성능이 입증된 것은 아니다. Arena 지표는 선택한 실제 LLM의 응답 품질·비용 또는 Claude 성능이 아니다.

## Natural과 balanced가 바꾼 것

두 view는 서로 다른 질문 데이터셋이 아니다. 같은 18,591개 질문·후보 outcome·순서를 쓰며 train loss weight만 다르다. Easy/medium/hard 개수를 맞춘 것도 아니고, 적은 source/task의 손실 비중을 높인 것이다. Weight 상한은 3이라 출처를 완전한 50:50으로 만들지 않았다.

| train의 loss 가중치 비중 | Natural | Balanced |
|---|---:|---:|
| RouterBench | 89.38% | 68.13% |
| xRouteBench | 10.62% | 31.87% |

R1의 balancing은 selection 품질·비용 조합을 악화시켰지만 holdout의 λ=0 성공률은 +1.3939%p 높였다. RouterBench MBPP는 .593156→.711027로 개선됐으나 xRouteBench MBPP는 .115880→.111588로 낮아져 모든 과제의 개선은 아니다. 기본 정책 holdout 품질도 +.016923 높아졌지만 상대비용은 .412801→.567213으로 증가했고 두 view 모두 xRouteBench 예산을 위반했다. 따라서 정확도만 우선하면 R1-balanced가 후보지만 종합 기본값으로 balanced를 추천할 근거는 부족하다.

R3 balancing은 기본 selection 품질 +.000673·상대비용 −.055686으로 소폭 이득이었으나 holdout λ=0 성공률은 −.1577%p, 기본 정책 품질은 −.2004%p였다. 기본 macro 상대비용은 .374278→.359348로 줄었다. 작은 품질 손실과 비용 감소의 교환관계이며, 모든 source에서 실제 비용이 줄어든 것은 아니다. Arena 변형 쌍 같은 선택 비율은 natural 51.70%→balanced 59.95%로 높아졌지만 라벨 정확도/F1은 낮아졌다. 같은 행동을 반복하는 것 자체가 정답을 의미하지 않는다.

R0/R2에는 이번 balanced 비교가 없어 효과를 추정하지 않는다. 한 seed의 점추정 차이이며 여러 seed의 안정적 우위를 증명하지 않는다. 학습에 없는 코드 생성 문항을 loss weight 변경으로 새로 학습한 것처럼 설명하지 않는다.

## 실제 라우터 추천

학습 방식·개발 기본값은 **R1 + natural**을 추천한다. 기존 A와 G1 selection에서 반복해서 좋은 비용–품질 선택 결과를 보였고, balanced보다 비용 부담이 작으며, 후보별 관측 성공 신호를 학습하는 목적이 실제 라우팅에 직접 대응한다. R3-natural은 미관측 과제의 비용 준수 대안으로 계속 비교한다. 정확도만 최우선으로 요구하는 별도 운영 조건이라면 R1-balanced를 후보로 검증한다.

현재 가중치의 즉시 배포는 권고하지 않는다. xRouteBench holdout에서는 always-low와 always-medium이 네 G1 라우터보다 품질이 높고 비용이 낮았고, HumanEval 거의 전부 실패·MBPP의 high 품질 반전은 원천 채점 조건 감사를 요구한다. R3도 이 문제를 해결하지 못했다. 서비스의 실제 모델 후보를 관측한 데이터로 단일후보 기준선·품질·비용·새 독립 평가를 통과해야 한다. 공개 pool의 low/medium/high를 Haiku/Sonnet/Opus로 이름만 바꾸어 이전하지 않는다.

재학습 경로에는 후보별 실제 관측 성공·판정 기준·mask가 필요하다. 현재 finalLabel만으로 R1을 학습하거나 호출하지 않은 후보를 실패로 채우지 않는다. 후보 방식이 정해지면 기존 사용자 계획의 R0 및 채택 방식별 main 분기·Spring 변경 목록·FastAPI score/policy·승격/rollback 검증을 이어간다. 이번 분석은 원래 연구 선정과 운영 인수를 대체하지 않는다.

## 근거

- [A의 고정 test](../routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md)
- [G1 완료 보고서](../routing-generalization-v1-20260925/report.md)
- [A 수치·공통 Arena 입력 검증](a-evidence.json)
- [G1 λ=0 재계산 및 source/task 표](g1-zero-penalty.md), [검증 JSON](g1-zero-penalty.json)
- [실제 학습 가중치 집계](weight-views.json)

λ=0에서 원본 route_actions와 독립 NumPy raw-logit argmax가 전 행에서 일치했다. 기존 선정 lock은 그대로이며 이 후속 진단으로 λ나 arm을 재선정하지 않았다.
