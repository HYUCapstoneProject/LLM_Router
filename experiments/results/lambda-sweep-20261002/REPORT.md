# λ 정밀 탐색: R0~R3 비용–품질 곡선과 기준별 최적점

실험 코드: **EV-03** · [실험 목록](../../REGISTRY.md)

2026-10-02. 기존에 저장된 라우터 점수만 사용했다. 재학습, GPU 사용, 신규 LLM 호출은 없다. 기존 selection lock과 봉인 결과는 수정하지 않았다.

## 무엇을 했나

라우팅 규칙은 기존과 같다.

```text
action(q) = argmax_t [ score_t(q) − λ × decision_cost_t ]
```

여기서 score는 R0이면 softmax, R1이면 sigmoid, R2와 R3이면 raw logit이다. λ 후보는 0과 `10^-4 ~ 10^2` 구간의 로그 간격 601점이다. 기존 grid는 13~15점이었다. λ는 **원천(source)마다 따로** 고른다. **고르는 데 쓴 데이터와 성능을 보고하는 데이터는 분리했다.**

| 실험 | 대상 | λ를 고른 데이터 | 성능을 보고한 데이터 |
|---|---|---|---|
| A | R0(CE), R1(success BCE), R2(pairwise) | D_dev 3,581 | 봉인 test 7,202 (RB 3,473 / xRB 3,729) |
| G1 | R1-natural/balanced, R3-natural/balanced | calibration 2,275 | selection 2,314, task holdout 1,807 |

**기준** (tuning 데이터에서 판정):
- `λ=0`: 비용을 보지 않는다.
- `budget ≤ .5`: 상대 비용이 0.5 이하인 λ 가운데 품질이 가장 높은 값.
- `95% / 98%`: always-high 품질의 95% 또는 98% 이상을 유지하는 λ 가운데 상대 비용이 가장 낮은 값.

상대 비용은 `(선택한 답의 평균 실제 비용 − always-low 비용) / (always-high 비용 − always-low 비용)`이다. 원천별로 계산하고 표에서는 두 원천을 같은 비중으로 평균했다.

## 결과: 평가 데이터, source-macro 평균

| 실험 | 방법 | 기준 | λ (RB / xRB) | 품질 | 상대 비용 | low / med / high 비율 |
|---|---|---|---|---:|---:|---|
| A·test | R0 | λ=0 | 0 / 0 | .657 | .113 | .64 / .28 / .07 |
| | R0 | budget .5 | 0 / .003 | .657 | .112 | .64 / .28 / .07 |
| | R0 | 95%, 98% | 달성 불가 | – | – | – |
| | R1 | λ=0 | 0 / 0 | .808 | .728 | .06 / .25 / .69 |
| | **R1** | **budget .5** | **.100 / .041** | **.787** | **.486** | .11 / .47 / .42 |
| | R1 | 98% | .079 / .120 | .780 | .394 | .17 / .50 / .33 |
| | R1 | 95% | .166 / .776 | .743 | .184 | .36 / .50 / .14 |
| | R2 | λ=0 | 0 / 0 | .807 | .736 | .09 / .21 / .70 |
| | R2 | budget .5 | .316 / .093 | .790 | .525 | .13 / .41 / .46 |
| | R2 | 98% | .246 / .407 | .774 | .371 | .22 / .46 / .32 |
| | R2 | 95% | .468 / 1.86 | .743 | .201 | .35 / .49 / .15 |
| G1·selection | R1-natural | λ=0 | 0 / 0 | .764 | .593 | .11 / .30 / .59 |
| | **R1-natural** | **budget .5** | **.083 / .013** | **.751** | **.426** | .15 / .47 / .39 |
| | R1-natural | 98% | .071 / .129 | .739 | .304 | .24 / .49 / .26 |
| | R1-natural | 95% | .214 / .263 | .722 | .170 | .32 / .55 / .12 |
| | R1-balanced | budget .5 | .087 / .007 | .752 | .487 | .12 / .42 / .46 |
| | R1-balanced | 98% | .069 / .089 | .746 | .328 | .23 / .49 / .28 |
| | R1-balanced | 95% | .200 / .195 | .723 | .172 | .28 / .59 / .12 |
| | R3-natural | λ=0 | 0 / 0 | .701 | .312 | .44 / .29 / .27 |
| | R3-natural | budget .5 | 0 / .074 | .703 | .291 | .45 / .29 / .26 |
| | R3-balanced | budget .5 | .013 / .047 | .702 | .255 | .48 / .29 / .23 |
| | R3 둘 다 | 95%, 98% | RB에서 달성 불가 | – | – | – |

원천별 전체 값과 task holdout 결과는 [macro-summary.csv](macro-summary.csv)와 [chosen-points.csv](chosen-points.csv)에 있다. 곡선 전체는 [curves.csv](curves.csv)에 있다.

### 그림
- 비용–품질 곡선과 기준별 위치: [A·test](figures/a-test-tradeoff.png), [G1·selection](figures/g1-selection-tradeoff.png), [G1·task holdout](figures/g1-task_holdout-tradeoff.png)
- λ에 따른 easy/medium/hard 비율: [A·test](figures/a-test-tier-share.png), [G1·selection](figures/g1-selection-tier-share.png), [G1·task holdout](figures/g1-task_holdout-tier-share.png)

## 해석

1. **R1과 R2는 λ로 비용–품질을 넓게 조절할 수 있다.** λ를 0에서 키우면 high→medium→low 순서로 선택이 옮겨 간다(tier-share 그림). A test에서 R1은 비용을 .73에서 .39로 46% 줄이면서 품질을 .808에서 .780으로 3.4%만 잃었다(98% 기준). RouterBench에서 비용 대비 효율이 가장 좋은 구간은 상대 비용 0.05~0.1 근처로, 대부분 medium으로 보낼 때다.

2. **촘촘한 grid는 A의 xRouteBench 예산 초과 문제를 풀었다.** 기존 봉인 test에서는 xRB에서 λ=.03을 썼고, 상대 비용이 .537로 상한 .5를 넘었다. 이번에는 λ=.041로 비용 .483에 품질 .788을 얻어 상한을 지켰다. 기존 품질은 .791이었다. 반면 G1 R1-natural selection에서는 기존 lock(품질 .752, 비용 .427)과 거의 같았다. 이미 기존 grid로도 최적점 근처에 있었다는 뜻이다.

3. **R0는 λ로 거의 움직이지 않는다.** softmax 출력은 easy 쪽으로 치우쳐 있다(low 64%). λ를 0 이상으로 두면 더 싼 쪽으로만 이동하므로, 품질 상한이 λ=0 지점(.657)에서 정해진다. 95%와 98% 유지 기준은 달성할 수 없다. 이 결과는 R0가 비용을 줄이는 쪽으로는 조절이 되지만 품질을 끌어올리는 손잡이가 없다는 뜻이다.

4. **R3는 학습 단계에서 이미 비용을 반영했기 때문에 λ의 효과가 작다.** λ=0에서도 상대 비용이 .26~.31 수준이다. RouterBench에서는 λ를 0으로 둬도 always-high 품질의 95%에 이르지 못한다. 높은 품질이 필요하면 R3 학습 자체의 λ grid를 바꿔야 한다. 추론 λ로는 해결되지 않는다.

5. **xRouteBench에서는 always-high가 가장 좋은 후보가 아닐 때가 있다.** G1 selection에서 always-medium(gpt-oss-120b)의 품질은 .745로 always-high(.738)보다 높고, 상대 비용은 .10에 불과하다. A의 xRB에서도 medium과 high의 차이가 작다(D_dev .719 vs .738). 그래서 95%·98% 기준이 낮은 비용에서 곧바로 충족된다. G1 task holdout의 xRB에서는 always-high 품질(.498)이 always-low(.704)와 always-medium(.735)보다 낮다. 이 구간에서는 비용을 많이 쓸수록 품질이 떨어진다. "비싼 모델 = 좋은 모델"이라는 전제가 원천과 과제에 따라 깨진다는 점을 서비스 후보를 정할 때 확인해야 한다.

6. **tuning과 평가의 차이.** RouterBench는 곡선이 거의 겹친다. xRouteBench는 표본이 237~634건으로 작아 곡선이 계단 모양이고, tuning에서 고른 λ의 효과가 평가 데이터에서 그대로 재현되지 않는 경우가 있다. xRB 수치는 RB보다 신뢰도가 낮다.

## 결론

- **R1의 기준별 추천 λ (A, 원천별):** budget .5는 RB .10 / xRB .041이고, 98% 유지는 RB .079 / xRB .12다. G1 R1-natural에서는 각각 RB .083 / xRB .013과 RB .071 / xRB .129다.
- 원천마다 λ가 다르다. 실제 서비스에서는 원천을 입력으로 알 수 없으므로 Claude 후보(Haiku/Sonnet/Opus) 기준으로 다시 정해야 한다. 이번 값은 그때 출발점과 기대 곡선의 모양을 알려 주는 참고 자료다.
- 단일 seed이고 bootstrap 신뢰구간을 계산하지 않았다. 방법 간 우열을 통계적으로 주장하지 않는다.

## 재현

```bash
# A 점수 원본 (GCS, 읽기 전용)
gcloud storage cp gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/comparison-results.tar a/ && tar xf a/comparison-results.tar -C a
gcloud storage cp gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/sealed-tests-v1/sealed-inference.tar sealed/ && tar xf sealed/sealed-inference.tar -C sealed
.venv/bin/python experiments/results/lambda-sweep-20261002/lambda_sweep.py --a-dir a --a-test-dir sealed/inference
.venv/bin/python experiments/results/lambda-sweep-20261002/plot.py
```

모든 입력 파일의 SHA-256은 [manifest.json](manifest.json)에 있다. G1 입력은 저장소 안의 `routing-generalization-v1-20260925` 파일이고, A test의 결과(outcome)는 sealed-inference 점수 파일에 함께 들어 있다.
