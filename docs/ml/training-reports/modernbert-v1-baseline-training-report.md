# ModernBERT v1 Baseline Training Report

작성일: 2026-08-15
대상 run: `modernbert-router-baseline-v1`
저장 위치: `MyDrive/LLM_Router/training/modernbert-router-baseline-v1/`
상태: **epoch 1에서 중단 (미완료)**

> 이 문서는 v1 run의 관측 결과와 v2로 전환한 근거를 기록한다. v1 라벨 규칙의
> 정의는 `docs/ml/routerbench-data-labeling-spec.md` §13이 소유한다. v2 결과는
> `modernbert-v2-balanced-training-report.md`를 참조한다.

## 1. 요약

v1은 3 epoch(10,428 step) 계획 중 **epoch 1(3,476 step)까지만 수행되고 중단**되었다.
`final-model`이 없으므로 완료된 학습 artifact는 존재하지 않는다.

핵심 관측은 **medium 클래스의 완전 붕괴**다.

| 관측 | 값 |
| --- | --- |
| medium F1 추이 (epoch 1 내부) | 0.015 → 0.004 → **0.000** |
| RouterArena에서 medium으로 예측한 건수 | **0건** (8,375행 중) |
| 실질 동작 | 3-class 모델이 아니라 easy/hard 2-class 분류기 |

단, medium을 제외하면 v1의 easy/hard 분리 성능은 오히려 v2보다 좋았다(§6). 이
사실이 v2 설계 판단의 핵심 근거이자, 동시에 v2가 놓친 부분이기도 하다.

## 2. 학습 설정과 중단 지점

`checkpoints/step-00003476/checkpoint_manifest.json` 기준이다.

| 항목 | 값 |
| --- | --- |
| training fingerprint | `86469dd6…7283` |
| training data SHA-256 | `eb67fb20…6dfc` |
| 계획 step | 10,428 (3 epoch) |
| 완료 step | **3,476 (epoch 1 종료 지점)** |
| 진행률 | 33.3% |
| 남은 checkpoint | `step-00003000`, `step-00003476` |
| `final-model` | **없음** |

epoch 1 종료 시점에 정확히 멈췄고 checkpoint는 정상 봉인(`_SUCCESS` + manifest)되어
있다. 즉 세션 유실로 인한 손상이 아니라, epoch 1 결과를 확인한 뒤 v2로 전환하기
위해 의도적으로 중단한 것으로 보인다.

step 수 계산: 자연분포 train 27,808행 / effective batch 8 = epoch당 3,476 step.

## 3. 데이터 — v1은 자연분포로 학습했다

라벨 규칙 `minimum-sufficient-tier-v1`, 전처리 `routerbench-training-preprocess-v1`.

| 항목 | 결과 |
| --- | ---: |
| 입력 / 라벨 완료 | 34,778 / 34,778 |
| easy / medium / hard | 12,765 / 5,348 / **16,665** |
| 학습 포함 / 제외 | 34,764 / 14 (low confidence weight 0) |
| train / validation / test | 27,808 / 3,478 / 3,478 |

validation split의 클래스 구성과 비중:

| 클래스 | support | 비중 |
| --- | ---: | ---: |
| easy | 1,277 | 36.7% |
| medium | **535** | **15.4%** |
| hard | 1,666 | 47.9% |

**medium이 전체의 15.4%에 불과하다.** v1은 균형 샘플링 없이 이 자연분포를 그대로
학습했고, 그 결과가 §4의 medium 붕괴다. majority-class baseline accuracy는
`1,666 / 3,478 = 47.9%`(hard)다.

## 4. RouterBench validation 관측

epoch 1 내부 33%/66%/100% 지점의 3개 관측점이다. v1 스키마
(`routerbench-validation-progress-v1`)에는 train loss 필드가 없다.

| obs | step | val loss | val acc | macro F1 | easy F1 | medium F1 | hard F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,159 | 0.7885 | **0.6325** | **0.4621** | 0.644 | 0.015 | 0.728 |
| 2 | 2,317 | 0.7913 | 0.6302 | 0.4586 | 0.658 | 0.004 | 0.714 |
| 3 | 3,476 | 0.7931 | 0.6259 | 0.4456 | 0.606 | **0.000** | 0.730 |

관찰:

1. **accuracy 62.6~63.3%는 majority baseline 47.9%를 크게 상회한다.** easy/hard
   구분에 대해서는 실제로 유의미한 학습이 일어났다.
2. **medium은 학습이 진행될수록 완전히 소멸한다.** F1이 0.015 → 0.004 → 0.000으로
   단조 감소하며, 마지막 관측에서는 medium을 단 한 건도 맞히지 못했다.
3. **easy/hard만 보면 성능이 준수하다.** 두 클래스 F1 평균은
   `(0.606 + 0.730) / 2 = 0.668`이다.
4. macro F1이 0.4621 → 0.4456으로 하락하는 것은 전적으로 medium 소멸 때문이다.
5. validation loss가 0.7885 → 0.7931로 미세하게 상승하고 있어, epoch 1 종료 시점에
   이미 개선이 멈춘 신호가 보인다.

**진단:** 자연분포에서 minority 클래스(15.4%)를 균형 보정 없이 학습하면, 모델이
medium을 아예 예측하지 않는 것이 loss 관점에서 유리해진다. 전형적인 class collapse다.

## 5. RouterArena transfer 결과 (epoch 1)

| 지표 | 값 |
| --- | ---: |
| 평가 행 | 8,375 |
| accuracy | 0.3827 |
| macro F1 | 0.3011 |
| macro precision / recall | 0.2976 / 0.3979 |
| under-routing rate | 0.1207 |
| over-routing rate | **0.4966** |
| 평균 tier 오차 | 0.9434 |
| truncated 행 | 5 |

클래스별 F1: easy 0.5092, hard 0.3942, **medium 0.0000**

예측 분포: easy 2,713 / medium **0** / hard 5,662

### 5-1. 예측 그룹 내부 라벨 분포

| 예측 그룹 | n | easy | medium | hard |
| --- | ---: | ---: | ---: | ---: |
| predicted easy | 2,713 | 1,702 (62.7%) | 550 (20.3%) | 461 (17.0%) |
| predicted medium | **0** | – | – | – |
| predicted hard | 5,662 | 2,270 (40.1%) | 1,889 (33.4%) | 1,503 (26.5%) |

RouterArena base rate(easy 47.4%, medium 29.1%, hard 23.5%) 대비:

- `predicted easy → easy` **+15.3pp** — v2의 +11.2pp보다 강한 신호
- `predicted hard → hard` **+3.1pp** — 약하지만 v2의 −1.6pp와 달리 양(+)의 방향
- medium 예측군 자체가 존재하지 않음

## 6. v1 대 v2 직접 비교

RouterBench validation은 라벨 규칙이 달라(v1 자연분포 vs v2 균형) 직접 비교할 수
없다. 그러나 **RouterArena는 동일한 8,375행, 동일한 공식 라벨**이므로 직접 비교가
가능하다.

| run | epoch | accuracy | macro F1 | under-routing | over-routing | 평균 tier 오차 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **v1** | 1 | **0.3827** | 0.3011 | **0.1207** | 0.4966 | 0.9434 |
| v2 | 1 | 0.3297 | 0.3014 | 0.1706 | 0.4997 | 0.9506 |
| v2 | 2 | 0.3582 | **0.3293** | 0.1781 | 0.4636 | 0.9100 |
| v2 | 3 | 0.3528 | 0.3212 | 0.1999 | **0.4473** | 0.9181 |

관찰:

1. **accuracy는 v1(epoch 1 하나만으로)이 모든 v2 epoch보다 높다.** v1이 예측을
   easy/hard에만 집중했고, 이 두 클래스가 RouterArena 라벨의 70.9%를 차지하기
   때문이다.
2. **macro F1은 v2 epoch 2가 가장 높다(0.3293).** 균형 학습이 medium을 살려낸
   효과가 macro 지표에는 반영되었다.
3. **under-routing은 v1이 유리하고(0.1207) v2는 악화된다(0.17~0.20).** 라우팅
   품질 관점에서는 중요한 손실이다. 반대로 over-routing은 v2 epoch 3이 가장 낮다.
4. **easy 판별 신호는 v1이 더 강했다** (+15.3pp vs v2 +11.2pp).

**종합 판단:** v2의 균형 학습은 medium 붕괴라는 v1의 명백한 결함을 해결했지만,
그 대가로 v1이 갖고 있던 easy/hard 분리 능력을 상당 부분 잃었다. 두 run 모두
RouterArena에서 macro F1이 균등추측 기대값(0.326) 근처에 머물러 있어, **어느
쪽도 transfer에 성공했다고 볼 수 없다.**

## 7. v1이 남긴 결론

1. **자연분포 그대로 학습하면 안 된다.** minority 클래스 15.4%는 균형 보정 없이는
   소멸한다. v2의 `balanced_effective_batch` 도입은 타당한 판단이었다.
2. **그러나 균형 학습만으로는 부족했다.** v2 보고서 §6~8이 보여주듯 근본 병목은
   클래스 불균형이 아니라 라벨 정의의 신호 부족이다. v1의 medium 붕괴는 그 병목의
   가장 눈에 띄는 증상이었을 뿐, 원인이 아니었다.
3. **v1의 easy/hard 성능은 회수할 가치가 있다.** 향후 라벨 재정의 실험에서 v1
   수준의 easy 판별력(+15.3pp)과 낮은 under-routing(0.12)을 유지하면서 medium을
   살리는 것이 목표가 되어야 한다.

## 8. Artifact 상태

| 항목 | 상태 |
| --- | --- |
| `final-model` | **없음 (학습 미완료)** |
| checkpoints | `step-00003000`, `step-00003476` (둘 다 `_SUCCESS` 봉인) |
| RouterBench validation 관측 | observation-01 ~ 03 (epoch 1분) |
| RouterArena epoch 관측 | epoch-01 만 존재 |
| 재개 가능 여부 | 가능 — `step-00003476`에서 `--resume-mode auto`로 epoch 2 재개 |

v1을 재개하려면 v1 config와 `training.parquet`(SHA-256 `eb67fb20…6dfc`)이 필요하며,
fingerprint가 v2와 다르므로 반드시 `modernbert-router-baseline-v1` 경로를 사용해야
한다. 다만 §7의 판단에 따라 **v1 재개보다는 라벨 재정의 실험이 우선순위가 높다.**
