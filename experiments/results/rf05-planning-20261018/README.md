# RF-05 설계 사전 분석: 경계를 그대로 쓰면 얼마나 잃나 (dev)

작성일: 2026-10-18. [최종 계획서](../../../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) 5.5절의 설계 근거다. EV-10 test는 읽지 않았다.

## 무엇을 쟀나

A 분할만 학습한 본체(R1-집단-A, kNN-A)와 섞어 학습한 본체(R1-집단, kNN)를 새 세대 원천(LLMRouterBench, SPROUT)의 dev에서 두 가지 경계로 평가했다.

- **A 경계 그대로(`frozen_A_boundary`):** A dev(RouterBench 3,101 + xRouteBench 379행)를 합쳐 맞춘 단계별 isotonic과 A의 ρ. RF-05의 U0에 해당한다.
- **그 원천 dev로 다시 맞춘 경계(`new_source_boundary_crossfit`):** EV-10 07b와 같은 5분할 교차 보정. RF-05 상한(모든 결과로 다시 맞춘 경계)에 해당한다.

λ는 두 경우 모두 그 원천의 비용 목표(25·50·75%)에 맞췄다. 이득은 EV-10과 같은 정의다(상향 고려 규칙, 같은 비용의 고정 시작 단계 대비, κ 격자와 비용 수준 평균). 평가 제외(영어가 아닌 문항, 문서 제공형)는 07b와 같이 뺐다.

## 결과 (`headroom-dev.csv`)

| 본체 | 원천 | A 경계 그대로 | 다시 맞춘 경계 | A 경계의 확률 오차(예측 − 실제, low/medium/high) |
|---|---|---:|---:|---|
| R1-집단-A | LLMRouterBench | −.080 | −.003 | +.15 / +.16 / +.19 |
| R1-집단-A | SPROUT | +.015 | +.015 | −.17 / −.13 / −.04 |
| kNN-A | LLMRouterBench | −.069 | −.024 | +.12 / +.16 / +.24 |
| kNN-A | SPROUT | +.011 | +.009 | −.23 / −.15 / −.08 |
| R1-집단 | LLMRouterBench | +.066 | +.089 | −.01 / +.03 / +.03 |
| R1-집단 | SPROUT | +.029 | +.027 | −.02 / −.02 / +.01 |

- LLMRouterBench에서는 A 경계가 성공을 15~24%p 과대평가해 너무 낮게 보내고, 고정 시작보다도 나빠진다. 경계를 다시 맞추면 R1-집단-A에서 +.077, kNN-A에서 +.045를 되찾는다.
- SPROUT에서는 경계를 다시 맞춰도 이득이 같다.
- LLMRouterBench dev는 1,150행이고 고정 시작 비용이 low .97, high 1.0으로 가까워 값이 불안정하다(lock 보고서 4절). 같은 비교의 test 값(EV-10 08, 경계는 둘 다 다시 맞춤)은 R1-집단 +.028, R1-집단-A +.019다.

## 재현

```bash
python headroom.py <repo> <Drive run001 경로> headroom-dev.csv
```

입력은 run001의 `prepare/`(dev 결과), `scores-modernbert/`, `common/knn*/`, 저장소의 평가 제외 목록이다. 로컬 맥 CPU로 약 40초.
