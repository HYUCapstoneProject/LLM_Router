# 라우터 최종 실험 노트북 (RT-08 · EV-10)

기준 문서: [라우터 최종 실험 계획서](../../../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) · 실험 코드: RT-08, EV-10

| 노트북 | 상태 | 하는 일 |
|---|---|---|
| `00_prepare` | 구현 | 데이터 준비. 아래 참고 |
| `01_embed` ~ `10_report` | 예정 | 계획서 4.5절 |

## 00_prepare

`prepare.py`가 본체이고 노트북은 실행과 요약만 한다. 설정은 `config.json`이다.

- **A 분할 결과:** 데이터 생성 코드(`routing_data/preprocessing.py`)의 로더를 그대로 불러 모델별 정답 여부를 만든다. v1 설정으로 다시 계산한 해결률이 기존 데이터와 100% 같은지 확인했고(`old_rate_v1_check`), v2는 채점 점검 정정·제외를 반영한다.
- **모델 집단:** 원천마다 같은 가격 이하 최고 성능보다 .03 넘게 낮은 모델을 빼고, 남은 모델을 로그 비용 1차원 k-means로 셋으로 나눈다. A는 기존 train 통계(`candidate-statistics.parquet`), 외부 데이터는 새 train 통계를 쓴다. SPROUT 모델 가격은 `experiments/results/router-final-inputs/sprout-model-cost-calibration.csv`(데이터 저자 단가표, 질문 원문 없음)를 쓴다.
- **질문별 값:** `p_low`·`p_medium`·`p_high`(집단 모델이 푼 비율), `needed_tier`(풀 확률 ≥ .8인 가장 싼 집단, 없으면 가장 높은 집단), `service_cost_*`(.2 / .6 / 1.0), `source_weight`(원천별 같은 비중).
- **RT-08용:** `old_label_v1`, `old_label_v2`(전체 모델 해결률, 경계 .27/.64), 대표 모델 셋의 v2 성공(`arm_success_v2_*`).
- **외부 재분할:** `split_group_id` 해시로 train 50 / dev 15 / test 35. RouterArena와 질문 본문이 겹치는 문항과 LLMRouterBench 판정 선호 진단은 test로 보낸다.
- **출력:** `train.parquet`, `dev.parquet`, `test.parquet`, `groups.json`, `manifest.json`(입력·출력 SHA, 행 수, 집단, train 라벨 비율). 질문 원문이 들어 있으므로 저장소에 넣지 않는다.

로컬 실행:

```bash
.venv/bin/python ml/notebooks/router_final/prepare.py --config ml/notebooks/router_final/config.json --repo . --data-root . --output-dir ml/artifacts/router-final-v1/prepare
```
