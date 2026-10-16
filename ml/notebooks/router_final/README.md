# 라우터 최종 실험 노트북 (RT-08 · EV-10)

기준 문서: [라우터 최종 실험 계획서](../../../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) · 실험 코드: RT-08, EV-10

| 노트북 | 상태 | 하는 일 |
|---|---|---|
| `00_prepare` | 구현 | 데이터 준비. 아래 참고 |
| `01_embed` | 구현 | BGE-large-en-v1.5(revision d4aa6901) 임베딩. Colab T4 |
| `02_rt08` | PR #65 | RT-08 |
| `03a_r0_group` · `03b_r1_group` · `03c_r1_group_a` | 구현 | 우리 라우터 학습. 서로 독립이라 다른 GPU 세션에서 동시에 실행 가능 |
| `04_common_track` ~ `10_report` | 예정 | 계획서 4.5절 |

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

## 01_embed

`embed.py`가 00_prepare 출력의 SHA를 먼저 확인한 뒤, train·dev·test 질문과 RouterArena 질문(`routerarena:<Global Index>`)을 512토큰, 정규화, float16으로 임베딩한다. 출력은 `embeddings.npy`, 행 순서가 같은 `ids.parquet`, `manifest.json`(모델 revision, 입력·출력 SHA, 역할별 행 수)이다.

Drive 구성(`LLM_Router/router-final/runs/run001/`): `code/`(embed.py, requirements-colab.txt), `prepare/`(00_prepare 출력), `inputs/routerarena-full.parquet`. 노트북은 결과를 `embed/`에 쓰고, 이미 있으면 멈춘다.

## 03_ours (03a · 03b · 03c)

`ours.py build`(로컬)가 00_prepare의 train·dev로 두 학습 세트를 만든다. `blended`는 원천 4개, `a_only`는 RouterBench·xRouteBench만이고, 둘 다 원천마다 같은 총 가중치를 준다. `ours.py train`(Colab)은 RT-05의 `train_objective`와 `routing_objective_continual_v1.json`을 그대로 쓰며 seed는 42다.

| 노트북 | 방법 | 데이터 | 정답 |
|---|---|---|---|
| 03a | R0-집단 (ordinal CE) | blended | 3단계 라벨(low/medium/high → easy/medium/hard) |
| 03b | R1-집단 (pointwise BCE) | blended | 집단별 풀 확률(0~1) |
| 03c | R1-집단-A | a_only | 집단별 풀 확률(0~1) |

- 비용 열(`decision_cost_*`, `actual_cost_*`)은 서비스 비용 .2 / .6 / 1.0이다.
- **빈 단계 확률:** LLMRouterBench의 medium 집단은 모델이 하나(Qwen3-235B-Thinking)여서, 그 호출 기록이 없는 문항(train 444, dev 136, 74%가 hard)은 medium 확률이 비어 있다. 빼면 어려운 문항만 골라 빠지므로, 학습에서는 양옆 단계 평균으로 채우고(`tier_imputed`) 3단계 라벨도 다시 계산한다. 평가에서는 실제 기록이 없는 행을 쓰지 않는다.
- 결과는 Drive `ours/models/<run>/`에 쓴다(model, validation-scores.parquet, metrics.json). D_dev 점수는 07_lock 보정에 쓴다.
- Colab 소스는 02와 겹치지 않게 `router-final-source-ours.zip`으로 따로 둔다.
