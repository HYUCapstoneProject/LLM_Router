# 라우터 최종 실험 노트북 (RT-08 · EV-10)

기준 문서: [라우터 최종 실험 계획서](../../../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) · 실험 코드: RT-08, EV-10

| 노트북 | 상태 | 하는 일 |
|---|---|---|
| `00_prepare` | 구현 | 데이터 준비. 아래 참고 |
| `01_embed` | 구현 | BGE-large-en-v1.5(revision d4aa6901) 임베딩. Colab T4 |
| `02_rt08` | PR #65 | RT-08 |
| `03a_r0_group` · `03b_r1_group` · `03c_r1_group_a` | 구현 | 우리 라우터 학습. 서로 독립이라 다른 GPU 세션에서 동시에 실행 가능 |
| `04_common_track` | 구현 | 논문 라우터 공통 트랙과 R1-BGE. 아래 참고 |
| `05_finetune_track` ~ `10_report` | 예정 | 계획서 4.5절 |

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

## 04_common_track

`common_track.py`가 본체다. `fetch`가 외부 코드를 고정 커밋에서 받아 파일마다 SHA-256을 확인하고, `run`이 원본 클래스를 불러 학습한다. 외부 코드는 저장소에 넣지 않는다.

| 방법 | 원본 | 설정(≤ 8) | 점수 |
|---|---|---|---|
| `knn` · `knn_a` | RouterBench `KNNRouter` | 이웃 5·10·40(원본 평가 설정)·50(기본값) | 이웃의 단계별 평균 성공 |
| `mlp` | RouterBench `MLPRouter` | 은닉층 (100), (100,100), (100,100,100) | 단계별 회귀값 |
| `svm` | RouterBench `SVMRouter` | 기본값 하나(LinearSVR) | 단계별 회귀값 |
| `mf` | LLMRouter `MFRouter`·`MFRouterTrainer` | epoch 5(YAML)·3(기본값) | 질문 × 단계 궁합 점수 |
| `graphrouter` | LLMRouter `GraphRouter`·`GraphTrainer` | 원본(YAML, 합 집계, 원본 체크포인트)과 부분 차용(평균 집계, 마지막 epoch) | 질문–단계 연결 점수 |
| `hybrid_llm` | LLMRouter `HybridLLMRouter` | probabilistic(YAML)·deterministic·transformed | low로 충분할 확률(점수 하나) |
| `irt_router` | IRT-Router `MIRT` | 시작 보정 λ 0·.1(기본값)·.3, 학습은 한 번 | 단계별 성공 확률 + 문항 난이도 b |
| `avengers_pro` | Avengers-Pro `BalanceClusterRouter` | 묶음 25·α .45 / .7 / 1, 묶음 64·α .7 | 가까운 묶음 3개의 순위 점수 |
| `r1_bge` | 우리 | 은닉 256·1024 × epoch 5·20 × 원천 가중치 유무 | 단계별 풀 확률 |

- **같은 조건:** 질문 입력은 01_embed BGE 벡터, 학습 행은 03_ours `blended/train.parquet`(35,017행, kNN-A는 `a_only`), 단계 순서는 low → medium → high라 "가장 잘 푼 단계"가 동점이면 싼 단계가 된다. 비용은 서비스 비용 .2/.6/1.0이다.
- **집단 설명:** GraphRouter의 LLM 노드와 IRT-Router의 LLM 벡터는 단계마다 "소속 모델 이름(원천별) + 가격대" 문장을 같은 BGE로 임베딩해 쓴다. 문장은 00_prepare manifest에 봉인된 집단 구성에서 만들고 `group-descriptions.json`에 남긴다.
- **원천 가중치:** 외부 방법은 원본 학습 방식 그대로(가중치 없음) 둔다. R1-BGE만 가중치 유무를 설정으로 둔다.
- **원본과 다른 점:** 결과마다 `metrics.json`의 `deviations`에 적는다. 대표적으로 GraphRouter는 새 질문을 학습 질문으로 맞춘 scaler로 정규화한다(원본은 배치마다 scaler를 다시 맞춰 새 질문의 척도가 달라진다).
- **상수 점수 표시:** 점수가 모든 질문에서 상수면 `degenerate: true`로 기록한다. 라우팅 정보가 없다는 뜻이고, 원인은 결과마다 적는다. 계획서 4.5절대로 억지로 맞추지 않고 결과표에서 구분한다.
- **GraphRouter:** 원본 그대로는 재현 실패다. 단계 노드 하나가 학습 질문 약 2.5만 개의 메시지를 더하므로(합 집계) 첫 업데이트 뒤 예측이 포화되고(BCE ≈ 33), 모든 점수가 0이 된다. 원본 체크포인트 규칙(고른 단계의 성능 평균이 가장 높은 epoch)도 거의 학습 전 모델을 고른다. 그래서 부분 차용 버전을 따로 둔다: `GeneralConv`를 평균 집계로 바꾸고 마지막 epoch 가중치를 쓴다. 두 결과는 결과표에서 구분한다.
- **Avengers-Pro α = 1(성능만):** 모든 묶음에서 순위가 high > medium > low로 같아 점수가 상수다. 구현 문제가 아니라 방법의 결과다.
- **출력:** `common/<방법>/<설정>/`에 `scores.parquet`(sample_id, source_dataset, role, score_low/medium/high, IRT는 irt_difficulty), `metrics.json`(학습·점수 시간, 파라미터 수, 원본 커밋, deviations), `_SUCCESS`. `common/manifest.json`은 입력 SHA, 외부 파일 SHA, 라이브러리 버전, 결과 SHA를 담는다. 설정 선택과 원천별 보정은 07_lock이 dev에서 한다.

로컬 실행(시험):

```bash
python ml/notebooks/router_final/common_track.py fetch --external-dir /tmp/ev10-external
python ml/notebooks/router_final/common_track.py run --run-dir <run001> --external-dir /tmp/ev10-external --output-dir /tmp/common-smoke --smoke
```
