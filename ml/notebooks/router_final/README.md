# 라우터 최종 실험 노트북 (RT-08 · EV-10)

기준 문서: [라우터 최종 실험 계획서](../../../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) · 실험 코드: RT-08, EV-10

| 노트북 | 상태 | 하는 일 |
|---|---|---|
| `00_prepare` | 구현 | 데이터 준비. 아래 참고 |
| `01_embed` | 구현 | BGE-large-en-v1.5(revision d4aa6901) 임베딩. Colab T4 |
| `02_rt08` | 구현 | RT-08: 기존 정의 R0를 v2 라벨로 재학습. Colab T4 |
| `03a_r0_group` · `03b_r1_group` · `03c_r1_group_a` | 구현 | 우리 라우터 학습. 서로 독립이라 다른 GPU 세션에서 동시에 실행 가능 |
| `04_common_track` | 구현 | 논문 라우터 공통 트랙과 R1-BGE. 아래 참고 |
| `05a_routerdc` · `05b_routellm_retrained` | 구현 | 인코더까지 학습하는 논문 라우터. 아래 참고 |
| `06_routellm_released` | 구현 | RouteLLM-BERT 공개본 점수 |
| `07a_score_ours` | 구현 | 03·02 ModernBERT 모델의 dev·test·RouterArena 점수 |
| `07b_lock` | 구현 | 설정 선택·보정·결정 규칙 고정. 여기서 멈추고 보고 |
| `08a_latency` | 구현 | 라우팅 지연(CPU 주, T4 참고). 07b 뒤 |
| `08_test` | 구현 | 봉인한 test를 한 번 열어 두 갈래 평가와 가설 판정. Colab CPU. 아래 참고 |
| `09_seeds` | 구현 | H5만 seed 17·73 추가(R1-집단·R1-집단-A 재학습). Colab T4. 아래 참고 |
| `10_report` | 구현 | 08·09·08a 출력으로 표·그림·결과 정리. CPU. 아래 참고 |

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

## 02_rt08 (RT-08)

`rt08.py build`(로컬)가 00_prepare의 A 분할 train·D_dev에서 v2 제외 문항을 빼고, 라벨을 기존 정의의 v2 라벨(전체 모델 해결률, 경계 .27/.64)로 바꿔 `training.parquet`과 manifest(v1·v2 라벨 비율, 바뀐 비율)를 만든다. `rt08.py train`(Colab)은 RT-04와 같은 `run_colab_training`·`modernbert_initial_v1_training.json`·런타임 계약으로 3 epoch를 학습한다. v2 제외 문항이 빠져 1,992 step이 된다(RT-04는 21,406행으로 2,007 step). 2,007은 상한으로만 둔다.

| | train | D_dev |
|---|---:|---:|
| 행 (v2 제외 180행 뺌) | 21,247 | 3,560 |
| 라벨 easy/medium/hard v1 | 49.7 / 37.9 / 12.4% | 49.5 / 38.0 / 12.5% |
| 라벨 v2 | 56.5 / 33.6 / 10.0% | 56.5 / 33.7 / 9.8% |
| 라벨이 바뀐 비율 | RB 7.8%, xRB 13.8% | RB 8.1%, xRB 13.5% |

Colab 소스는 `build_kit.py`가 만드는 `router-final-source.zip`(ml/src·ml/config·이 폴더 코드, 파일별 SHA 포함)이다.

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
| `graphrouter` | LLMRouter `GraphRouter`(단계 노드 3개), GraphRouter 공식 `GNN_prediction`(모델 노드) | LLMRouter 원본 하나 + 공식 구조 셋(논문 정답 1000 epoch, 성공 정답 1000·100 epoch) | 질문–단계 연결 점수, 또는 질문–모델 연결 점수를 끝에서 집단 평균 |
| `hybrid_llm` | LLMRouter `HybridLLMRouter` | probabilistic(YAML)·deterministic·transformed | low로 충분할 확률(점수 하나) |
| `irt_router` | IRT-Router `MIRT` | 시작 보정 λ 0·.1(기본값)·.3, 학습은 한 번 | 단계별 성공 확률 + 문항 난이도 b |
| `avengers_pro` | Avengers-Pro `BalanceClusterRouter` | 묶음 25·α .45 / .7 / 1, 묶음 64·α .7 | 가까운 묶음 3개의 순위 점수 |
| `r1_bge` | 우리 | 은닉 256·1024 × epoch 5·20 × 원천 가중치 유무 | 단계별 풀 확률 |

- **같은 조건:** 질문 입력은 01_embed BGE 벡터, 학습 행은 03_ours `blended/train.parquet`(35,017행, kNN-A는 `a_only`), 단계 순서는 low → medium → high라 "가장 잘 푼 단계"가 동점이면 싼 단계가 된다. 비용은 서비스 비용 .2/.6/1.0이다.
- **집단 설명:** GraphRouter의 LLM 노드와 IRT-Router의 LLM 벡터는 단계마다 "소속 모델 이름(원천별) + 가격대" 문장을 같은 BGE로 임베딩해 쓴다. 문장은 00_prepare manifest에 봉인된 집단 구성에서 만들고 `group-descriptions.json`에 남긴다.
- **원천 가중치:** 외부 방법은 원본 학습 방식 그대로(가중치 없음) 둔다. R1-BGE만 가중치 유무를 설정으로 둔다.
- **원본과 다른 점:** 결과마다 `metrics.json`의 `deviations`에 적는다. 대표적으로 GraphRouter는 새 질문을 학습 질문으로 맞춘 scaler로 정규화한다(원본은 배치마다 scaler를 다시 맞춰 새 질문의 척도가 달라진다).
- **상수 점수 표시:** 점수가 모든 질문에서 상수면 `degenerate: true`로 기록한다. 라우팅 정보가 없다는 뜻이고, 원인은 결과마다 적는다. 계획서 4.5절대로 억지로 맞추지 않고 결과표에서 구분한다.
- **GraphRouter:** 구현 두 가지를 둔다.
  - LLMRouter 판(단계 3개를 노드로)은 재현 실패다. 단계 노드 하나가 학습 질문 약 2.5만 개의 메시지를 더하므로(합 집계) 첫 업데이트 뒤 예측이 포화되고(BCE ≈ 33) 모든 점수가 0이 된다.
  - 공식 저장소 판은 논문처럼 실제 모델을 노드로 두고(모델 풀마다 그래프 하나), 공식 신경망과 하이퍼파라미터(hidden 8, lr 3e-4, 1000 epoch, batch 32, mask .5)를 그대로 쓴다. 3단계는 맨 끝에서 집단 모델들의 예측 평균으로 만든다. 과제 노드 특성은 0(과제 이름 금지)이고, RouterArena 질문은 모델 풀이 없어 점수를 내지 않는다.
  - 논문의 정답(질문마다 one-hot "가장 좋은 모델")은 0/1 결과에서 거의 늘 "가장 싼 모델"이라 정보가 적다(시험 실행에서 dev 상관 약 0). 그래서 같은 신경망으로 질문–모델 성공을 직접 맞히는 부분 차용(마지막 epoch)도 둔다. 시험 실행에서는 원천에 따라 kNN보다 조금 낮거나 높았다.
  - 모델 노드에는 모델별 성공 기록이 필요하다. `build-outcomes`(로컬)가 03_ours 학습 행만으로 `run001/common-inputs/`를 만든다. 빈 기록(LLMRouterBench 513칸)은 그 행의 집단 풀 확률로 채우며, 집단 평균이 R1-집단 정답과 같은지 확인한다.
- **Avengers-Pro α = 1(성능만):** 모든 묶음에서 순위가 high > medium > low로 같아 점수가 상수다. 구현 문제가 아니라 방법의 결과다.
- **출력:** `common/<방법>/<설정>/`에 `scores.parquet`(sample_id, source_dataset, role, score_low/medium/high, IRT는 irt_difficulty), `metrics.json`(학습·점수 시간, 파라미터 수, 원본 커밋, deviations), `_SUCCESS`. `common/manifest.json`은 입력 SHA, 외부 파일 SHA, 라이브러리 버전, 결과 SHA를 담는다. 설정 선택과 원천별 보정은 07_lock이 dev에서 한다.

Colab 실행 전 로컬에서 한 번(GraphRouter 모델 노드용, 원본 데이터가 있는 checkout 필요):

```bash
python ml/notebooks/router_final/common_track.py build-outcomes --run-dir <run001> --repo . --data-root <원본 데이터 checkout> --output-dir <run001>/common-inputs
```

로컬 실행(시험):

```bash
python ml/notebooks/router_final/common_track.py fetch --external-dir /tmp/ev10-external
python ml/notebooks/router_final/common_track.py run --run-dir <run001> --external-dir /tmp/ev10-external --output-dir /tmp/common-smoke --smoke
```

## 05_finetune_track (05a · 05b) · 06_routellm_released

`finetune_track.py`가 본체다. 04처럼 외부 코드를 고정 커밋에서 받아 파일마다 SHA-256을 확인하고(`fetch`), 원본 클래스를 불러 학습한다(`run --method ...`). 학습 행은 03·04와 같은 `blended/train.parquet`(35,017행)이고, 입력은 질문 원문이다. 결과는 04와 같은 형식으로 Drive `finetune/<방법>/<설정>/`에 쓴다(`scores.parquet`, `metrics.json`, `_SUCCESS`, 방법별 `manifest.json`). 체크포인트는 Colab 로컬(`/content/finetune-work`)에만 둔다.

| 노트북 · 방법 | 원본 | 설정 | 점수 |
|---|---|---|---|
| 05a `routerdc` | LLMRouter `RouterModule`·`DCDataset`·`add_clusters`·`DCTrainer`, mDeBERTa-v3-base | `llmrouter`(YAML: 500 step, batch 32, eval 50) · `official_routerbench`(RouterDC 저장소 RouterBench 스크립트: 1000 step, batch 64, eval 100) × 체크포인트 `best_dev` · `final` | 단계별 유사도 점수 |
| 05b `routellm_retrained` | RouteLLM BERT 구조(XLM-RoBERTa-base, 3분류), HF Trainer | 공개 체크포인트의 실제 학습 인자, 2000 step | 강한(high) 집단이 필요할 확률(점수 하나) |
| 06 `routellm_released` | `routellm/bert_gpt4_augmented`(revision 86237e3d), 학습 없음 | 하나 | 강한 모델이 필요할 확률(점수 하나) |

- **RouterDC:** 집단 셋을 후보 LLM 셋으로 두고 점수는 집단 풀 확률이다. 군집은 원본처럼 점수 벡터의 k-means(3)다. 원본 학습기는 "test" 분할로 `best_model.pth`를 고르는데, 그 자리에 우리 dev를 연결했다(계획서 4.3). 원본이 매 평가 때 학습 데이터 전체를 다시 평가하는 부분(추론에 쓰지 않는 `best_training_model.pth`용)은 건너뛴다. top_k·last_k는 후보가 셋이라 3이다.
- **RouteLLM 재학습본:** 공개 체크포인트에 남은 `training_args.bin`(실행하지 않고 내용만 읽음)과 `trainer_state.json`의 값을 그대로 쓴다: lr 1e-5, weight decay .01, AdamW, 4 epoch 선형 감소, 실효 batch 32, 500 step마다 검증 손실로 선택, fp32, seed 42. 논문의 "약 2000 step"대로 2000에서 멈춘다. 정답은 원본의 세 분류이며, 집단 풀 확률로 만든 부드러운 확률이다(강한 승 = p_high(1 − p_low), 약한 승 = p_low(1 − p_high), 무승부 = 나머지).
- **원본 함수와 대조:** 두 RouteLLM 점수는 묶음 계산이지만, 원본 `BERTRouter.calculate_strong_win_rate`를 고정 커밋에서 불러 첫 8문항을 직접 비교한다(차이 1e-4 넘으면 멈춤). 로컬 시험에서 차이는 1e-7 수준이었다.
- **속도·메모리만 바꾼 것:** gradient checkpointing, 묶음마다 전부 패딩인 열 제거, 동적 패딩, 옵티마이저 상태 없는 체크포인트. 결과는 같다.
- **CPU 라우팅 시간:** 방법마다 dev 50문항을 CPU에서 한 문항씩 처리한 중앙값을 `cpu_ms_per_question`에 남긴다(08의 시간 비교용).
- 세 노트북은 서로 독립이라 다른 Colab 세션에서 동시에 돌릴 수 있다. 소스는 `router-final-source-finetune.zip`이다.

## 07a_score_ours · 07b_lock

**07a** (`score_modernbert.py`, T4): 03 모델 셋(R0-집단·R1-집단·R1-집단-A)과 RT-08로 dev·test·RouterArena 질문 전체의 점수(logit)를 계산해 Drive `scores-modernbert/<모델>/`에 쓴다. 03은 자기 검증 행에만 점수를 냈고(R1-집단-A는 A 분할만) 02는 점수를 내지 않았기 때문이다. 학습과 같은 토큰화(512토큰)와 RT-05의 `_predict_logits`를 쓰며, 로컬 시험에서 03의 검증 점수와 차이는 2e-5 이하였다.

**07b** (`lock.py`, CPU): dev 결과와 03~07a의 점수로 아래를 정해 Drive `lock/`에 쓴다. test 결과는 읽지 않는다.

- **평가 모형:** 단계 t를 부르면 집단 풀 확률 p_t로 성공하고(집단에서 모델 하나를 무작위로 부름), 실패하면 다음 단계로 올라간다. 시작 단계마다 첫 시도 성공, 최종 해결, 기대 비용(서비스 비용 .2/.6/1.0), 실패 횟수를 정확히 계산한다.
- **어댑터:** 원천·단계별 isotonic 회귀. 분류 출력(R0-집단, RT-08, 운영 R0)은 먼저 train의 "분류별 단계 성공률" 표를 곱하고, 점수 하나만 내는 방법(Hybrid LLM, RouteLLM)은 그 점수를 세 단계에 함께 쓴다. 설정 선택과 λ에는 dev 5분할 교차 보정을, test 적용에는 dev 전체로 맞춘 보정을 쓴다.
- **ρ:** EV-05와 같은 식을 dev의 집단 풀 확률에 적용한 원천별 단계 간 실패 상관이다.
- **비용 목표:** 원천마다 가장 싼·가장 비싼 고정 시작 단계의 기대 비용 사이 25·50·75% 지점이다(계획서 2.5, 10-17 확정). 상향이 반드시 따라와서 .25·.5는 대부분 원천에서 도달할 수 없기 때문이다.
- **길이 반영 민감도:** 질문마다 비용 = 단계 가격 × L / (원천 dev의 L 평균), L = 입력 토큰 + 5 × 출력 토큰 중앙값. 입력은 `length_inputs.py`(로컬, 원본 데이터 필요)가 만든 `run001/length-inputs/`이고, 07b가 SHA를 확인한다. 선택된 설정마다 이 비용의 비용 목표와 λ 표를 만든다(`sensitivity_length`). 설정의 이유와 나중에 점검할 것은 계획서 2.3에 있다.
- **집단 가격 민감도:** 단계 비용을 Claude 1:3:5 대신 집단 모델의 평균 데이터 가격(`prepare/groups.json`, high = 1)으로 바꿔, 선택된 설정마다 비용 목표와 λ 표를 한 번 더 만든다(`sensitivity_group_price`). 설정 선택은 1:3:5로만 한다. groups.json의 모델 목록은 봉인된 00_prepare manifest와 대조한다.
- **λ 표:** (원천, 규칙, κ, 비용 수준)마다 dev 기대 비용이 목표에 가장 가까운 λ다. 규칙은 상향 고려와 한 번에 고르기이고, 갈래 ②의 τ 규칙(τ = .8, 곡선 .5~.95)은 λ가 필요 없다.
- **설정 선택:** 상향 고려 규칙의 (최종 해결 − κ × 실패 횟수)에서, 같은 dev 비용에서 고정 시작 단계(혼합 포함)가 낼 수 있는 가장 좋은 값을 뺀 "이득"을 원천·κ 격자·비용 수준에 걸쳐 평균하고, 가장 큰 설정을 고른다.
- **평가 제외:** [문서 제공형 제외 목록](../../../experiments/results/router-final-inputs/document-grounded-eval-exclusions.md)의 xRouteBench SQuAD·BoolQ 180건(dev 80, test 100)과 [영어 제외 목록](../../../experiments/results/router-final-inputs/non-english-eval-exclusions.md)의 영어가 아닌 문항 246건(모두 LLMRouterBench test arenahard)을 dev와 모든 보정 출력에서 지운다. 08은 이 문항을 볼 수 없다. 목록의 id가 점수에 없으면 멈춘다.
- **RT-08:** RT-08(v2 재학습)과 운영 R0(v1, 저장소의 512토큰 점수)를 A 분할에서 집단 기준과 대표 모델 기준 두 가지로 보정한다.
- **출력:** `selection-lock.json`(선택 설정, ρ, 비용 목표, λ 표와 두 민감도의 λ 표, 규칙, 평가 제외 목록, 의존 파일 SHA), `calibrated/<방법>.parquet`(dev·test 단계별 확률), `selection-table.parquet`(모든 설정의 dev 근거), `lock-summary.md`.

dev 점수 SHA는 각 단계의 manifest와 대조하며, 빠진 점수가 있으면 멈춘다(`--allow-missing`은 시험용). 07b가 끝나면 lock 요약을 보고하고 확인을 받은 뒤 08로 간다. 소스는 `router-final-source-lock.zip`이다(운영 R0 점수 파일과 영어 제외 목록 포함).

## 데이터 구성 (카테고리별)

`categories.py composition`(로컬)이 train·dev·test의 카테고리별 문항 수와 train·dev의 필요 단계 비율을 [data-composition.md](../../../experiments/results/ev10-router-final-20261018/data-composition.md)에 쓴다. test는 문항 수만 센다.

## length_inputs (길이 반영 민감도 입력, 로컬)

`length_inputs.py build`가 dev·test 문항마다 입력 토큰(라우터 토크나이저, 자르지 않음)과 출력 토큰 중앙값(그 질문에 답한 데이터 모델 전부)을 `run001/length-inputs/lengths.parquet`에 쓴다. 정답 여부는 쓰지 않는다. 출력 길이는 LLMRouterBench `completion_tokens`, SPROUT `num_output_tokens`, xRouteBench `output_tokens`이고, RouterBench는 답 원문을 같은 토크나이저로 센다. 원본 데이터가 있는 checkout에서 한 번 실행한다(약 30초).

```
python ml/notebooks/router_final/length_inputs.py --run-dir <run001> --repo . --output-dir <run001>/length-inputs
```

## 08a_latency

`routing_latency.py`가 07b lock에서 고른 라우터마다 **질문 원문 → 시작 단계** 시간을 잰다(계획서 2.5). 같은 Colab 세션에서 dev 200문항(원천마다 50, sample_id 해시로 고정)을 한 문항씩, 워밍업 20문항 뒤 측정하고 중앙값·p95를 `latency/latency-<cpu|cuda>.parquet`에 쓴다.

- **나누어 재는 구간:** 인코더(BGE 또는 라우터 자체 인코더), 라우터 본체(이웃 검색·회귀·그래프·묶음 점수 등), 보정과 결정 규칙(isotonic 세 개 + 상향 고려 규칙, 모든 라우터 공통).
- **BGE 라우터:** `total_with_bge`(BGE 포함)와 `added_given_bge`(서비스가 캐시용으로 BGE를 이미 계산했을 때 더해지는 부분)를 함께 남긴다.
- **다시 만드는 라우터:** 지연은 구조와 데이터 크기로 정해지므로, 가중치를 남기지 않은 04 본체(같은 클래스·설정·학습 데이터 크기, 짧게 학습)와 05 인코더(같은 구조)는 다시 만들어 잰다. ModernBERT(03·RT-08)와 RouteLLM 공개본은 실제 가중치를 쓴다. 운영 R0의 8,192토큰 조건은 같은 구조의 RT-08 가중치로 따로 잰다.
- **GraphRouter:** 원본 추론처럼 새 질문을 학습 그래프에 붙여 그래프 전체를 한 번 계산한다.
- 로컬 시험(Mac CPU, 문항 12개): BGE-large 인코딩 약 300ms, ModernBERT R1-집단 약 130ms, BGE 라우터 본체는 MLP·SVM·MF·R1-BGE 1ms 미만, kNN 약 37ms, IRT 약 25ms, GraphRouter 공식 약 53ms·LLMRouter 판 약 141ms였다. 값은 장비에 따라 다르므로 Colab 한 세션의 값만 비교한다.
- 소스는 `router-final-source-latency.zip`이다.

## 08_test

`test_eval.py`가 본체다. 07b lock에서 고정한 것만 쓰고 아무것도 고르지 않는다. Colab CPU에서 돌리며 소스는 `router-final-source-test.zip`이다. A test의 v1 채점값은 원본 데이터에서 다시 만들고, bootstrap 묶음 ID는 원천 파일에서 읽는다. 이 원본 11개(87MB)를 저장소와 같은 경로로 Drive `inputs/test-data/`에 두었고(`test-data-manifest.json`에 SHA), 노트북이 실행 전에 대조한다. 로컬에서는 `--data-root`를 데이터가 있는 checkout으로 주면 된다. 읽은 입력 파일의 SHA는 출력 manifest의 `input_sha256`에 남는다.

- **입력 확인:** lock이 시험 실행(빠진 점수)이 아닌지, 보정 파일·영어 제외 목록·00 test 출력의 SHA가 기록과 같은지 확인한다. 출력 폴더 `RUN/test`가 이미 있으면 멈춘다(test는 한 번 연다).
- **평가 행:** test에서 v2 제외 문항, 평가 제외 목록(영어가 아닌 246건, 문서 제공형 100건), 집단 풀 확률이 비어 있는 행을 뺀다. RouterArena 진단은 `Context`가 붙은 1,603건을 뺀다. 건수는 `manifest.json`의 `counts`에 남는다.
- **갈래 ①:** 라우터 × 규칙(상향 고려, 한 번에 고르기) × κ 21개 × 비용 수준 3개마다 test 효용(최종 해결 − κ × 실패 횟수)에서 같은 test 비용의 고정 시작 단계(혼합 포함) 값을 뺀 이득. 기준은 Claude 1:3:5(주), 집단 평균 가격·길이 반영(민감도), A test v1 채점(점 추정). oracle(참 확률, λ를 test에서 맞춤)과 운영 R0(A 분할)는 참고용으로만 싣고 순위에 넣지 않는다.
- **구간:** 질문 묶음 단위 paired bootstrap 10,000회(seed 42). 묶음은 A 분할이 `duplicate_group_id`, 외부 데이터가 `split_group_id`다. 한 원천 안에서는 모든 라우터·규칙·κ·수준이 같은 표본을 쓰고, 원천끼리는 따로 뽑는다. macro는 원천 4개의 평균이다.
- **κ\*:** 이득 > .01이고 95% 구간 하한 > 0인 가장 작은 κ(`kappa-star.parquet`).
- **갈래 ②:** τ .5~.95마다 필요 단계 정확도·첫 시도 성공·최종 해결·호출 수·비용. 고정 시작과 "필요 단계에서 시작"도 함께 싣는다. RouterArena와 겹치는 외부 test 문항은 `track2-arena-overlap.parquet`에 따로 낸다.
- **가설:** H1~H7과 RT-08을 계획서 4.4·3절의 규칙대로 계산해 `hypotheses.json`에 쓴다. H3은 IRT-Router 난이도와 우리 필요 단계·RouterArena Bloom 라벨의 Spearman 상관이다. RouterArena에서는 R0-집단의 예측 단계와 Bloom 라벨의 일치도도 낸다.
- **풀리는 정도별 층(기술 통계):** solvable(어느 집단이든 ≥ .8), beyond_partial(모두 .8 미만, 0은 아님), beyond_zero(모두 0)마다 같은 지표를 낸다(`by-stratum.parquet`). 필요 단계 정확도는 solvable 층에서만 센다(`needed_tier_accuracy_solvable`, H2).
- **카테고리별(기술 통계, 판정 아님):** 원천별 카테고리와 공통 과제 유형(`categories.py` `TASK_TYPE`)마다 평균 이득(고정 시작 기준선을 그 카테고리 안에서 다시 만든다), τ = .8의 갈래 ② 지표와 시작 단계 비율. 30문항 미만 카테고리는 문항 수만 낸다. 원천을 합친 과제 유형은 문항 수 가중 평균이다(`by-category.parquet`).
- **출력:** `track1.parquet`, `fixed-starts.parquet`, `mean-gain.parquet`, `kappa-star.parquet`, `track2.parquet`, `track2-arena-overlap.parquet`, `rt08.parquet`, `by-category.parquet`, `by-stratum.parquet`, `test-starts.parquet`(질문마다 각 라우터의 시작 단계), `hypotheses.json`, `diagnostics.json`, `manifest.json`, `test-summary.md`.
- **리허설:** `--rehearse-on-dev`는 같은 코드를 dev 결과로 돌려 다른 폴더에 쓴다. test는 읽지 않는다. dev에서 맞춘 보정을 dev에 쓴 값이라 낙관적이며 해석하지 않는다.
- **주의(dev 리허설에서 확인):** LLMRouterBench는 1:3:5에서 고정 시작 비용이 low .97, high 1.0, medium 1.09로 거의 같다. 그런데 해결률과 실패 수는 크게 달라, 같은 비용 비교가 비용의 작은 차이에 크게 흔들리고 구간이 넓다. 그대로 보고하고 집단 가격 민감도(비용 범위가 넓다)를 함께 본다.

## 09_seeds (H5만)

계획서 4.3절대로 판정 경계(±.01) 근처 결과에만 seed를 더한다. 08에서 경계에 걸린 것은 H5(R1-집단 − R1-집단-A, 외부 test +.009) 하나다. kNN은 무작위성이 없어 R1 두 모델만 seed 17·73으로 다시 학습한다(`ours.py train --seed`, 결과 `ours/models/seed<N>/`). 점수는 `score_modernbert.py --seed`(`scores-modernbert-seeds/seed<N>/`). `seeds.py`가 각 seed 모델을 07b와 같은 방식(같은 어댑터, 잠근 ρ·비용 목표, 자기 λ 표)으로 dev에서 보정하고, 08과 같은 bootstrap 표본으로 test를 평가해 세 seed 평균으로 H5를 다시 판정한다(`seeds/`). 설정은 다시 고르지 않는다. seed마다 T4 세션 하나(3~4시간)로 동시에 돌릴 수 있다.

## 10_report

`report.py`가 08(필수)·09·08a(있으면) 출력을 읽어 `results.md`, `tables/*.csv`, `figures/*.png`, `manifest.json`(입력·출력 SHA)을 만든다. 결과를 다시 계산하지 않는다. 그림 글자는 영어다(Colab에 한글 글꼴이 없다). 저장소에는 `experiments/results/ev10-router-final-20261018/report/`와 사람이 쓴 [REPORT.md](../../../experiments/results/ev10-router-final-20261018/REPORT.md)를 둔다.

`arena_labels.py`(로컬 또는 CPU)는 RouterArena 난이도 라벨과 각 라우터 판정의 일치도(정확도·macro-F1·Cohen's κ·Spearman, 분야·세부 주제별, 혼동 행렬)를 `report/routerarena/`에 쓴다. 판정은 원래 출력(분류 모델), dev 단계 비율로 자르기, τ = .8 결정 세 가지이고 어느 것도 RouterArena 라벨로 맞추지 않는다.
