# R0 재학습 효과 실험 (F0/F1/F2) 계획서

작성일: 2026-09-28. 이 문서는 **실행 전 사전 계획**이며 결과 보고서나 운영 변경 승인이 아니다. 전체 연구·운영 계약은 [ML 단일 문서](README.md)의 [5.6~5.9절](README.md#retraining-6)을 따르며, 이 문서는 그 사양을 현재 운영 조건에 맞춰 실행 가능한 범위로 고정한다. 실행 코드는 [`ml/notebooks/retraining_effect/`](../../ml/notebooks/retraining_effect/README.md)에 있다(2026-09-29).

## 1. 목적과 질문

운영 R0는 에스컬레이션 피드백이 쌓이면 재학습한 모델로 교체된다. 이 실험은 **재학습을 해야 하는가, 한다면 어떤 데이터로 해야 하는가**를 같은 피드백 조건에서 비교한다.

| 질문 | 비교 |
|---|---|
| Q1. 재학습이 다음에 들어올 질문의 라우팅을 실제로 개선하는가? | F1·F2 대 F0 |
| Q2. 운영처럼 과거 피드백과 초기 학습 데이터를 섞으면(F2) 새 피드백만 학습(F1)할 때보다 망각이 줄어드는가? | F2 대 F1 |
| Q3. 개선이 있다면 비용(더 비싼 단계로 보내는 비율)은 얼마나 늘어나는가? | 세 군의 품질–비용 |

이 실험은 학습 목표 비교(R0/R1/R2)나 출력층 전이(R1→R0) 실험과 다른 축이다. 목적함수는 R0 3-class CE로 고정한다.

## 2. 출발점과 재사용 자원

| 자원 | 상태 | 이번 사용 |
|---|---|---|
| 출발 모델 | 운영 1.0 `router-modernbert-r0-a-20260927-v1` (A train 21,406행·2,007 step R0) | 세 군의 공통 anchor. 준비 단계에서 R1→R0 전이 실험의 R0 pin과 digest가 같은지 확인 |
| 연속 학습 분할 | `continual-60-10-10-15-5-v1-seed-42`: train 21,406 / dev 3,581 / gate 3,581 / stream 5,368 / retention 1,797 | 전이 실험과 같은 분할을 그대로 사용. 다시 나누지 않음 |
| 후보 결과 | 각 행에 `success_low/medium/high`, `quality_*`, `actual_cost_*` 기록 | 피드백 시뮬레이터가 호출한 단계의 결과만 공개 |
| 실행 기반 | [R1→R0 전이 실험](R1_TO_R0_TRANSFER_EXPERIMENT.md)의 helper·노트북·Drive kit·checkpoint 복구 | `prepare / train / score / compare` 구조를 재사용해 새 디렉터리에 구현 |
| 실행 환경 | 사용자 개인 Colab T4, `alexyoo0506@gmail.com` Drive | 동일. GCP·Vertex·운영 GCS는 사용하지 않음 |

**전이 실험과의 관계.** 전이 실험의 A(`r0_continued`)는 구성상 F2에 가깝지만, 이번 실험에 그 결과를 재사용하지 않는다. 세 가지 조건이 다르기 때문이다.
1. 피드백 라벨로 benchmark 난이도를 그대로 썼다. 이번에는 시뮬레이터가 실제로 성공한 단계를 쓴다.
2. 각 cycle을 직전 cycle 모델에서 이어 학습했다. 이번에는 운영 버전 규칙대로 anchor에서 다시 시작한다.
3. window마다 100 step을 고정했다. 이번에는 운영 학습 설정인 3 epoch를 쓴다.

## 3. 비교군

| Arm | 업데이트 학습 행 | 비고 |
|---|---|---|
| F0 Frozen | 없음 | anchor 1.0 유지. 로깅 모델과 같음 |
| F1 Cycle-only | 현재 cycle의 누적 피드백 | base 학습 행과 과거 cycle replay를 제거 |
| F2 Replay+rehearsal | 현재 cycle 피드백 + 과거 피드백 replay(라벨별 최대 32) + base train rehearsal(라벨별 32) | 운영 merge와 같은 구성 |

F1/F2 모두 D_dev에서 고정한 validation 48행(라벨별 16)을 학습 내부 검증에만 사용한다.

피드백을 전부 쓰므로 cycle 누적 피드백은 약 250(첫 window)~750행(cycle 셋째 window)이다. F2의 복습 행은 운영 상한대로 최대 160행(base 96 + history 64)이라 F2 학습 데이터에서 새 피드백 비중은 약 70~90%가 된다. 운영에서도 같은 비중이 생기므로 복습 상한은 바꾸지 않고, 이 조건에서 F2의 보호 효과를 측정한다. F1/F2 차이는 history replay와 base rehearsal의 **결합 효과**이며 각각의 개별 효과를 주장하지 않는다.

## 4. 피드백 생성: 고정 로거 시뮬레이터

1. D_stream 5,368행을 `stream_seed`로 한 번 섞고, 도착 순서대로 **6개 window**(약 895행씩)로 나눈다.
2. 로거는 anchor 1.0을 8192 추론으로 한 번만 실행한다. 예측 단계를 초기 단계로 삼는다.
3. 초기 단계의 `success_*`가 실패면 다음 단계로 올리고, 처음 성공한 단계에서 멈춘다. 모든 시도 비용을 합산한다. 끝까지 실패한 요청도 분모에 남긴다.
4. **상향 후 성공한 요청만** 피드백 event가 된다. `finalLabel`은 성공한 단계이며 `label_origin=observed_escalation_action`(시뮬레이션)으로 기록한다. `userSatisfied=true`는 benchmark 성공의 대리 신호다.
5. 각 window 끝에서 그 window에서 발생한 **유효 피드백 전부**를 제출한다(상한 없음, 2026-09-28 결정). README 5.8절의 연구 설정값 32행 상한은 발생한 피드백 대부분을 버리므로 사용하지 않는다. 유효 피드백이 32행 미만인 window가 있으면 학습 전에 설계를 수정하고 새 version으로 등록한다. 결과를 본 뒤 조정하지 않는다.

**예상 규모.** anchor 1.0의 D_dev 예측(3,581행)에 같은 규칙을 적용하면 첫 시도 해결 59.7%, 상향 후 해결 27.8%, 끝까지 실패 12.5%였다. 이를 stream에 적용하면 **window당 약 250건, 전체 약 1,490건**의 피드백이 예상된다. 피드백 정답은 medium 약 58%·hard 약 42%이며 easy는 0건이다. 원천은 RouterBench가 약 94%로 xRouteBench 피드백은 적다. 실제 수는 00 준비 단계의 stream 로거 추론으로 확정한다.

F1과 F2는 **같은 피드백 batch**를 받는다(off-policy 고정 로거). 각 arm이 자기 피드백을 만드는 on-policy 실험은 범위 밖이다.

## 5. 버전·anchor 규칙

운영 `successfulHttpRequest` 정책과 현재 운영 모드인 **직접 승격**을 따른다. 품질 gate가 없으므로 모든 업데이트가 승격된다.

| window | 새 버전 | 시작 weight | cycle 학습 데이터 |
|---:|---|---|---|
| 1 | 1.1 | anchor 1.0 | B1 |
| 2 | 1.2 | anchor 1.0 | B1+B2 |
| 3 | 2.0 | anchor 1.0 | B1+B2+B3 |
| 4 | 2.1 | anchor 2.0 | B4 |
| 5 | 2.2 | anchor 2.0 | B4+B5 |
| 6 | 3.0 | anchor 2.0 | B4+B5+B6 |

weight만 가져오고 optimizer·scheduler는 매번 새로 만든다. 1.2는 1.1이 아니라 1.0에서 시작한다. 품질 gate 모드 재현과 15성공 pristine rebase는 범위 밖이다.

## 6. 학습 조건

| 항목 | 값 |
|---|---|
| F2 학습량 | 운영 학습 설정과 같은 **3 epoch**, `max_train_steps` 없음 |
| F1 학습량 | 같은 window에서 F2의 **실제 완료 optimizer step과 같도록** epoch를 늘리고 step 상한으로 맞춤. 두 값을 기록·검증 |
| optimizer | AdamW, lr `2e-5`, weight decay `.01`, warmup `.1`, clip 1.0 |
| batch | micro 8 × accumulation 4 = 32 |
| sampler | 두 arm 모두 shuffle, `sample_weight=1`. 운영 balanced sampler로 바꾼 F2-P는 범위 밖 |
| 길이 | 학습 512, 평가 8192 |
| 반복 | adaptation seed 42/43/44. 초기학습 반복이 아님을 명시 |
| pilot | 없음. 학습량은 운영 설정에서 가져오며 결과를 보고 바꾸지 않음 |

## 7. 평가

| 시점·데이터 | 지표 |
|---|---|
| **prequential**: 각 window를 업데이트 **이전** 모델로 예측 (window 2~6 주 분석, 1은 참고) | 초기 선택 품질, 첫 시도 해결률, 최종 해결률, 평균 호출 깊이, 누적 모의 LLM 비용 |
| 최종 모델(3.0), D_retention_test | source-macro-F1, easy→hard 과잉 상향 비율, 선택 품질·비용. F0 대비 변화 |
| 최종 모델, D_gate | 위와 같은 라벨·선택 지표(연구용 보고, 승격 판단에는 쓰지 않음) |
| 최종 모델, RouterArena | 외부 난이도 정렬 진단. 학습·선택에 쓰지 않음 |

- 같은 질문의 arm 간 차이는 duplicate group 단위 paired bootstrap으로 요약하고 seed별 값·평균·범위를 함께 보고한다. 여섯 window를 독립 표본으로 취급하지 않는다.
- 봉인 RouterBench/xRouteBench test 평가는 adapter가 없으므로 이번 기본 범위에서 제외한다.
- 실제 CPU/GPU 비용과 모의 LLM 호출 비용을 분리해 기록한다.

## 8. 사전 판정 기준

주 지표는 **window 2~6의 prequential 초기 선택 품질 평균**이고, 보조로 누적 모의 비용을 같이 본다.

1. **재학습 효과(Q1):** F1 또는 F2가 F0보다 주 지표에서 0.01 초과로 높으면 “효과 있음”으로 기록한다. 누적 비용도 함께 보고하며, 비용이 F0보다 늘어난 개선은 “비용 증가를 동반한 개선”으로 따로 구분한다.
2. **망각(Q2):** 최종 모델의 retention source-macro-F1이 F0보다 0.01 넘게 떨어지면 “망각 기준 미달”이다. 전이 실험과 같은 허용폭이다.
3. **F2 대 F1:** 두 기준을 함께 보고 우열을 기록한다. 차이가 신뢰구간 안이면 “차이 확인 안 됨”으로 둔다.

결과가 어떻든 운영 설정(직접 승격, 3 epoch, 운영 merge)은 이 실험으로 자동 변경하지 않는다. 운영 변경이 필요하면 별도 결정으로 다룬다.

## 9. 실행 구조와 예산

새 디렉터리 `ml/notebooks/retraining_effect/`에 다음을 만든다.

| 노트북 | 역할 |
|---|---|
| `00_prepare.ipynb` | 입력·pin 검증, anchor digest 확인, 로거 추론과 시뮬레이션, window별 피드백 수 기록과 최소 32행 검사, plan 봉인 |
| `01_f1_cycle_only.ipynb` | F1 × seed 3의 6 window 학습과 prequential 예측 |
| `02_f2_replay.ipynb` | F2 × seed 3의 6 window 학습과 prequential 예측 |
| `03_compare.ipynb` | F0 포함 최종 평가(retention·gate·RouterArena), 판정, 보고서·그림, `results-for-review.zip` |

사용 흐름은 전이 실험과 같다. 00 전체 실행 → 01/02를 각각 다른 GPU 세션에서 실행 → 둘 다 끝나면 03 실행. 01과 02는 같은 피드백 plan을 공유하고 자기 arm 경로에만 쓴다.

학습 run은 2 arm × 3 seed × 6 window = 36회다. window당 학습 행이 수백 행이라 학습 시간은 짧고, 비용의 대부분은 8192 추론이다. prequential 약 900행 × 6 window와 최종 평가 약 13,800행이 arm·seed마다 필요하다. 전이 실험의 실측(학습 4.5시간, forget 1,797행 추론 약 75초)을 기준으로 **T4 총 3~5시간**을 예상한다. 첫 window 기록으로 남은 시간을 다시 추정하되, 학습량은 바꾸지 않는다.

## 10. 범위에서 제외

- 과적합 가설 검증, 학습량 sweep, pilot
- 품질 gate 모드 재현, 15성공 pristine rebase, `validatedUniqueBatch` 모드
- 분포가 바뀌는 shift trace, on-policy 로깅, F2-P(balanced sampler)
- R1 재학습, 운영 인수 4단계(실서버 8192 부하 시험)·5단계(전체 재학습·복구 시험)
- 새 Claude API 호출, 운영 GCS·CURRENT 변경

## 11. 실행 전·완료 체크리스트

- [x] anchor 1.0 digest와 전이 실험 R0 pin 일치 확인 — 운영 1.0 approved `model.safetensors`와 pin이 모두 `e1a40fec…`, 원본 archive `d15da837…` (2026-09-29)
- [ ] stream·dev·gate·retention의 행 수·SHA와 역할 간 duplicate group 누수 0 재확인
- [x] 시뮬레이터 단위 테스트: 상향 경로, 끝까지 실패, high 시작, 비용 합산 — `ml/tests/test_retraining_effect_experiment.py`
- [ ] 여섯 window 모두 유효 피드백 32행 이상 확인 후 plan 봉인 — 로컬 CPU 미리보기(봉인 아님)에서 249/253/238/255/237/260건, 전체 1,492건(medium 886·hard 606, RouterBench 1,409·xRouteBench 83). 공식 값은 Colab 00의 GPU 로거로 확정
- [ ] F1/F2 window별 실제 완료 step 일치 기록
- [ ] F0/F1/F2 prequential·최종 예측 SHA와 receipt 일치
- [ ] 결과 보고서를 `experiments/results/retraining-effect-v1-<실행일>/`에 반입하고 README 5.12에 요약

## 12. 추가 비교군 F3: easy 보충 (2026-09-30 추가)

**배경.** run001(`retraining-effect-v1-run001`)에서 F1·F2 모두 첫 재학습부터 easy 예측이 60%→0%가 되었다. 초기 선택 품질은 F0보다 약 0.18 높았지만, 질문당 비용이 약 77% 늘고 최종 해결률이 87.0%→83.0%로 낮아졌다. retention source-macro-F1도 약 0.27 떨어졌다. 상향 피드백에는 구조적으로 easy 정답이 없고, F2의 base rehearsal(easy 32행)은 학습 데이터의 10~30%에 불과해 이 쏠림을 막지 못했다.

**F3 구성.** 사용자 결정에 따라 F2 구성에 easy 행을 보충해 세 단계의 학습 비율을 최대한 균일하게 만든다.

| 항목 | 규칙 |
|---|---|
| easy 출처 | 이미 도착한 window(현재 update 이하)에서 로거가 easy로 보냈고 low 후보가 첫 시도에 해결한 질문. 서비스에서 “Haiku로 보냈고 상향 없이 끝난 요청”에 해당한다. 라벨 출처는 `simulated_confirmed_first_try_easy` |
| 보충량 | F2 학습 행의 medium·hard 수의 평균(올림)이 되도록 easy를 채운다. 이미 있는 base easy 32행을 포함해 계산한다 |
| 중복 방지 | 한 번 학습에 쓴 easy 질문은 이후 update에서 다시 뽑지 않는다. 남은 후보 중 `easy_supplement_seed`로 무작위 선택하며, 후보가 모자라면 재사용하지 않고 부족분을 기록한다 |
| 학습 | 운영과 같은 버전 규칙, 자기 학습 행 기준 3 epoch. F2보다 학습 행이 많으므로 optimizer step도 많다(F2와 step을 맞추지 않음) |
| 비교 | 같은 로거·stream을 쓰는 새 run(`retraining-effect-v2-f3-run001`)에서 F3만 학습하고, run001의 F1/F2를 해시와 행 내용 검증 후 불러와 한 보고서로 비교한다 |

로컬 CPU 미리보기(봉인 아님)에서는 update별 easy 보충이 125/251/370/160/278/408행으로 부족분이 없었다. 서로 다른 easy 질문 1,592건이 쓰였고, 학습 비율은 easy·medium·hard 약 33/38/28%였다. seed당 학습량은 510 step이다(F2는 360).

**판정.** 8절 기준을 그대로 쓰고, F3−F2의 품질·비용 차이를 추가로 보고한다. F3가 easy 비율과 비용을 F0에 가깝게 유지하면서 품질·해결률을 개선하는지, retention 하락이 0.01 이내인지를 본다. F3와 F2는 데이터 구성과 학습량이 함께 다르므로, 차이를 easy 보충만의 효과라고 주장하지 않는다.
