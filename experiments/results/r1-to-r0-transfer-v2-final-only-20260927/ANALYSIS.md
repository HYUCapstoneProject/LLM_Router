# R1→R0 전이 실험 결과 분석 — 2026-09-27

실험 코드: **RF-01** · [실험 목록](../../REGISTRY.md)

실행 ID `r1-to-r0-transfer-v2-run001`, config `r1-to-r0-transfer-v2-final-only`, plan SHA-256 `05fcfd4c…7e77185`. 사전 계획은 [설계서](../../../docs/ml/R1_TO_R0_TRANSFER_EXPERIMENT.md)를 따른다. 이 문서는 04 노트북이 생성한 `report/`를 검토한 결과이며, 서비스 승격 승인이 아니다.

## 1. 결론

1. **주 가설(B가 A보다 낫다)은 지지되지 않았다.** 개발검증 source-macro-F1은 A 0.411, B 0.405로 B−A = −0.0059(질문 그룹 bootstrap 95% CI −0.022~+0.011)이다. 사전 규칙(B가 0.01 초과로 높아야 B 선택)에 따라 **A(`r0_continued`)가 선택 고정**되었다. gate·forget에서도 B−A는 음수이고 CI가 0을 포함하며, RouterArena에서는 −0.012(CI −0.019~−0.004)로 B가 유의하게 낮다.
2. **A와 B 모두 망각 기준을 통과하지 못했다.** A는 원본 R0 대비 forget source-macro-F1이 seed별 0.092~0.119 떨어졌다(허용 0.01). B는 `retention-check.csv`에서 통과로 표시되지만 기준선이 동결 R1 정책(0.227)이기 때문이다. 같은 라벨 지표로 원본 R0(0.517)와 비교하면 B도 0.10~0.14 낮다. 따라서 **어느 비교군도 배포 후보가 아니며**(`production_approved: false`), 사전 규칙대로 결과를 보고 재학습·재튜닝하지 않는다.
3. **성능 하락의 방향은 “상향 편향”이다.** 피드백 적응 후 hard recall은 0.25→0.52, under-route는 0.27→0.10, hard→easy 오류는 0.043→0.010으로 줄었다. 대신 easy 예측 비율이 61%→14%(A seed 42, 실제 easy 49%)로 무너져 accuracy가 0.60→0.40으로 내려갔다. 선택 품질은 0.610→0.734로 올랐지만 선택 비용이 약 3.5배(0.00067→0.0023)로 늘었다. 같은 비용에서 좋아진 것이 아니라 더 비싼 단계로 보내 얻은 품질이다.
4. **C(동결 R1)** 는 단계 라벨 지표가 0.20으로 가장 낮지만 품질 0.789·비용 0.0036으로 가장 비싸고 품질이 높은 쪽을 고른다. 목적(후보별 성공+비용 정책)이 다르므로 라벨 F1만으로 C가 나쁘다고 결론 내리지 않는다. B−C가 +0.20인 것도 출력 의미 전환 효과를 포함한다.

## 2. 비교군별 평균 (seed 42/43/44, 8192 평가)

| 역할 | 지표 | 원본 R0 | A cycle-06 | B cycle-06 | C 동결 R1 |
|---|---|---|---|---|---|
| dev (3,581) | source-macro-F1 | 0.524 | 0.411 (0.398–0.419) | 0.405 (0.386–0.441) | 0.200 |
| gate (3,581) | source-macro-F1 | 0.527 | 0.422 | 0.413 | 0.213 |
| forget (1,797) | source-macro-F1 | 0.517 | 0.415 | 0.400 | 0.227 |
| RouterArena (8,379) | macro-F1 | 0.394 | 0.316 | 0.304 | 0.176 |
| dev | accuracy | 0.604 | 0.401 | 0.387 | 0.242 |
| dev | hard recall | 0.247 | 0.518 | 0.507 | 0.530 |
| dev | under-route | 0.268 | 0.102 | 0.116 | 0.065 |
| dev | 선택 품질 / 비용 | 0.610 / 0.00067 | 0.734 / 0.0023 | 0.725 / 0.0023 | 0.789 / 0.0036 |

B는 seed 편차가 A보다 크다(dev 0.386–0.441). seed 42에서는 B가 A보다 높았지만 43/44에서 낮아 방향이 안정적이지 않다.

## 3. 원인 진단

- **작은 feedback window의 과적합.** 각 cycle은 128–192행을 100 optimizer step × effective batch 32로 학습했다. 이는 window당 약 17–25 epoch이다. train loss는 0.10–0.38까지 내려가지만 내부 validation(48행) loss는 A 2.0–2.8, B 2.9–5.4로 커졌다. `max_steps`를 두 군에 같게 맞춘 계약이 작은 데이터에서는 반복 학습으로 작용했다.
- **상향 교정 대리 신호의 분포.** feedback 라벨은 정의상 상향 교정된 문항이다(예: cycle-03 easy 48 / medium 99 / hard 86). replay가 포함되어도 이를 반복 학습하면 전체 예측이 medium/hard 쪽으로 이동한다. 평가 라벨은 원래 benchmark 난이도이므로 이 이동이 곧 F1 하락으로 나타난다.
- **중간 평가 생략의 한계.** 계획대로 bridge·중간 cycle의 전체 dev 추론을 하지 않았다. 따라서 하락이 bridge(200 step)에서 왔는지 feedback cycle에서 누적됐는지 분리할 수 없다. bridge 내부 validation F1은 0.47–0.53으로 원본과 비슷해, 주된 하락은 feedback cycle 쪽으로 추정되지만 확정할 수 없다.
- B는 출력층을 새로 초기화하므로 bridge 200 step 동안 새 head를 맞추는 비용이 있다. validation loss가 A보다 계속 높은 것과 일치한다. 인코더 출처의 순수 효과를 분리하려면 설계서의 “R0 인코더 + 새 출력층” 보조군이 필요하다.

## 4. 무결성·실행 기록

- 예측 32개 모두 receipt의 `predictions_sha256`과 일치했다. 6개 최종 모델 digest는 `selection-lock.json`과 각 score receipt에서 같다.
- 6 run × 7 stage = 42 stage 모두 완료했다. bridge 200 step, 각 cycle 100 step을 달성했고 재개 없이 끝났다. 순수 학습 시간 합계는 약 4.5 GPU 시간(T4)이다.
- `status.json`: 주 예측·망각 비교 누락 없음, `all_retention_seeds_within_tolerance: false`, `production_approved: false`.
- Drive(`alexyoo0506@gmail.com` `MyDrive/LLM_Router/transfer-experiments/runs/r1-to-r0-transfer-v2-run001/`)에 arms·scores·report·shared-data와 단계별 `model.safetensors`(598,442,860 bytes)가 남아 있음을 2026-09-27에 확인했다. 가중치 재해시는 다운로드 비용 때문에 하지 않았다. 대신 receipt digest의 일관성으로 확인했다.
- 원본 전달 ZIP `results-for-review.zip` SHA-256 `8c41de8e225c0e7f61e0c72fc25b6ac566c5ae1cf55260c4d2afcd58e6fb35aa`. 이 폴더에는 예측 parquet(42MB)을 제외한 report·plan·lock·arm/score JSON만 반입했다. 예측 원본은 Drive와 ZIP에 있다.
- GCS 업로드, `CURRENT` 변경, Spring/FastAPI 변경은 없었다.

## 5. 후속 과제 (이번 범위 밖, 기록용)

2026-09-28 사용자 결정: 과적합 가설은 추가로 검증하지 않고 이 비교를 현재 결과로 종료한다. 아래 항목은 향후 참고용 기록일 뿐 예정된 실험이 아니다.


1. feedback 학습량을 epoch 기준(예: window당 1–3 epoch)으로 제한하거나 early stopping을 두고 재실행하면 과적합 가설을 직접 검증할 수 있다. 이는 새 plan·새 RUN_ID로 해야 하며 이번 결과에 맞춘 재튜닝으로 표시하지 않는다.
2. 망각을 줄이려면 replay 비율 확대, 낮은 학습률, 또는 원본 R0 anchor 정규화를 비교한다.
3. bridge 직후 dev 평가를 한 번 추가해 전환 효과와 feedback 효과를 분리한다.
4. 현재 운영 R0(A 1.0) 유지가 이 결과와 일치한다.
