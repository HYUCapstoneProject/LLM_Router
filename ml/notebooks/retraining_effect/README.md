# R0 재학습 효과 실험 노트북 (F0/F1/F2)

상세 계약은 [재학습 효과 실험 계획서](../../../docs/ml/RETRAINING_EFFECT_EXPERIMENT.md)를 따른다. 이 디렉터리는 그 계획을 개인 Colab에서 실행하는 코드다. 클라우드 작업 제출, GCS 게시, 운영 `CURRENT` 변경은 하지 않는다.

| 순서 | 파일 | 역할 |
|---|---|---|
| 0 | [00_prepare.ipynb](00_prepare.ipynb) | anchor(운영 1.0) 확인, 고정 로거 추론, 피드백 시뮬레이션, 계획 봉인 |
| F1 | [01_f1_cycle_only.ipynb](01_f1_cycle_only.ipynb) | 현재 cycle 피드백만 학습 × seed 3, window별 prequential 예측 |
| F2 | [02_f2_replay.ipynb](02_f2_replay.ipynb) | 피드백 + history replay + base rehearsal × seed 3, window별 prequential 예측 |
| 3 | [03_compare_results.ipynb](03_compare_results.ipynb) | 최종 모델 잠금 → gate·retention·RouterArena 평가 → 보고서 |
| F3-0 | [00_prepare_f3.ipynb](00_prepare_f3.ipynb) | 첫 실험의 로거로 F3 계획 봉인(GPU 추론 없음) |
| F3-1 | [01_f3_balanced_easy.ipynb](01_f3_balanced_easy.ipynb) | F2 + 확인된 easy 보충 × seed 3, window별 prequential 예측 |
| F3-2 | [02_compare_f3.ipynb](02_compare_f3.ipynb) | F3 최종 평가 후 run001의 F1/F2와 한 보고서로 비교 |
| F4-0 | [00_prepare_f4.ipynb](00_prepare_f4.ipynb) | 첫 실험의 로거로 F4 계획 봉인(GPU 추론 없음), 봉인 test 입력 확인 |
| F4-1 | [01_f4_bidirectional.ipynb](01_f4_bidirectional.ipynb) | 양방향 오류 + 복습 × seed 3, window별 prequential 예측 |
| F4-2 | [02_compare_f4.ipynb](02_compare_f4.ipynb) | 초기 R0(F0)와 F4 최종 모델을 봉인 test·RouterArena에서 평가·비교 |
| F5-0 | [00_prepare_f5.ipynb](00_prepare_f5.ipynb) | F5 계획 봉인(GPU 추론 없음), window별 선택 수와 학습 라벨 비율 확인 |
| F5-1 | [01_f5_matched_prior.ipynb](01_f5_matched_prior.ipynb) | 오답 전부 + 비율을 맞추는 정답 문항 + 실제 비율 복습 × seed 3 |
| F5-2 | [02_compare_f5.ipynb](02_compare_f5.ipynb) | F0·F4(참고 run)·F5를 봉인 test·RouterArena 같은 문항에서 비교 |
| G-1 | [04_gain_metric.ipynb](04_gain_metric.ipynb) | F0~F5의 저장된 예측을 첫 시도 이득 G(단계 가격 1:3:5)로 다시 평가 |
| F6-1 | [04_f6_training_free.ipynb](04_f6_training_free.ipynb) | 학습 없이 갱신(문턱만 다시 맞춤) 비교군을 기존 지표로 평가 |
| F6-2 | [05_f6_gain_metric.ipynb](05_f6_gain_metric.ipynb) | G-1에 F6 비교군을 더해 G로 평가(1:3:5) |
| F6-3 | [06_f6_gain_dataprice.ipynb](06_f6_gain_dataprice.ipynb) | F6-2와 같은 예측으로 단계 가격만 데이터 평균 가격(high = 1)으로 바꿔 평가 |

G 재평가(G-1, F6-2, F6-3)의 집계 결과는 `experiments/results/retraining-gain-20261021/`, `retraining-gain-f6-20261021/`, `retraining-gain-f6-dataprice-20261022/`에 있다. 원본은 개인 Drive `MyDrive/LLM_Router/`의 같은 이름 폴더다. 05와 06의 노트북은 각각 [`build_f6_gain_notebook.py`](build_f6_gain_notebook.py), [`build_f6_dataprice_notebook.py`](build_f6_dataprice_notebook.py)로 만든다. 06이 쓰는 후보 통계 파일의 SHA는 [`dataprice_stats.sha256`](dataprice_stats.sha256)에 있다.

## 실행 흐름

1. [`build_bundle.py`](build_bundle.py)로 만든 `retraining-effect` kit(F3는 `--variant f3`, F4·F5는 `--variant f4`·`--variant f5`로 만든 kit)를 개인 Drive의 `MyDrive/LLM_Router/`에 올린다. F4·F5 kit에는 봉인 test(`inputs/test.parquet`)가 들어 있으며, 로컬 frozen manifest의 SHA를 확인한 뒤 만든다.
2. 00 → (01, 02를 별도 세션에서) → 03 순서로 전체 실행한다.
3. 03이 내려받는 `results-for-review.zip`을 검토에 사용한다.

입력 데이터(6개 Parquet)와 anchor 모델은 R1→R0 전이 실험 kit가 이미 올려 둔 `MyDrive/LLM_Router/transfer-experiments/inputs/experiment-A`를 읽기 전용으로 재사용한다. 새 분할이나 초기학습을 하지 않는다.

## 핵심 구현 규칙

- **피드백 생성:** anchor를 stream 전체에 8192로 한 번만 추론하고, 기록된 `success_*`로 실패 시 한 단계씩 올린다. 상향 후 성공한 질문만 피드백이며 정답은 처음 성공한 단계다. window당 상한 없이 전부 쓰고, 32건 미만인 window가 있으면 학습 전에 중단한다.
- **버전 규칙:** update 1~3은 anchor에서, 4~6은 해당 arm의 update-03(2.0)에서 가중치만 가져온다. cycle 데이터는 cycle 안에서 누적한다.
- **학습량:** F2는 3 epoch, F1은 같은 update의 F2와 실제 optimizer step이 같도록 epoch를 늘린다. 완료 step이 다르면 실패로 처리한다.
- **평가 보호:** window j+1은 update j 모델이 학습 전에 예측한다. gate·retention·RouterArena는 모든 학습과 window 평가가 끝나고 `final-lock.json`이 생긴 뒤에만 평가한다.
- **F3 easy 보충:** 도착한 window에서 로거가 easy로 보내 첫 시도에 해결된 질문을, medium·hard 평균 수가 되도록 F2 구성에 더한다. 한 번 쓴 easy는 다시 뽑지 않는다. 계획서 12절 참조.
- **F4 양방향 오류 교정:** 도착한 window에서 로거 예측이 원래 benchmark 난이도 라벨과 다른 질문을 방향(높게/낮게)과 관계없이 모두 골라 **원래 라벨**로 학습한다. 지난 cycle 오류의 라벨별 32개와 초기 학습 라벨별 32개를 F2와 같이 섞고, 자기 학습 행 기준 3 epoch로 학습한다. 최종 평가는 봉인 test와 RouterArena이며 `compare-final`이 F0 대비 차이를 보고한다. 계획서 14절 참조.
- **F5 라벨 비율 맞춤:** window마다 F4의 오답을 전부 넣고, 맞게 예측한 질문을 라벨별로 더해 그 window의 원래 라벨 비율을 되돌린다. 비율을 묶는 라벨(현재 hard)은 오답만 쓴다. 복습은 F2·F4와 같은 크기(지난 cycle 96행 + 초기 학습 96행)를 실제 라벨 비율대로 뽑는다. `compare-final --reference-run`에 F4 run을 주면 같은 평가 데이터(SHA 일치)를 쓴 참고 run의 최종 예측을 함께 읽어 F5−F4 차이를 낸다. 계획서 15절 참조.
- **평가 역할 사이 중복:** 봉인 test와 RouterArena는 둘 다 학습에 쓰지 않으므로 같은 질문 텍스트를 허용하고 개수를 계획에 기록한다(현재 2문항). 학습·검증·stream과 겹치면 기존처럼 중단한다.
- **판정:** window 2~6의 초기 선택 품질 차이(0.01 초과)와 비용·호출 깊이 차이를 질문 그룹 bootstrap으로 보고하고, retention source-macro-F1 하락 0.01 이내를 망각 기준으로 쓴다.

테스트는 [`ml/tests/test_retraining_effect_experiment.py`](../../tests/test_retraining_effect_experiment.py)에 있으며 합성 데이터만 사용한다.
