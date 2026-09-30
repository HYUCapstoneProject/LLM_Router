# R0 재학습 효과 실험 노트북 (F0/F1/F2)

상세 계약은 [재학습 효과 실험 계획서](../../../docs/ml/RETRAINING_EFFECT_EXPERIMENT.md)를 따른다. 이 디렉터리는 그 계획을 개인 Colab에서 실행하는 코드다. 클라우드 작업 제출, GCS 게시, 운영 `CURRENT` 변경은 하지 않는다.

| 순서 | 파일 | 역할 |
|---|---|---|
| 0 | [00_prepare.ipynb](00_prepare.ipynb) | anchor(운영 1.0) 확인, 고정 로거 추론, 피드백 시뮬레이션, 계획 봉인 |
| F1 | [01_f1_cycle_only.ipynb](01_f1_cycle_only.ipynb) | 현재 cycle 피드백만 학습 × seed 3, window별 prequential 예측 |
| F2 | [02_f2_replay.ipynb](02_f2_replay.ipynb) | 피드백 + history replay + base rehearsal × seed 3, window별 prequential 예측 |
| 3 | [03_compare_results.ipynb](03_compare_results.ipynb) | 최종 모델 잠금 → gate·retention·RouterArena 평가 → 보고서 |
| F4-0 | [00_prepare_f4.ipynb](00_prepare_f4.ipynb) | 첫 실험의 로거로 F4 계획 봉인(GPU 추론 없음) |
| F4-1 | [01_f4_balanced_easy.ipynb](01_f4_balanced_easy.ipynb) | F2 + 확인된 easy 보충 × seed 3, window별 prequential 예측 |
| F4-2 | [02_compare_f4.ipynb](02_compare_f4.ipynb) | F4 최종 평가 후 run001의 F1/F2와 한 보고서로 비교 |

## 실행 흐름

1. [`build_bundle.py`](build_bundle.py)로 만든 `retraining-effect` kit(F4는 `--variant f4`로 만든 `retraining-effect-f4` kit)를 개인 Drive의 `MyDrive/LLM_Router/`에 올린다.
2. 00 → (01, 02를 별도 세션에서) → 03 순서로 전체 실행한다.
3. 03이 내려받는 `results-for-review.zip`을 검토에 사용한다.

입력 데이터(6개 Parquet)와 anchor 모델은 R1→R0 전이 실험 kit가 이미 올려 둔 `MyDrive/LLM_Router/transfer-experiments/inputs/experiment-A`를 읽기 전용으로 재사용한다. 새 분할이나 초기학습을 하지 않는다.

## 핵심 구현 규칙

- **피드백 생성:** anchor를 stream 전체에 8192로 한 번만 추론하고, 기록된 `success_*`로 실패 시 한 단계씩 올린다. 상향 후 성공한 질문만 피드백이며 정답은 처음 성공한 단계다. window당 상한 없이 전부 쓰고, 32건 미만인 window가 있으면 학습 전에 중단한다.
- **버전 규칙:** update 1~3은 anchor에서, 4~6은 해당 arm의 update-03(2.0)에서 가중치만 가져온다. cycle 데이터는 cycle 안에서 누적한다.
- **학습량:** F2는 3 epoch, F1은 같은 update의 F2와 실제 optimizer step이 같도록 epoch를 늘린다. 완료 step이 다르면 실패로 처리한다.
- **평가 보호:** window j+1은 update j 모델이 학습 전에 예측한다. gate·retention·RouterArena는 모든 학습과 window 평가가 끝나고 `final-lock.json`이 생긴 뒤에만 평가한다.
- **F4 easy 보충:** 도착한 window에서 로거가 easy로 보내 첫 시도에 해결된 질문을, medium·hard 평균 수가 되도록 F2 구성에 더한다. 한 번 쓴 easy는 다시 뽑지 않는다. 계획서 12절 참조.
- **판정:** window 2~6의 초기 선택 품질 차이(0.01 초과)와 비용·호출 깊이 차이를 질문 그룹 bootstrap으로 보고하고, retention source-macro-F1 하락 0.01 이내를 망각 기준으로 쓴다.

테스트는 [`ml/tests/test_retraining_effect_experiment.py`](../../tests/test_retraining_effect_experiment.py)에 있으며 합성 데이터만 사용한다.
