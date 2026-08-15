# Training Reports

완료되었거나 중단된 ModernBERT 학습 run별로 **실제 관측된 결과와 진단**을 기록하는
폴더다. 계약·명세 문서와 역할이 다르다.

| 문서 | 대상 run | 상태 | 핵심 결론 |
| --- | --- | --- | --- |
| `modernbert-v2-balanced-training-report.md` | `modernbert-router-balanced-v2` | 3 epoch 완주 | epoch 3은 과적합, 성능 상한은 라벨 정의에 있음 |
| `modernbert-v1-baseline-training-report.md` | `modernbert-router-baseline-v1` | epoch 1에서 중단 | 자연분포 학습으로 medium 클래스 완전 붕괴 |

## 작성 원칙

- 이 폴더의 문서는 **관측 기록**이다. 라벨 규칙, 학습 계약, 평가 metric의 *정의*는
  각각 `routerbench-data-labeling-spec.md`, `modernbert-training-spec.md`,
  `routerarena-routing-evaluation-spec.md`가 소유하며 여기서 재정의하지 않는다.
- 진행 상태 체크리스트는 `workstream-checklist.md`에서만 관리한다.
- 모든 수치는 Drive의 봉인된 artifact(`metrics.json`, `checkpoint_manifest.json`)
  또는 저장소의 `ml/artifacts/`에서 직접 읽은 값이며, 출처를 본문에 명시한다.
- run 하나당 문서 하나를 두고, 재학습 시 기존 문서를 덮어쓰지 않고 새 run 문서를
  추가한다.

## 공통 해석 기준

두 보고서가 반복해서 참조하는 기준선이다.

| 기준 | 값 | 의미 |
| --- | ---: | --- |
| `ln(3)` | 1.0986 | 3-class 균등추측의 cross-entropy. validation loss가 이 값을 넘으면 확률 출력이 균등추측보다 나쁨 |
| RouterArena 균등추측 macro F1 | 0.326 | RouterArena 8,375행 분포에서 무작위 예측의 기대 macro F1 |
| RouterArena base rate | easy 47.4% / medium 29.1% / hard 23.5% | 예측 그룹 분포를 이 값과의 차이(pp)로 해석해야 실제 신호가 보임 |
| v1 validation majority baseline | 47.9% (hard) | v1 자연분포 기준 |
| v2 validation majority baseline | 46.3% (easy) | v2 자연분포 validation 기준 |
