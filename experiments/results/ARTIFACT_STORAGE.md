# 실험 결과와 원본 보관 범위

2026-09-25 통합 시 Git에는 재사용 가능한 ML 코드, 실험 조건, 집계 결과, 분석 문서, 그래프와 검증·보관 영수증을 포함했다. 질문 원문을 포함하는 Parquet, 행 단위 예측 CSV 및 일회성 실행 로그·스크립트는 포함하지 않는다. 원본 작업 공간과 별도 보존본의 파일을 삭제하거나 변경하지 않았다.

- G1 전체 평가 archive: [보관 영수증](routing-generalization-v1-20260925/archive-receipt.json). 개인 Drive의 `LLM_Router/routing-generalization-v1-20260925/evaluation-final-1790317240`, SHA-256 `ff3c72721881f7dbec6ec75ad47c7ecf8b03998413354675c7625ef8487448f0`이다. 학습 가중치는 기존 R1/R3 세션 폴더에 별도 보관되어 있다.
- A 실험의 기존 연구 결과 보관: [영수증](routing-objectives-continual-v1-A-20260924/report-archive-receipt.json). 후속 진단의 행별 중간 파일은 원본 작업 공간에 유지된다.
- 각 실험의 원래 manifest는 **전체 실행·보관 묶음**의 해시 목록이다. Git checkout의 파일 목록이 아니며, 원본 무결성을 위해 부분 목록으로 다시 쓰지 않았다. 제외된 파일을 사용하는 재현 명령은 해당 원본 묶음을 복원한 후 실행한다.
- 2026-09-28 원본 작업 공간에만 남아 있던 파일 중 main 문서·코드가 직접 참조하는 재현 입력과 영수증을 추가로 반입했다. 대상은 길이 검증의 `inputs.json`·`bundle-manifest.json`, 질문 원문이 없는 512 기준 점수 `recovery/historical512/A/*.parquet`, `source-manifest.json`에 기록된 분석 코드 `recovery/r0|r1/*.py`, 보관·재분석 검증 JSON, `runtime/*-status.json`, A 진단의 `category-baselines/original-audit.parquet`이다. 질문 원문 Parquet, 행 단위 CSV, 일회성 실행 로그·스크립트, 중복 보고서 tar.gz는 위 원칙대로 계속 제외한다. 영수증의 로컬 경로는 해시 보존을 위해 원문 그대로 둔다.
- `routing-method-review-20260925/manifest.json`은 해당 소규모 분석 폴더 전체의 해시 목록이다.
- Colab 실행 스크립트의 보관 계정 지정은 재사용을 위해 명시적 옵션으로 바꾼다. 실제 실험 당시 코드의 해시는 동결한 run plan과 보관 묶음에 남으므로 현재 소스의 해시와 혼동하지 않는다.

모든 결과는 연구 평가다. Git 병합 또는 파일 보관은 운영 모델 승인이나 `CURRENT.json` 활성화를 의미하지 않는다.
