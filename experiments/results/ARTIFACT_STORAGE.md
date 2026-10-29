# 실험 결과와 원본 보관 범위

2026-09-25 통합 시 Git에는 재사용 가능한 ML 코드, 실험 조건, 집계 결과, 분석 문서, 그래프와 검증·보관 영수증을 포함했다. 질문 원문을 포함하는 Parquet, 행 단위 예측 CSV 및 일회성 실행 로그·스크립트는 포함하지 않는다. 원본 작업 공간과 별도 보존본의 파일을 삭제하거나 변경하지 않았다.

- G1 전체 평가 archive: [보관 영수증](routing-generalization-v1-20260925/archive-receipt.json). 개인 Drive의 `LLM_Router/routing-generalization-v1-20260925/evaluation-final-1790317240`, SHA-256 `ff3c72721881f7dbec6ec75ad47c7ecf8b03998413354675c7625ef8487448f0`이다. 학습 가중치는 기존 R1/R3 세션 폴더에 별도 보관되어 있다.
- A 실험의 기존 연구 결과 보관: [영수증](routing-objectives-continual-v1-A-20260924/report-archive-receipt.json). 후속 진단의 행별 중간 파일은 원본 작업 공간에 유지된다.
- 각 실험의 원래 manifest는 **전체 실행·보관 묶음**의 해시 목록이다. Git checkout의 파일 목록이 아니며, 원본 무결성을 위해 부분 목록으로 다시 쓰지 않았다. 제외된 파일을 사용하는 재현 명령은 해당 원본 묶음을 복원한 후 실행한다.
- 2026-09-28 원본 작업 공간에만 남아 있던 파일 중 main 문서·코드가 직접 참조하는 재현 입력과 영수증을 추가로 반입했다. 대상은 길이 검증의 `inputs.json`·`bundle-manifest.json`, 질문 원문이 없는 512 기준 점수 `recovery/historical512/A/*.parquet`, `source-manifest.json`에 기록된 분석 코드 `recovery/r0|r1/*.py`, 보관·재분석 검증 JSON, `runtime/*-status.json`, A 진단의 `category-baselines/original-audit.parquet`이다. 질문 원문 Parquet, 행 단위 CSV, 일회성 실행 로그·스크립트, 중복 보고서 tar.gz는 위 원칙대로 계속 제외한다. 영수증의 로컬 경로는 해시 보존을 위해 원문 그대로 둔다.
- `routing-method-review-20260925/manifest.json`은 해당 소규모 분석 폴더 전체의 해시 목록이다.
- Colab 실행 스크립트의 보관 계정 지정은 재사용을 위해 명시적 옵션으로 바꾼다. 실제 실험 당시 코드의 해시는 동결한 run plan과 보관 묶음에 남으므로 현재 소스의 해시와 혼동하지 않는다.

모든 결과는 연구 평가다. Git 병합 또는 파일 보관은 운영 모델 승인이나 `CURRENT.json` 활성화를 의미하지 않는다.

## 학습 모델 가중치 (2026-10-29)

가중치는 파일 하나가 GitHub 커밋 한도(100MB)를 넘으므로 Git에 커밋하지 않고 이 저장소의 Release [`models-20261029`](https://github.com/HYUCapstoneProject/LLM_Router/releases/tag/models-20261029)에 첨부했다. 저장소가 비공개라서 Release도 팀원만 볼 수 있다. Drive의 원본(`LLM_Router/router-final/runs/run001/ours/models/r0_group/`, `LLM_Router/transfer-experiments/inputs/experiment-A/models/r0/`)은 그대로 둔다.

| 첨부 파일 | 모델 | `model.safetensors` SHA-256 |
|---|---|---|
| `router-r0-group-20261017.tar` | R0-집단(EV-10 `03_ours`, 서비스 라우터로 정함, 배포 전) | `426d70c2b4b9ba5bcb48a6889699a91c491cf49c7bb0498754a36e0c23165f67` |
| `router-r0-operational-1.0-20260927.tar` | 운영 1.0 `router-modernbert-r0-a-20260927-v1` | `e1a40fec7d269d8f925b5391e4f5e75df5d0fedbb1e37b4eea96521e27b1a6a7` |

```bash
gh release download models-20261029 -R HYUCapstoneProject/LLM_Router
shasum -a 256 -c SHA256SUMS-release.txt
```

각 묶음에는 `model/`(가중치·config·tokenizer), 학습 manifest·지표, `MODEL_CARD.md`, ModernBERT의 Apache-2.0 `LICENSE`, 파일별 `SHA256SUMS`가 들어 있다. R0-집단 묶음에는 보정을 다시 맞출 때 쓰는 dev·test logit(`scores/scores.parquet`, `sample_id`만 있고 질문 원문 없음)도 있다. 질문 원문과 모델 응답은 어느 묶음에도 넣지 않았다.

**이용 조건(2026-10-29 확인).** 베이스 모델 ModernBERT-base는 Apache-2.0이다. 학습 데이터 가운데 RouterBench와 xRouteBench는 코드 저장소가 MIT이고 데이터셋 카드에는 라이선스가 없다. R0-집단만 추가로 쓴 LLMRouterBench(README에 MIT 배지만 있음)와 SPROUT(명시 없음)는 라이선스가 분명하지 않다. 팀 내부 보관과 연구 사용에는 문제가 없지만, 외부 공개는 아래를 먼저 확인한다.

- Release는 저장소 공개 여부를 따른다. 가중치만 공개하려면 저장소를 공개로 바꾸지 말고 별도 공개 저장소나 Hugging Face Hub에 모델 카드와 함께 올린다.
- R0-집단은 LLMRouterBench와 SPROUT 저자에게 학습 가중치 공개 가능 여부를 확인한다. 운영 1.0은 공개 시점에 두 데이터셋 카드의 라이선스를 다시 본다.
- 원 벤치마크 조건(예: ARC의 CC-BY-SA)이 가중치 공개에 걸리는지 다시 본다.
- 공개본에도 질문 원문은 넣지 않는다.
