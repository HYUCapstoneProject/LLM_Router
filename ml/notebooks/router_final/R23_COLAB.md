# R2·R3 모델 집단 비교 Colab

2026-10-22 요청: EV-10에서 빠졌던 R2와 R3만 모델 집단 기준으로 학습하고 기존 라우터들과 G를 비교한다.

## 실행 순서

1. 배포 ZIP을 푼다. `router-final-source-r23.zip`을 기존 Drive의 `LLM_Router/router-final/runs/run001/code/`에 넣는다. 넣지 않은 경우 각 노트북의 소스 설정 셀에서 이 파일을 업로드할 수 있다.
2. `12a_r2_group.ipynb`를 Colab에서 열어 T4 GPU·Python 3.12로 실행한다. 시험 학습 → 본 학습 → Drive 보존 → dev/test 점수 저장을 수행한다.
3. `12b_r3_group.ipynb`를 같은 방식으로 실행한다.
4. 두 점수 저장이 끝나면 `13_r23_gain.ipynb`를 CPU에서 실행한다. dev 보정·λ 고정 → dev 리허설 → test → 결과 다운로드 순서다.

기존 실험 폴더가 다른 위치에 있으면 각 노트북 상단의 `RUN`만 바꾼다. 기존 `gain-lock`, `gain-test`, `lock`, `prepare`, `ours/data`, `ours/models/r1_group/manifest.json`, `length-inputs`, `inputs/test-data`가 필요하다. 질문 원본과 모델은 배포 묶음에 포함하지 않는다.

## 비교 조건

| 항목 | 고정 조건 |
|---|---|
| 학습 대상 | R2-group(pairwise), R3-group(utility-gap-weighted CE) 두 모델만 |
| 후보 | 기존 EV-10 `prepare/groups.json`의 원천별 low/medium/high 모델 집단 |
| 데이터 | 기존 `ours/data/blended`의 동일 행·분할·원천별 동일 총 가중치 |
| target | 각 집단에서 성공한 모델 비율. `quality_* = success_* = p_*` |
| R2 | 집단 해결 확률 차이로 순위 학습, 동률 가중치 .25 |
| R3 | p − λc 최선 집단을 CE로 학습, 최선·차선 차이로 가중. λ grid는 0/.05/.1/.2/.4/.8/1.6/3.2 |
| 모델·학습량 | ModernBERT-base 고정 revision, 512토큰, 3 epoch, seed 42, batch 8 × 누적 4, lr 2e−5, 최종 checkpoint |
| 비용 | 주 학습·평가 .2/.6/1.0 (1:3:5). 집단 평균 가격과 길이 반영은 τ=.8 평가 민감도 |
| 보정 | 기존과 동일한 dev 5분할 교차 isotonic. R2/R3 raw logit을 후보별로 보정 |
| G | 첫 시도 성공 A − 같은 비용 이하 고정 단계/혼합의 최선 성공 H(C). 상향 비용·실패 가중치 κ는 포함하지 않음 |
| 비교군 | 기존 gain-lock의 설정·보정·λ를 byte 단위로 보존. R1-BGE 기존 제외 유지 |
| 신뢰구간 | 기존 질문 묶음 단위 paired bootstrap 10,000회. R2/R3와 기존 방법의 쌍별 차이 포함 |

과거 대표 모델 R2/R3와 후보·target·학습 분포가 다르다. 결과가 달라질 가능성은 있지만 실행 전 성능 개선을 가정하지 않는다. 집단의 성공 비율은 연속 숫자여도 원래 답변의 연속 품질 점수와 다르다. R3 학습 λ와 평가에서 dev로 맞추는 λ도 서로 구분한다.

기존 R0/R1 데이터의 단일 결측 집단 보간 규칙을 그대로 사용한다. 본 평가에서는 기록이 없는 집단을 보간하지 않고 기존과 동일한 평가 행을 쓴다. test는 이전에 이미 본 데이터이므로 이 실험은 후속 비교이며 새로운 독립 검증으로 표현하지 않는다.

## 출력과 재실행

`run001/r23-extension/` 아래 `models/r2_group`, `models/r3_group`, `scores/`, `gain-lock/`, `gain-test/`에 저장한다. 원본 모델·결과·운영 배포는 수정하지 않는다.

- `gain-summary.md`: 모든 라우터의 G·비용 민감도 요약.
- `r0-r3-comparison.csv`: R0~R3 네 방법의 A/C/H/G와 95% 구간, 비용 조건별 비교.
- `gain.csv`, `gain-macro.csv`: 원천별/범위별 전체 값과 단계 선택 비율.
- `pairs.csv`: 같은 질문 bootstrap으로 계산한 방법 간 G 차이.
- `manifest.json`: 입력·출력 SHA, test 재사용 명시, 기존 점수·구간 재현 여부.

학습은 기존 trainer를 재사용하므로 중간 checkpoint에서 재개하지 않는다. 완료 모델은 무결성 검사 뒤 재사용한다. 미완료 출력은 자동 삭제하지 않는다. 실패 로그를 확인한 뒤 해당 미완료 폴더만 별도 보관하고 다시 실행한다. 평가 결과가 이미 있으면 덮어쓰지 않는다.

## 로컬 배포 묶음 생성

```bash
python ml/notebooks/router_final/build_r23_notebooks.py --output-dir /absolute/path/to/colab-r23
```

생성한 소스 ZIP의 전체 SHA와 내부 파일별 SHA를 노트북에서 검사한다. 코드가 바뀌면 소스와 노트북을 함께 다시 생성한다. 노트북 작성·CPU 테스트를 완료한 것과 실제 Colab GPU 학습 완료를 구분한다.
