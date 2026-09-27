# R1→R0 전이 실험 노트북

상세 계약은 [별도 실험 설계서](../../../docs/ml/R1_TO_R0_TRANSFER_EXPERIMENT.md), 전체 맥락은 [ML 단일 README](../../../docs/ml/README.md#r1-to-r0-transfer-experiment)를 따른다. `r1-to-r0-transfer-v2-final-only` 실행 kit의 데이터·pin·모델 archive 목록·정책 준비를 완료했으며 GPU 학습·평가 결과는 아직 없다.

| 순서 | 파일 | 역할 |
|---|---|---|
| 0 | [00_prepare.ipynb](00_prepare.ipynb) | 원본·분할·공통 logger 검증과 학습 준비. 별도 pilot 없음 |
| A | [01_r0_continued.ipynb](01_r0_continued.ipynb) | 기존 R0의 CE 추가 학습. 중간 전체 dev 평가 없음 |
| B | [02_r1_to_r0.ipynb](02_r1_to_r0.ipynb) | R1 encoder + 새 head/classifier를 CE로 학습. 중간 전체 dev 평가 없음 |
| C | [03_r1_frozen.ipynb](03_r1_frozen.ipynb) | 설명용. C는 04에서 함께 평가하므로 별도 실행 불필요 |
| 4 | [04_compare_results.ipynb](04_compare_results.ipynb) | 전체 학습 후 baseline/final dev → 선택 고정 → 보호 평가·최종 보고서 |

A/B는 같은 데이터·200+6×100 step·seed 42/43/44, C는 학습하지 않는 단일 참조다. **학습 512 / 평가 8192**다. 이미 수행한 길이 검증의 512 평가, 별도 pilot, 중간 전체 dev 추론은 이번 실행에서 생략한다. 공통 trainer의 작은 validation 48행은 내부적으로 유지되며 사용자가 따로 실행하지 않는다. 학습 길이를 변경하려면 별도 실험으로 정의한다. 원본 R0/R1은 동일 A 초기 코호트여야 하며 G1 R1-natural을 묵시적으로 대체하지 않는다.

## 사용자가 Colab에서 직접 실행하기

1. 제공된 **`user-kit/transfer-experiments` 폴더 전체**를 개인 `alexyoo0506@gmail.com` Drive의 **`MyDrive/LLM_Router/`** 아래에 업로드한다. 최종 경로는 `MyDrive/LLM_Router/transfer-experiments/`다.
2. **00_prepare.ipynb를 전체 실행**한다. 개인 Drive에 이미 보관된 R0/R1 tar archive의 SHA를 검증하여 모델을 복구하고, 공통 logger 추론과 입력 준비를 수행한다.
3. **01과 02를 각각 별도 Colab GPU 세션에서 전체 실행**한다. 두 노트북은 같은 RUN_ID와 준비된 입력을 사용하며 자기 arm의 세 seed를 학습한다.
4. **둘 다 완료되면 04_compare_results.ipynb를 전체 실행**한다. baseline/final dev → 선택 동결 → gate/forget/RouterArena 평가 → 최종 분석 순서다. **03은 별도 실행하지 않아도 된다.**

경로·SHA·source pin은 kit에 채워져 있다. 사용자가 checksum을 계산하거나 빈 template을 작성할 필요가 없다. 최초 제공된 실행은 기본 경로·RUN_ID 그대로 사용한다. 모델 archive가 이미 복구되었거나 완료 phase가 있으면 검증 후 재사용한다. 아직 원격 main에 게시되지 않은 파일이므로 main을 clone하여 실행하지 않고 제공한 소스 ZIP을 사용한다. Vertex/GCS/GCP 결제 작업은 실행하지 않는다.

```text
transfer-experiments/
  source/transfer-source.zip, transfer-source.sha256
  inputs/experiment-A/
    data/                         # 검증된 6개 Parquet
    source_pins.json               # 실제 원본 file hash·모델·정책 pin
    model-archives.json            # 개인 Drive 기존 tar 위치·SHA
    r1-policy.json, runtime.json, data-manifest.json
  runs/                           # 00 이후 생성되는 학습·평가 결과
```

[`prepare_local_data.py`](prepare_local_data.py)가 기존 봉인 artifact에서 6개 데이터를 정규화하고 pin·정책·archive 목록을 자동 생성한다. [`prepare_models.py`](prepare_models.py)는 00에서 기존 개인 Drive tar의 검증된 파일만 복구한다. [`build_bundle.py`](build_bundle.py)는 소스 ZIP과 SHA를 만드는 개발용 도구다. 이 준비 작업의 결과를 kit에 제공하므로 사용자가 다시 수행할 필요는 없다. `source_pins.example.json`은 개발용 형식 예시로만 남긴다.

## 준비된 실제 입력

재학습 입력도 **기존 RouterBench+xRouteBench**다. 새로운 분야·다른 출처에 적응시키는 실험은 미래 연구이며 이번 졸업프로젝트 범위에서 제외한다. 저장된 A R0/R1 모델을 불러오므로 초기학습을 반복하지 않는다. 초기 train은 전환 적응·rehearsal에, 미학습 feedback 분할은 상향 교정 대리 신호에 사용한다. 공통 feedback 로그는 00 실행 시 원본 R0의 추론으로 생성한다.

준비된 `data/`의 6개 Parquet는 기존 봉인 분할을 정규화한 것이며 새로 나누지 않았다.

| 파일 | 기존 역할 |
|---|---|
| train.parquet | D_initial_train |
| dev.parquet | D_dev |
| gate.parquet | D_gate |
| feedback.parquet | D_stream |
| forget.parquet | D_retention_test |
| routerarena.parquet | 고정 RouterArena 진단 |

필수 열: `sample_id`, `duplicate_group_id`, `model_input_text`, `difficulty_label`(easy/medium/hard), `source_dataset`, `category`. ID·중복 그룹은 문자열로 정규화한다. 같은 duplicate group 안에 서로 다른 질문이 여러 행 있을 수 있으며 이를 삭제하지 않는다. RouterArena full 8,379행 중 240행은 같은 본문을 공유한 다른 질문으로 그룹 ID가 반복되는 정상 사례다. sample ID·duplicate group·NFKC/공백/대소문자 정규화 입력이 **역할 사이**에 겹치면 실패한다. 비교의 bootstrap은 그룹의 모든 질문을 함께 재표집한다. 의미 중복 방어는 원본 `duplicate_group_id` 생성 품질에도 의존한다.

후보 outcome 열(`quality_low/medium/high`, `actual_cost_low/medium/high`)이 있으면 실제 후보 순서와 출처가 검증된 값만 보존한다. 없으면 품질·비용 수치를 생성하지 않는다. RouterArena의 난이도 라벨로 Claude 응답 성공률을 만들지 않는다.

R1 정책은 기존 봉인 정책에서 자동 추출한 `r1-policy.json`을 사용한다. source별 λ·decision cost와 후보 순서는 원본에 고정되어 있다. RouterArena에는 해당 후보 pool이 없으므로 cost-free sigmoid argmax의 난이도 진단만 수행한다. 런타임은 [`modernbert_runtime_cu124_v1.json`](../../config/modernbert_runtime_cu124_v1.json)에서 복사한 `runtime.json`으로 고정한다. 모든 세션이 같은 Drive 입력을 읽는다.

## 결과 디렉터리

```text
<BASE>/runs/<RUN_ID>/
  plan.json / plan.sha256.json
  shared-data/bridge.parquet, cycle-01.parquet … cycle-06.parquet
  arms/r0_continued/seed-42/{bridge,cycle-01,…}/
  arms/r1_to_r0/seed-42/{conversion,bridge,cycle-01,…}/
    <phase>/stage-contract.json, attempts/, checkpoints/, model/, completed.json
  scores/<arm>/seed-<seed>/<phase>/<role>/8192/
    predictions.parquet, receipt.json
  selection-lock.json
  report/metrics.csv, source-category-metrics.csv, paired-comparisons.csv
  report/retention-check.csv
  report/status.json, REPORT.md, *.png
```

다른 seed도 같은 구조이며 C는 `r1_frozen/seed-0/baseline` 점수만 생성한다. 같은 arm/seed를 두 세션에서 실행하지 않는다. 사용자가 A/B를 각각 한 Colab 세션에서 직접 진행할 수 있다. A/B의 모든 seed가 끝나면 04 노트북에서 C를 포함한 최종 평가를 한 번에 수행한다. baseline/final dev 평가 → selection lock → gate/forget/RouterArena의 baseline/final 평가 → 분석 순서다. bridge·중간 cycle 성능 곡선은 만들지 않는다.

학습 단계마다 optimizer를 새로 만들지만, **같은 단계의 중단 복구**는 25 step 간격의 checkpoint에 저장된 optimizer·scheduler·RNG·cursor를 복원한다. 완료 phase는 모델 hash 검사 후 재사용한다. 사용자가 `alexyoo0506@gmail.com` Drive의 5TB 용량을 확인했으므로 용량 입력·검사 절차는 생략한다. 해당 계정에 단계별 최종 모델과 최근 checkpoint 2개를 보관하며 임의로 과거 단계 모델을 삭제하지 않는다.

`lock` 후에는 학습을 거부한다. `gate/forget/routerarena` 추론은 lock과 명시적 최종 평가 플래그가 있어야 실행된다. 원본 기준선과 최종 checkpoint만 평가하며 외부 점수로 승자를 다시 고르지 않는다. `status.json`에 누락 결과가 있으면 완료 보고서로 취급하지 않는다. 망각은 A의 원본 R0, B의 원본 R1 정책 대비 source-macro-F1 하락폭을 seed별로 확인한다. B의 변화에는 출력 의미·선택 정책 전환 효과도 포함된다. 서비스 승격은 별도 판단이다.

## 로컬 검증

```bash
PYTHONPATH=ml/src python -m unittest discover -s ml/tests -p 'test_transfer_experiment.py'
python ml/notebooks/r1_to_r0_transfer/experiment.py --help
```

테스트는 작은 합성 fixture의 누수 방지·공통 window·출력층 전환·계약 검사를 확인한다. 합성 fixture를 실제 연구 데이터나 완료된 학습 결과로 사용하지 않는다.

### Colab Python 버전 오류

`AssertionError: 계약에 맞는 Colab Python 런타임이 필요합니다`가 나오면 런타임 → 런타임 유형 변경 → 런타임 버전을 **2026.07 (Python 3.12.13)**로 선택하고 T4를 유지한 채 처음부터 모두 실행한다. 기본 최신 Python은 기존 torch 2.5.1 실험 계약과 다를 수 있다. assertion 삭제나 Python 3.13 허용으로 우회하지 않는다. 기존 업로드 데이터·Drive 모델은 다시 올리지 않아도 된다. 00·01·02·04 각각 같은 설정을 사용한다. [Google 공식 버전 목록](https://github.com/googlecolab/backend-info#colabs-past-runtime-versions).
