# RF-06 버전 정책 대 전체 재학습 노트북

기준 문서: [RF-06 계획서](../../../docs/ml/VERSION_POLICY_EFFICIENCY_EXPERIMENT.md) · 실험 코드: RF-06

| 노트북 | 장비 | 하는 일 | 시간(추정) |
|---|---|---|---|
| `00_pilot` | T4 | smoke → pilot 학습 곡선 5점 → n0·stream 크기 결정 → 9 round batch 봉인 | smoke 10분, pilot 2.5~3시간, 봉인 15분 |
| `01_p1_version_policy` | T4 | P1 9 round | 1.5~2.5시간 |
| `02_p2_full_retrain` | T4 | P2 9 round (선택: `ARM = 'p2_eq'`) | 6~8시간 (p2_eq 1.5~2.5시간) |
| `03_compare` | CPU | 품질·계산량·효율 비교, 판정, 보고서 초안 | 몇 분 |

01과 02는 서로 독립이라 다른 GPU 세션에서 동시에 돌려도 된다(p2_eq만 P1의 같은 seed 뒤). 시간은 RT-04 실측(T4, 512토큰, batch 32에서 2.82초/step)으로 잡았고, pilot의 실제 step 시간으로 다시 추정한다.

## 준비 (로컬, 한 번)

EV-10 00_prepare 출력과 원본 그룹 id가 있는 checkout에서 실행한다.

```bash
python ml/notebooks/version_policy/plan.py build --repo . --data-root <ml/artifacts가 있는 checkout> --output-dir <작업 폴더>/data
```

```bash
python ml/notebooks/version_policy/build_kit.py --output-dir <작업 폴더>/code
```

Drive 구성(`MyDrive/LLM_Router/version-policy/runs/run001/`):

```text
code/version-policy-source.zip, .sha256   ← build_kit.py
data/pool.parquet, dev.parquet, test.parquet, outcome-manifest.json, manifest.json   ← plan.py build (질문 원문 포함, 저장소에 넣지 않음)
pilot/n<크기>/   plan/   arms/<arm>/seed<N>/round<kk>/   compare/   ← 노트북이 씀
```

로컬 build 결과(2026-10-20): pool 35,017 / dev 7,591 / test 17,662행. pool 라벨 easy .548 · medium .262 · hard .190.

## 코드

| 파일 | 내용 |
|---|---|
| `plan.py` | `build`(로컬 표), `pilot`(크기별 학습·dev 채점), `seal`(크기 규칙, 분할, M0 채점, batch 봉인), 공통 학습·채점 함수 |
| `rounds.py` | arm·seed별 round 실행. P1 anchor 규칙, P2 누적, P2-eq step 상한. round마다 receipt와 `_SUCCESS`, 끝난 round는 건너뜀 |
| `compare.py` | dev·prequential·test 품질, 라우팅 지표, 계산량, bootstrap 판정, 그림·보고서 |
| `config.json` | pilot 크기, 규칙 상수, round 수, 버전 정책 상한, seed, 판정 기준 |
| `build_kit.py` | Colab 소스 ZIP (ml/src, ml/config, 이 폴더, EV-10 `ours.py`·`prepare.py`) |

- 학습은 EV-10·RT-05의 `train_objective`(ordinal = R0 3분류 CE)를 그대로 쓴다. 이어 학습을 위해 `initial_model_dir` 인자를 추가했다(가중치만 가져오고 optimizer는 새로, 지정하지 않으면 기존과 같음).
- 채점은 학습과 같은 512토큰 토큰화의 logit이다. test 채점은 질문 원문만 쓰고, 결과(풀 확률)는 03에서만 읽는다.
- 모델 가중치는 P1 anchor(x.0)와 각 arm의 마지막 round만 Drive에 남긴다. pilot 가중치는 00의 4절에서 지울 수 있다.
- 단위 테스트: `ml/tests/test_version_policy_experiment.py`.

## 로컬 smoke (CPU, 약 15분)

```bash
PYTHONPATH=ml/src python ml/notebooks/version_policy/plan.py pilot --repo . --data-dir <data> --run-dir <scratch> --smoke --device cpu
```

이어서 `seal --smoke --n0 200`, `rounds.py --arm p1|p2|p2_eq --smoke`, `compare.py`를 같은 `<scratch>`로 실행한다. 00 노트북의 1절과 같은 순서다.
