# LLM Router ML 실행 진입점

데이터·학습 목표·평가·재학습·캐시의 기준은 [ML 최종 실험 문서](../docs/ml/README.md)를 따른다. 이 파일에는 설치와 코드 진입점만 둔다. 기존 운영 v2와 신규 비교 실험은 별도 config·artifact 계약이며, 기본 CLI가 최신 연구 설계를 전부 구현했다고 가정하지 않는다.

## 환경

저장소 루트에서 Python 3.11–3.13 환경을 사용한다. Colab/GPU runtime의 고정 버전과 실행 위치는 [실행·산출물 관리](../docs/ml/README.md#execution)를 따른다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r ml/requirements-lock.txt
.venv/bin/pip install --no-deps -e ml
```

학습·평가·게시 의존성은 [pyproject.toml](pyproject.toml)의 optional dependency와 해당 runtime lock을 확인해 실행 환경별로 설치한다. 문서 정리만으로 dependency를 갱신하지 않는다.

선택 실행용 비교 실험은 기본 환경에 [requirements-comparison.lock](requirements-comparison.lock)을 추가 설치한다. scikit-learn 계열은 `comparison` extra로 분리되어 있으며 기존 CPU 전처리 이미지의 기본 lock에는 포함하지 않는다.

```bash
.venv/bin/pip install -r ml/requirements-comparison.lock
```

## 코드 구조

| 경로 | 역할 |
|---|---|
| [routing_data](src/llm_router_labeling/routing_data) | RouterBench+xRouteBench 공통 데이터와 train-only threshold 선택 |
| [comparison](src/llm_router_labeling/comparison) | TF-IDF 기준선과 분류 평가 |
| [objective_comparison](src/llm_router_labeling/objective_comparison) | outcome 구성, R0/R1/R2/R3, validation lock과 최종 평가 |
| [training](src/llm_router_labeling/training) | 기존 RouterBench 라벨링, ModernBERT 공통 코어, feedback 전처리 |
| [evaluation](src/llm_router_labeling/evaluation) | RouterArena 전처리와 난이도 정렬 평가 |
| [deployment](src/llm_router_labeling/deployment) | package, gate, candidate/approved/CURRENT 게시 |
| [shared](src/llm_router_labeling/shared) | 데이터 IO, 정규화, object store |
| [kubeflow](kubeflow) | KFP IR와 Vertex feedback workflow |
| [config](config) | versioned 데이터·학습·평가·게시 설정 |
| [notebooks](notebooks) | Colab 진입점; 각 notebook의 계약 버전 확인 필요 |

## CLI 목록

진입점 등록의 기준은 [pyproject.toml](pyproject.toml)이다. 학습·외부 호출을 시작하기 전에 각 명령의 `--help`와 [구현 보완 항목](../docs/ml/README.md#implementation-gaps)을 확인한다.

| 명령 | 역할 |
|---|---|
| `routing-comparison-data` | xRouteBench 다운로드/검증, 통합 데이터 구성 |
| `routing-tfidf-baseline` | TF-IDF+Logistic Regression |
| `routing-objective-comparison` | outcome 준비, 목표별 학습, validation 동결, test/RouterArena 평가 |
| `routerbench-labeling` | 기존 RouterBench 원천 다운로드·라벨링·전처리 |
| `routerarena-preprocessing` | 평가 입력 조립·품질 필터·겹침 검사 |
| `routerarena-evaluate` | checkpoint의 외부 난이도 정렬 평가 |
| `escalation-feedback-preprocessing` | feedback 검증·정규화·재학습 입력 생성 |
| `modernbert-train` | 공통 학습 코어의 Docker/Vertex 진입점 |
| `modernbert-colab-train` | Colab 실행·checkpoint 동기화 |
| `modernbert-package` | 패키지 조립·검증 |
| `modernbert-deploy-gate` | 지표·baseline 기준 검사 |
| `modernbert-publish` | 모델 게시·CURRENT 조회·rollback |

```bash
.venv/bin/routing-comparison-data --help
.venv/bin/routing-objective-comparison --help
.venv/bin/routing-objective-comparison train --help
.venv/bin/routerarena-evaluate --help
```

설치된 entry point가 아직 없다면 같은 모듈의 도움말을 직접 확인할 수 있다.

```bash
PYTHONPATH=ml/src .venv/bin/python -m llm_router_labeling.routing_data.cli --help
PYTHONPATH=ml/src .venv/bin/python -m llm_router_labeling.objective_comparison.cli --help
```

`ordinal`이라는 기존 CLI 값은 **일반 3-class cross entropy인 R0**를 뜻한다. `regret`는 구현되어 있어도 현재 최소 실험의 필수 비교군이 아니다. RouterDC·BGE+kNN·캐시·feedback 실험 전체의 실행기가 준비되어 있다는 의미도 아니다.

R0 교체 브랜치의 로컬 release 준비기는 `PYTHONPATH=ml/src .venv/bin/python ml/scripts/prepare-r0-release.py --help`로 확인한다. 필수 실제 baseline·pilot·8192 평가 SHA 및 출력 계약은 [R0 release 문서](../docs/ml/R0_RELEASE_CONFIG.md)를 따른다.

## 실행과 검증

공통 데이터 동결 → 소규모 pilot → 비교 학습 → validation 정책 동결 → 최종 test 순서를 따른다. 운영 기본 config에 과거 v2의 8192-token·balanced 설정이 있어도 신규 공정 비교의 설정으로 자동 재사용하지 않는다.

소스 변경 시 관련 테스트를 실행한다. 전체 회귀 검사가 필요한 경우의 진입점은 다음과 같다.

```bash
.venv/bin/python -m unittest discover -s ml/tests -v
```

Docker build context는 저장소 루트이며 image 역할·digest·IR 갱신과 Vertex 실행은 [실행·산출물 관리](../docs/ml/README.md#execution), 온라인 계약과 version 정책은 [재학습 파이프라인](../docs/ml/README.md#retraining)에 정의한다.

## 산출물과 과거 실행

`ml/data/`와 `ml/artifacts/`는 Git 제외 대상이다. 데이터·가중치·로그는 버전과 checksum을 기록한 저장소에 따로 보존한다. Git commit에는 실행한 코드·config와 결과 색인이 필요하다.

기존 v1/v2 명령 예시, Docker smoke 결과, 설치 문제 기록, 학습 보고서 원문은 `fa09cb8` 이전 history에서 조회할 수 있다. 현재 문서에 과거 실행 절차를 중복 유지하지 않는다.
