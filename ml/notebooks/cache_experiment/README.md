# 캐시 실험 노트북 (C0/C1/C2/C3)

상세 계약은 [캐시 실험 계획서](../../../docs/ml/CACHE_EXPERIMENT.md)를 따른다. 대량 추론은 **개인 Colab GPU**에서, 지연·pgvector 측정(05)만 **로컬 Mac**에서 한다. LLM을 부르지 않고, 운영 DB·GCS·설정을 바꾸지 않는다.

| 순서 | 파일 | 실행 위치 | 역할 | 사람 작업 |
|---|---|---|---|---|
| 0 | [00_prepare.ipynb](00_prepare.ipynb) | Colab GPU | 입력 SHA, 라우터 anchor 관문과 원문 라우팅, 512토큰 초과 제외, pool, 채점기 자기 일치, 운영 패리티 관문, 임베딩, D2 원질문·작성 시트 | — |
| 1 | [01_d1_replay.ipynb](01_d1_replay.ipynb) | Colab GPU | trace 봉인, C0~C3·용량 민감도 재생, bootstrap | — |
| 2 | [02_d2_sheets.ipynb](02_d2_sheets.ipynb) | Colab | 작성 시트 검증 → 블라인드 주석 시트 → 일치도·조정 → 라벨 봉인 | 작성·주석·조정 |
| 3 | [03_d2_calibrate.ipynb](03_d2_calibrate.ipynb) | Colab GPU | 후보표·격자 평가, 보정 정책 1회 봉인 | — |
| 4 | [04_d2_test.ipynb](04_d2_test.ipynb) | Colab GPU | test 1회 평가(시도 기록), 보정 정책 D1 재생 | — |
| 5 | [05_latency.ipynb](05_latency.ipynb) | **로컬 Mac** | 구간 지연(CPU 2스레드), 라우터 CPU 재분류 대조, HNSW recall, 실제 pgvector 판정 일치 | Docker |
| 6 | [06_report.ipynb](06_report.ipynb) | Colab 또는 로컬 | 표·그림·`REPORT-draft.md`·검토용 zip | 결론 작성 |

## Colab 실행

1. 저장소 루트에서 kit을 만든다. 입력 SHA를 확인한 뒤 `ml/artifacts/cache-experiment-v1-user-kit/cache-experiment`(약 200MB)를 만든다.

```bash
python3 ml/notebooks/cache_experiment/build_kit.py
```

2. 그 `cache-experiment` 폴더를 개인 Drive의 `MyDrive/LLM_Router/`에 올린다. 다음 두 입력은 이미 Drive에 있는 것을 SHA 확인 후 읽기 전용으로 쓴다.
   - 라우터 로그: `retraining-effect/logger/logger-log.parquet`
   - 운영 R0 패키지: `transfer-experiments/inputs/experiment-A/models/r0` (운영 승인 패키지와 파일 4개가 바이트 단위로 같다)
3. Colab 런타임 버전 2026.07(Python 3.12)과 GPU T4를 고르고, `notebooks/`의 노트북을 순서대로 **전체 실행**한다. 첫 셀이 Drive 연결, 소스 ZIP·파일별 SHA 확인, torch 2.5.1(cu124)·transformers 4.48.3 설치를 한다.
4. 산출물은 Drive `cache-experiment/runs/<RUN_ID>/`에 write-once로 쌓인다. D2 시트는 이 폴더의 CSV를 내려받아 채운 뒤 같은 폴더에 올린다.

## 로컬 실행 (05)

```bash
python3.11 -m venv .venv-cache
```

```bash
.venv-cache/bin/pip install -r ml/notebooks/cache_experiment/requirements.txt
```

저장소 루트에서 만들고 이 venv를 Jupyter 커널로 쓴다. 05는 Google Drive 데스크톱 동기화 폴더의 `cache-experiment/runs/`를 자동으로 찾아 Colab과 같은 run을 이어 쓴다. 입력은 이 체크아웃의 `ml/data`·`ml/artifacts`를 쓴다. worktree에서 실행하면 `CACHE_EXP_DATA_ROOT`를 원본 체크아웃으로 지정한다. Docker가 실행 중이어야 한다.

| 환경 변수 (로컬) | 기본값 |
|---|---|
| `CACHE_EXP_DATA_ROOT` | 이 저장소 루트 |
| `CACHE_EXP_RUN_ID` | `run001` (Colab도 같은 기본값) |
| `CACHE_EXP_OUTPUT_ROOT` | Drive 동기화 폴더의 `cache-experiment/runs`, 없으면 `<DATA_ROOT>/ml/artifacts/cache-experiment-v1` |
| `CACHE_EXP_DRIVE_LOGGER` | Drive 동기화 폴더의 라우터 로그 |

## 구성

| 파일 | 내용 |
|---|---|
| [`cache_experiment.py`](cache_experiment.py) | 운영 규칙 재현(정규화 키, 조회, 리랭커 판정), pool 결합, 재사용 채점기, trace·재생, 지표·bootstrap, D2 선정·주석·보정. 순수 Python/numpy |
| [`models.py`](models.py) | 서빙과 같은 BGE·리랭커 추론과 운영 패리티 검사 |
| [`latency.py`](latency.py) | 로컬 pgvector 컨테이너, 운영 SQL(`CacheService.java`에서 읽음), 구간 지연, HNSW recall |
| [`pipeline.py`](pipeline.py) | 노트북이 부르는 단계 함수. 각 단계는 write-once로 산출물을 봉인한다 |
| [`report.py`](report.py) | 봉인 산출물로 표·그림·보고서 초안 생성 |
| [`experiment_config.json`](experiment_config.json) | 입력 SHA, 모델 revision, 라우터 파일 SHA, 서비스 입력 한도, 운영 설정값, D1·D2·보정·지연 조건 |
| [`build_kit.py`](build_kit.py) | Colab kit 생성(소스 ZIP·입력 복사·SHA 확인) |
| [`annotation_tool.py`](annotation_tool.py) | 주석 시트로 브라우저 주석 도구(HTML 한 파일) 생성: 한 문항씩 판정, 시트 id 숨김, 자동 저장, 시트 형식 그대로 CSV 내보내기 |
| `requirements.txt` / `requirements-colab.txt` | 로컬 / Colab 고정 버전 |

## 핵심 규칙

- **운영 재현:** exact 키는 운영 Java와 같은 공백 규칙(테스트 fixture로 고정), 조회 SQL은 실행 시점에 `CacheService.java`에서 읽는다. REAL 임계값의 float32 반올림까지 맞춘다. 00의 패리티 관문은 8월 30일 배포 스택의 73개 프로브 점수를 재현해야 통과한다.
- **장치:** GPU(fp32, TF32 끔)로 계산해도 운영 패리티 관문이 서빙(CPU) 점수와 10⁻⁴ 안에서 같아야 진행한다. 보고서는 임계값 근처 판정 수를 함께 적는다.
- **서비스 입력 한도:** BGE 512토큰을 넘는 질문은 캐시 대상이 아니므로 pool에서 제외하고, D2 변형도 이 한도를 넘으면 거부한다.
- **라우팅:** 운영 R0 패키지(파일 4개 SHA 고정)로 요청 원문을 분류한다(서비스와 같은 입력). 재학습 실험의 로거 로그는 관문이다: 실행 환경의 라우터가 로거 입력(`model_input_text`)에서 로그 판정을 99% 이상 재현해야 한다.
- **기록 답변:** RouterBench는 5-shot 실행의 응답(stream 점수·비용과 행마다 일치 확인), xRouteBench는 `llmrouter_generic`의 응답·정답·응답 시간을 쓴다. 키와 임베딩은 사용자 질문(`query_text`)이다.
- **재사용 채점:** 다른 질문의 답이 나가면 현재 질문 기준으로 다시 채점한다. 객관식은 선택 글자와 선택지 문장이 모두 맞아야 정답이다. 채점할 수 없으면 unknown으로 두고 품질 범위를 낸다.
- **봉인:** 산출물은 write-once다. 다른 내용으로 다시 쓰려 하면 새 run 디렉터리를 요구한다. 보정 정책은 한 번 봉인한다. D2 test는 모든 시도를 `test-lock.json`에 기록하고, 결과 파일이 생긴 뒤에는 다시 실행할 수 없다.

테스트는 [`ml/tests/test_cache_experiment.py`](../../tests/test_cache_experiment.py)에 있으며 합성 데이터만 사용한다.
