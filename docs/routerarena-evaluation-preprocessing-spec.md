# RouterArena Evaluation Preprocessing Specification

## 1. 목적과 사용 금지 범위

RouterBench로 학습한 3-class 라우터가 라벨 출처와 도메인이 다른 질문에도
일반화되는지 RouterArena로 검증한다. 공식 배포 Parquet의 질문별 `Difficulty`는
이미 `easy/medium/hard` 세 값이므로 별도의 6단계→3단계 변환을 수행하지 않는다.
RouterArena README는 Bloom taxonomy의 6개 인지 단계를 설명하지만 공개 Parquet에는
질문별 6단계 컬럼이 없다. 따라서 전처리기에서 이를 추정하거나 재생성하지 않는다.

RouterArena의 공식 3단계 난이도와 RouterBench의 최소 충분 모델 tier 라벨은 생성
기준이 같지 않다.
따라서 이 데이터는 transfer evaluation과 robustness evaluation에만 사용하고
RouterBench 학습 또는 threshold fitting에 섞지 않는다.

## 2. 고정한 공식 배포본

```text
repository: RouteWorks/RouterArena
revision: a4a062ce3313b56bb09c042e1bc37b61d34e3bd8
dataset_version: routeworks-routerarena-a4a062ce3313b56bb09c042e1bc37b61d34e3bd8
```

| split | 행 | 파일 크기 | SHA-256 | 용도 |
| --- | ---: | ---: | --- | --- |
| `full` | 8,400 | 3,674,459 | `358a6bb4160320b5cb5b68d6d0a885c89b128b06d3dea267d77ed6daeebe4369` | 주 일반화 평가 |
| `sub_10` | 809 | 380,482 | `e7b4b7579e98df0b83202324357d0b91af78666ddbf1038a513d19edaf58efd5` | 로컬 smoke 전용 |
| `robustness` | 420 | 233,165 | `6012eeea11e2d7b2018155e65c35ad887e9daa70e45aa38a40321da74d0cc2e9` | 변형 질문 강건성 평가 |

`sub_10`은 `full`의 809행 부분집합이므로 두 split의 결과를 합산하지 않는다.
`robustness`의 420개 `Global Index`도 모두 `full`에 존재한다.

공식 원본 schema:

```text
Category, Domain, Dataset name, Global Index
Context, Question, Options, Answer
Metadata, Keywords, Difficulty
```

## 3. RouterBench와 다른 점

| 항목 | RouterBench | RouterArena |
| --- | --- | --- |
| 질문 | 직렬화된 `prompt` | `Context`, `Question`, `Options` 분리 |
| 난이도 | 모델 tier별 성공률로 재산정 | 배포본의 공식 3단계 `Difficulty` 사용 |
| 정답 | 모델별 performance에 반영 | `Answer` 컬럼으로 별도 제공 |
| split | 우리가 group-aware split 생성 | 공식 `full/sub_10/robustness` 의미 보존 |
| robustness | 없음 | 라벨이 비어 있어 `full`과 ID join 필요 |

## 4. 전처리 규칙

### 4-1. 모델 입력 조립

- HTML entity를 복원하고 Unicode NFKC, 줄바꿈, 제어문자를 정규화한다.
- context가 있으면 `Context:` 블록으로 먼저 둔다.
- question을 뒤에 연결한다.
- options가 있으면 `A)`, `B)` 형식으로 연결한다.
- `Answer`, `Difficulty`, domain/category 정보는 `model_input_text`에 넣지 않는다.
- 질문이 비어 있으면 `missing_question`으로 제외하고 metadata에는 보존한다.
- 전처리 단계에서는 입력을 자르지 않는다. full split의 최대 입력이 54,601자이므로
  evaluator는 학습과 동일한 tokenizer/max length를 사용하고 truncation 행 수와
  label/domain 분포를 별도 metric으로 남겨야 한다.

```text
Context:
<context>

<question>

A) <option 1>
B) <option 2>
```

### 4-2. 라벨과 robustness 결합

- `full`, `sub_10`: `Difficulty`를 소문자 `easy/medium/hard`로 검증해 그대로 사용한다.
- `Remember/Understand/Apply/Analyze/Evaluate/Create`를 임의로 `2/2/2` 묶거나
  `Difficulty`에서 역추정하지 않는다.
- `robustness`: 자체 Difficulty가 모두 비어 있으므로 `Global Index`로 `full`과
  결합해 label, domain, category, source dataset과 원 질문을 상속한다.
- join되지 않는 robustness ID가 하나라도 있으면 전처리를 실패시킨다.
- robustness metadata의 `reference_model_input_text`로 원 질문과 변형 질문의 예측
  일관성을 후속 evaluator에서 비교한다.

### 4-3. 품질 필터

- 정규화된 입력 hash가 같은 source duplicate는 안정적인 sample ID 순서로 한 건만
  평가하고 나머지는 `duplicate_content`로 보존한다.
- 질문 대신 생성된 거절 문구는 `refusal_placeholder`로 제외한다.
- 변형 질문 끝에 정답이 `\\boxed{{...}}`로 첨부된 행은 `embedded_answer`로 제외한다.
- RouterBench 학습 artifact가 전달되면 정규화된 입력 hash를 비교한다. exact overlap은
  `exact_training_overlap`으로 제외한다.
- production 평가와 배포 gate에서는 overlap 검사 상태가 `not_run`이면 실패시킨다.
  로컬 smoke에서는 선택적으로 생략할 수 있다.

## 5. 출력 계약

모델 evaluator가 읽는 projection에는 정답과 분석 metadata를 넣지 않는다.

```text
evaluation.parquet
  sample_id
  model_input_text
  difficulty_label
  evaluation_split
  source_sample_id

metadata.parquet
  context, question, options, answer
  category, domain, source_dataset
  reference_model_input_text
  content_hash, training_overlap, exclusion_reason
  dataset_version, preprocessing_version

metrics.json
  row/exclusion/label/domain 분포
  source dataset 수, duplicate/overlap 수
  입력 길이와 overlap 검사 상태
```

## 6. 실행

```bash
.venv/bin/routerarena-preprocessing download \
  --data-dir ml/data/raw/routerarena \
  --split all

.venv/bin/routerarena-preprocessing preprocess \
  --data-dir ml/data/raw/routerarena \
  --output-dir ml/artifacts/routerarena-evaluation-smoke-100 \
  --split sub_10 \
  --sample-size 100 \
  --seed 42 \
  --training-path ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet
```

주 평가에서는 `--split full`과 전체 RouterBench training artifact를 사용한다.
강건성 평가에서는 `--split robustness`를 사용하며 전처리기가 full split을 함께 읽는다.

## 7. Kubeflow 책임 경계

전처리 코드가 schema 검증, 입력 조립, label join, 품질 필터와 누수 검사를 담당한다.
Kubeflow Pipeline은 다음만 담당한다.

- RouterBench training artifact와 RouterArena source artifact 전달
- checksum/config 기반 cache와 ML Metadata 계보
- `sub_10` smoke, `full` generalization, `robustness` 평가 component 분리
- prediction/evaluation metric artifact 전달과 배포 gate 조건
- 동일 모델 버전의 full/robustness 평가 병렬 실행

RouterArena 전처리는 RouterBench 학습 전처리 component와 별도 component로 유지한다.

## 8. 구현 체크리스트

- [x] 공식 revision, 파일 크기, checksum, row count를 고정한다.
- [x] 공식 질문별 라벨이 `easy/medium/hard` 3단계임을 확인하고 identity mapping으로
  사용한다.
- [x] Context/Question/Options 입력 조립기를 구현한다.
- [x] 공식 `easy/medium/hard` difficulty 라벨 검증을 구현한다.
- [x] robustness-full ID join과 원 질문 보존을 구현한다.
- [x] missing/refusal/embedded-answer/source-duplicate 필터를 구현한다.
- [x] RouterBench exact training overlap 검사를 구현한다.
- [x] evaluation/metadata projection을 분리한다.
- [x] 100행 generalization/robustness smoke를 실행한다.
- [ ] 전체 RouterBench 학습 artifact로 overlap 검사를 다시 실행한다.
- [x] label source를 `routerarena_official_difficulty` 계열로 명명한다.
- [ ] ModernBERT checkpoint를 연결해 transfer accuracy/macro-F1을 계산한다.
- [ ] robustness 원본-변형 prediction consistency를 계산한다.

## 9. 실제 전처리 결과

### Full generalization

RouterBench 100행 training smoke artifact를 overlap 비교 대상으로 사용했다.

| 항목 | 결과 |
| --- | ---: |
| 원본 | 8,400 |
| 평가 포함 | 8,397 |
| 빈 질문 제외 | 1 |
| source duplicate 제외 | 2 |
| easy / medium / hard | 3,990 / 2,442 / 1,965 |
| domain | 9 |
| source dataset | 79 |
| exact training overlap | 0 |
| 빈 model input | 0 |

### Sub-10 smoke 100

| 항목 | 결과 |
| --- | ---: |
| 입력 후보 | 809 |
| 고정 seed 표본 | 100 |
| easy / medium / hard | 42 / 31 / 27 |
| domain | 9 |
| source dataset | 45 |
| exact training overlap | 0 |

### Robustness smoke 100

| 항목 | 결과 |
| --- | ---: |
| 원본 변형 질문 | 420 |
| 품질 필터 통과 후보 | 415 |
| 거절 placeholder 제외 | 3 |
| embedded answer 제외 | 2 |
| 고정 seed 표본 | 100 |
| easy / medium / hard | 46 / 27 / 27 |
| exact training overlap | 0 |

현재 overlap 0건은 RouterBench **100행 smoke training artifact**와 비교한 결과이므로
최종 누수 없음의 근거로 사용할 수 없다. 전체 RouterBench training artifact가 생성되면
같은 설정으로 full/robustness 검사를 다시 실행해야 한다.
