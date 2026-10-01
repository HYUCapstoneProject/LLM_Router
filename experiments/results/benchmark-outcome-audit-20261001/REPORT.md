# 벤치마크 채점 오류 점검과 검증 데이터셋 v2 — 2026-10-01

F0~F3 결과를 검토하다가 "GPT-4는 틀리고 Mistral은 맞힌" 역전 사례가 365건(8.2%) 나왔다. 응답 원문을 열어 보니 GPT-4가 맞게 답한 문제가 오답으로 기록되어 있었다. 그래서 학습·평가에 쓴 두 벤치마크의 기록 점수를 **공개 정답과 모델 응답으로 전부 다시 채점**했다. 설계 기준과 정정 규칙의 요약은 [ML 명세 2.5.1절](../../../docs/ml/README.md#outcome-audit)에 있다.

## 1. 결론

1. **두 벤치마크 모두 기록 점수에 체계적인 오류가 있었다.** RouterBench는 82,519칸(GSM8K 재채점 81,950칸 중 성공 여부 반전 14,904칸 포함), xRouteBench는 13,733칸을 정정했다. MMLU 정답 키 오류 등으로 RouterBench 290문항을 제외했다.
2. **오류는 비싼 모델에 불리한 방향으로 쏠려 있었다.** RouterBench GSM8K에서 GPT-4의 성공률은 기록상 68.1%였지만 재채점하면 95.3%다(Mistral-7B는 28.0%→38.2%). 싼 모델이 비싼 모델보다 잘 푼다는 신호가 실제보다 크게 들어가 있었다.
3. **v1 기반의 결론은 이 오류의 영향을 받는다.** R0 1.0의 학습 라벨, R0~R3 비교, D_gate 품질 기준, F0~F3의 최종 해결률 비교(F0 1위)가 모두 영향을 받는다. 특히 F0 우위의 주된 원인이었던 "high 모델만 틀리는 문제"의 상당수가 채점 오류였다.
4. 정정 규칙·공개 정답·정정표를 고정한 **검증 데이터셋 v2**를 만들었다. 현재 기준은 v1을 유지하고, v2는 config 교체로 언제든 실험할 수 있게 보관한다. 기존 실험의 v2 재실행 여부는 보류한다.

## 2. 점검 방법

| 원천 | 대조 기준 | 방법 |
|---|---|---|
| RouterBench MMLU·HellaSwag·ARC-Challenge·Winogrande | HF `cais/mmlu`, `Rowan/hellaswag`, `allenai/ai2_arc`, `allenai/winogrande` | 질문과 **보기 집합**으로 매칭(RouterBench는 보기 순서를 섞어 출제), 응답 글자를 정답 글자와 비교 |
| RouterBench MMLU 정답 키 | `edinburgh-dawg/mmlu-redux-2.0` (5,700문항 사람 검수) | 정답 오류는 정정 답으로 재채점, 문항 자체 결함은 제외 |
| RouterBench GSM8K | HF `openai/gsm8k` | 질문 원문으로 매칭(7,450/7,450), 응답의 최종 답 숫자와 비교 |
| xRouteBench 전 과제 | 데이터에 포함된 `ground_truth` | 응답을 직접 재채점해 기록 점수와 비교 |

RouterBench 학습 라벨은 행별로 5-shot(34,271행) 또는 0-shot(507행) 원본에서 왔다. 응답 대조도 같은 원본을 썼다. 모든 공개 정답 파일의 URL과 SHA-256은 [v2 config](../../../ml/config/modernbert_initial_v2_verified_data.json)에 고정했다.

## 3. 발견한 오류

### 3.1. RouterBench GSM8K: 점수가 정오가 아니다

점수는 0.25 단위의 평가값이고, 기존 파이프라인은 `>= 0.70`을 성공으로 봤다. 재채점 결과와 비교하면 다음과 같다.

| 기록 점수 | 재채점 실패 | 재채점 정답 | 기존 판정 |
|---:|---:|---:|---|
| 0.00 | 184 | 147 | 실패 |
| 0.25 | 23,298 | 873 | 실패 |
| **0.50** | 428 | **12,544** | **실패** ← 대부분 정답 |
| 0.75 | 1,080 | 43,110 | 성공 |
| **1.00** | **260** | 26 | **성공** ← 대부분 오답 |

예를 들어 "Elsa의 남은 데이터"(정답 120)는 Mistral·Yi·GPT-4가 모두 120이라고 답했는데 0.75/0.5/0.5로 기록됐다. 점수가 정오와 단조 관계도 아니다.

| 모델 | 기록 기준 성공률 | 재채점 정답률 |
|---|---:|---:|
| GPT-4-1106-preview | 68.1% | **95.3%** |
| Claude-v2 | 75.0% | 89.9% |
| GPT-3.5-turbo-1106 | 59.7% | 81.2% |
| Yi-34B-Chat | 51.3% | 67.9% |
| Mistral-7B-Chat | 28.0% | 38.2% |

재채점 정답률은 공개된 GSM8K 성능과 일치한다. 최종 답 추출은 박스·`####` 답, 그 외에는 마지막 줄의 첫 숫자와 끝 숫자만 쓴다. 정답 판정 표본 25건을 사람이 확인했고 오탐은 없었다.

### 3.2. MMLU 원본 정답 키 오류

RouterBench 객관식 채점은 공개 정답과 일치했다(5-shot 기준 불일치는 claude-instant 8칸). 문제는 MMLU 정답 키 자체다. 예: `Solve −47 = g + 24`의 공식 정답이 −23(실제 −71)이다. 그래서 정답 C를 고른 GPT-4는 0점, 오답 B를 고른 Yi는 1점이 됐다.

MMLU-Redux 2.0과 대조한 결과:

| Redux 판정 | 문항 | 처리 |
|---|---:|---|
| 정답 오류(정정 답 있음) | 99 | 정정 답으로 재채점, 781칸 변경 |
| 정답 오류(정정 답 없음) | 5 | 제외 |
| 질문 불명확 | 132 | 제외 |
| 복수 정답 | 39 | 제외 |
| 정답 없음 | 36 | 제외 |
| 전문가 판단 필요 | 32 | 제외 |
| 보기 불명확 | 25 | 제외 |

그 밖에 공개 정답과 매칭되지 않거나 정답이 보기에 없는 21문항도 제외했다. RouterBench 제외는 모두 290문항이다.

### 3.3. xRouteBench: 형식 때문에 정답이 0점

| 과제 | 오류 | 정정 칸 |
|---|---|---:|
| `em_mc` 7개 과제 | `(C)`만 정답 처리하고 `C`, `C. 보기문`은 0점. 정답 글자 단독 응답 1,499건이 전부 0점이었고, 형식 `(X)`인 응답은 정오 판정이 100% 일치했다 | 5,546 |
| GSM8K | `$720.00`, `\boxed{12}`, `24{,}960` 등 정답 형식을 실패 처리 | 1,804 |
| MATH | `\text{odd}`↔`odd`, `(-4,\,1)`↔`(-4,1)`, `5,\!866,\!608`↔`5866608` | 124 |
| SQuAD | 문장형 정답의 token F1이 0.70 미만. 응답의 80.6%가 정답을 포함하는데 성공은 2.4% | 6,259 |

과제별 `em_mc` 정정: MMLU 2,269 · BoolQ 1,195 · HellaSwag 961 · MMLU-Pro 385 · ARC 349 · OpenBookQA 295 · CommonsenseQA 92.

xRouteBench에서는 **기록된 성공을 취소하지 않는다.** 반대 방향 불일치에는 원 채점기의 판단이 타당한 경우(MATH 단위 생략 `8 cm`→`8`)와 제 추출기의 한계가 섞여 있어, 자동으로 실패로 바꾸면 새 오류를 만들 수 있다. 다만 `(B) (A)`처럼 답이 두 개인 응답이 정답 처리된 사례가 소수 남는다.

### 3.4. 검증하지 못한 범위

| 항목 | 상태 |
|---|---|
| RouterBench MBPP, xRouteBench HumanEval·MBPP | 같은 응답에 다른 점수가 붙은 사례 0건. 코드를 실행해 재채점하지는 않았다 |
| RouterBench MT-Bench | judge 평점이라 정오 개념이 없어 그대로 둔다(158행) |
| Redux 밖 MMLU 8,342문항 | 정답 키 오류가 남아 있을 수 있다(Redux 표본의 정답 오류율은 약 1.9%) |
| HellaSwag | 알려진 라벨 잡음이 있으나 검수 데이터가 없다 |
| RouterBench 5-shot 원본 중복 `sample_id` 14개 | 같은 ID에 서로 다른 내용 182칸. 기존 라벨 파일이 고른 행을 그대로 쓴다 |
| RouterArena | 모델 응답·채점 기록 없이 공식 난이도·정답만 사용하므로 대상 아님 |

## 4. 검증 데이터셋 v2

같은 seed·분할 규칙·임계값 후보로 v1과 똑같이 빌드하고, 기록 점수 대신 정정표를 적용했다. 정정표 manifest SHA는 `c5a2a47e…665b1`이다.

| 항목 | v1 | v2 |
|---|---:|---:|
| 전체 행 | 42,935 | 42,646 (−289, 제외 문항) |
| D_initial_train / D_dev / D_gate | 21,406 / 3,581 / 3,581 | 21,245 / 3,557 / 3,557 |
| D_stream / D_retention_test | 5,368 / 1,797 | 5,328 / 1,785 |
| RouterBench 고정 test / xRouteBench 공식 test | 3,473 / 3,729 | 3,445 / 3,729 |
| 선택된 T1 (hard / easy 경계) | 0.27 / 0.64 | **0.36 / 0.73** |
| easy / medium / hard | 50.1% / 36.8% / 13.1% | 45.5% / 39.2% / 15.4% |

- **난이도 라벨이 19.6% 바뀌었다.** 전체 성공률이 올라 train-only 임계값 선택이 더 높은 경계(T1 0.36/0.73)를 골랐기 때문에, 점수가 바뀌지 않은 과제(Winogrande 22.7%, HellaSwag 17.6%)의 라벨도 함께 움직였다.
- 정정이 직접 닿은 과제의 평균 성공률: RouterBench GSM8K 0.543→0.692, xRouteBench SQuAD 0.024→0.798, GSM8K 0.773→0.878, BoolQ 0.656→0.803, MMLU 0.613→0.746.
- 연속 학습 분할은 라벨 층화 후 같은 seed로 다시 나눴으므로 문항의 59.0%만 v1과 같은 분할에 있다. **v1과 v2의 D_gate·D_stream 수치는 같은 문항 집합의 비교가 아니다.**
- 운영 R0 1.0은 v1 라벨로 학습했다. v2로 R0를 다시 학습하고 1.0 승격 하한을 다시 산정할지는 보류한다. R1 계열의 후보별 성공 행렬(objective data)도 같은 정정표를 읽도록 연결했으나, 아직 v2로 다시 만들지 않았다.

## 5. 산출물과 재현

| 파일 | 내용 |
|---|---|
| [audit-summary.json](audit/audit-summary.json) | 규칙별 정정·제외 건수 |
| [routerbench-corrections.parquet](audit/routerbench-corrections.parquet) | RouterBench 칸 단위 정정(샘플 ID, 모델, 원점수, 정정 점수, 규칙) |
| [routerbench-exclusions.parquet](audit/routerbench-exclusions.parquet) | 제외 문항과 사유 |
| [xroutebench-corrections.parquet](audit/xroutebench-corrections.parquet) | xRouteBench 칸 단위 정정(질문은 해시로만 기록) |
| [manifest.json](audit/manifest.json) | 위 파일의 SHA-256. v2 config의 `audit_manifest_sha256`과 일치해야 빌드된다 |
| [dataset-manifest.json](dataset-manifest.json) | v2 데이터셋 manifest 사본 |

질문 원문이 든 데이터셋 parquet은 [보관 정책](../ARTIFACT_STORAGE.md)에 따라 Git에 넣지 않는다. 재현 명령은 [ML 명세 2.5.1절](../../../docs/ml/README.md#outcome-audit)에 있다.
