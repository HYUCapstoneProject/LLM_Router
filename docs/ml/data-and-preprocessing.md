# 데이터와 전처리

기준일: 2026-09-20. 이 문서는 최종 실험의 데이터 계약이다. 실제 실행 상태와 artifact 위치는 [실행·산출물](execution-and-artifacts.md)이 관리한다. 문서의 규칙과 현재 코드가 다른 부분은 마지막의 구현 보완 목록을 따른다. 이미 생성된 artifact를 묵시적으로 덮어쓰지 않는다.

## 1. 데이터의 역할

| 데이터 | 역할 | 허용되는 사용 |
| --- | --- | --- |
| RouterBench text-only | 공개 질문×모델 결과, 난이도 학습·비교 | train으로 학습, validation으로 선택, 고정 test로 최종 평가 |
| xRouteBench `llmrouter_generic` | 별도 모델 pool의 공개 결과, 통합 학습·3-action 비교 | 공식 train을 train/validation으로 분할, 공식 test는 최종 평가 |
| RouterArena | 독립적인 인지 수준 기반 외부 난이도 진단 | 중복 검사에 질문 text 사용; 모델·설정 동결 후 label 평가 |
| 순차 feedback 표본 | 재학습·망각 실험 | 기존 개발 데이터에서 별도 봉인; 초기 모델도 해당 질문을 보지 않음 |
| 캐시용 반복·의미 문항 | 응답 재사용 검증 | 학습·튜닝 표본과 분리; [캐시 계약](semantic-cache.md) 적용 |

새 Claude 응답이나 judge 결과를 대량 생성하지 않는다. 원천에 이미 포함된 과거 Claude 결과는 공개 benchmark 기록으로 재사용한다. 후보 모델의 답변·점수·비용은 target 또는 evaluator에만 제공하고, 질문 라우터의 입력에 넣지 않는다.

공식 원천: [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), [xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench), [RouterArena](https://github.com/RouteWorks/RouterArena). 다운로드 revision, 원본 파일 크기와 SHA-256은 dataset manifest에 고정한다. 최신 branch를 실행 시마다 다시 받는 방식은 사용하지 않는다.

## 2. 포함 범위와 원본 고정

RouterBench는 기존 canonicalization을 재사용한다.

- MMLU, HellaSwag, GSM8K, ARC-Challenge, WinoGrande는 5-shot source를 우선한다.
- MBPP와 MT-Bench는 0-shot source를 우선한다. RAG는 제외한다.
- 동일 `sample_id`의 0-shot/5-shot 행은 source 우선순위로 한 행만 남긴다.
- 직렬화된 prompt list는 안전한 literal parser로 복원한다. 5-shot은 마지막 미응답 질문을 추출한다.
- 원문 `prompt`, 추출한 `query_text`, 학습용 `model_input_text`를 별도로 보존한다.

xRouteBench는 다음 revision의 generic train/test와 후보 metadata만 사용한다.

```text
revision = ea4b6e1b29d9a734f55f0a637baf326bad6aa681
llmrouter_generic/train.parquet
llmrouter_generic/test.parquet
llm_candidates/train.parquet
```

파일별 checksum은 [통합 데이터 config](../../ml/config/router_model_comparison_v1.json)에 있다. 현재 확인한 pivot 전 generic query 수는 train 4,489, test 3,729다. 이는 정제·중복 제거 후 최종 학습 행 수가 아니다.

다음은 주 실험에서 제외하고 사유별 행 수를 기록한다.

- RouterBench RAG/`needs_retrieval=true`; xRouteBench memory, timeseries, video, multimodal, personalized 구성.
- 빈 질문, 파싱 불가, 범위 밖 performance, 중복 query/model 실행처럼 의미를 확정할 수 없는 행.
- 정답·생성 응답·라벨·모델명·가격이 benchmark wrapper를 통해 입력에 노출된 행.
- OpenHermes 등 질문별 동일 후보의 채점 결과가 없어 이 계약의 라벨·비용을 만들 수 없는 데이터.

문제 본문에 원래 등장하는 모델명·숫자·전문용어를 무조건 지우지는 않는다. benchmark 부가 정보와 과제 수행에 필요한 내용을 구분한다.

## 3. 공통 schema와 namespace

| 필드 | 의미 |
| --- | --- |
| `sample_id` | source namespace를 포함한 안정적인 질문 ID |
| `source_dataset`, `source_split`, `source_task` | 원천·공식 분할·원래 문항 식별 정보 |
| `category` | source 안에서의 분석용 영역 |
| `query_text`, `model_input_text` | 추출 원문과 모델에 전달할 정제문 |
| `valid_model_count`, `accepted_model_count`, `overall_solve_rate` | 유효 결과 수, 성공 수, 성공 비율 |
| `difficulty_label`, `difficulty_label_t2` | T1 주 라벨과 T2 민감도 라벨 |
| `label_confidence`, `sample_weight` | 원천 신뢰도와 기존 호환 metadata; 최종 학습 적용 여부는 §7에 고정 |
| `duplicate_group_id`, `split` | 누수 방지 그룹과 최종 분할 |
| `dataset_version`, `label_rule_version`, `preprocessing_version`, `split_version` | 재현 가능한 변환 계약 |
| `label_origin` | benchmark difficulty, 관측된 escalation 성공 action, 인지 주석 등 라벨 근거 |

RouterBench ID에는 `routerbench:`를 붙인다. xRouteBench는 `task_name + task_id(결측이면 embedding_id) + normalized query`를 query key로 만들고 hash에 `xroutebench:`를 붙인다. 키와 원래 task ID를 함께 보존한다. 단순한 정수 ID만 source 사이에서 결합하지 않는다.

영역은 RouterBench의 coarse `domain`을 `category`로, `eval_name`을 `source_task`로 둔다. xRouteBench는 `task_name`을 category로 보존한다. `knowledge`와 개별 xRoute task를 이름만으로 같은 영역으로 합치지 않는다. 공통 taxonomy가 필요하면 별도 version의 mapping table을 만들며 주 결과는 항상 `source × category`로도 제시한다.

학습 입력은 `model_input_text`뿐이다. category, source ID, 정답, 모든 후보의 결과, 실제 응답 길이는 입력 feature가 아니다. 메타데이터가 Parquet에 있어도 collator가 이를 text에 연결하지 않는지 검사한다.

원본 outcome matrix는 감사용 별도 artifact로 유지한다. 분류 학습 projection과 R1/R2 target projection은 공통 `sample_id`로 결합하며, 모델 응답을 text feature로 복사하지 않는다. 새 manifest에는 원천 checksum, 필터 이유, 성공 판정 규칙, 입력 정제 전후 hash를 남긴다.

## 4. 텍스트 정제와 중복 검사

`query_text`를 그대로 남기고 `model_input_text`를 결정적으로 만든다.

1. Unicode NFKC, 줄바꿈 통일, 불필요한 제어 문자·행 끝 공백·중복 빈 줄을 정리한다.
2. 객관식 출력 형식 지시처럼 benchmark에만 필요한 알려진 wrapper를 제거한다.
3. 질문 본문·선택지·코드·필요한 대화 문맥과 context는 유지한다.
4. xRoute의 `[User Query]` wrapper는 실제 원본 구조를 확인해 제거한다. system 부분이 과제 조건을 담고 있으면 삭제하지 않는다.
5. 정제 후 빈 입력, 선택지 누락, 코드 구조 손상은 전처리 오류로 처리한다.

모델 입력에서 case나 구두점을 제거하지 않는다. 코드 들여쓰기와 수식·부정·숫자도 의미의 일부다. 정제 예시를 source/category별로 샘플링해 원문과 대조한다.

중복 탐지는 입력 변환과 별도 경로다. 현재 구현은 NFKC+case-fold+단어 추출 hash로 exact 후보를 만들고, token 3-gram SimHash 후보에 Jaccard `>=0.90`을 적용한다. 최소 5 token, bucket 상한 200을 사용하며 큰 bucket을 건너뛴 수를 기록한다. 이것은 모든 의미 중복을 찾았다는 보장이 아니다.

단어만 남기는 hash는 코드 연산자 등 의미 차이를 놓칠 수 있다. 이런 충돌을 곧바로 동등한 문항으로 선언하지 않고, 원문 보존·오탐 감사와 구조 보존 검증을 거쳐 그룹을 확정한다. 같은 task ID도 namespace를 확인한 뒤 그룹 후보로 사용한다.

확정된 duplicate group은 split을 넘지 않는다. cross-source 그룹은 RouterArena 보호 집합, xRoute 공식 test, RouterBench 고정 test, validation, train 순으로 보호한다. 외부 평가와 겹치는 개발 행을 먼저 제외한다. 평가셋끼리의 overlap은 별도 집계하고 source별 점수를 제시한다.

같은 그룹에서 대표를 남기는 규칙은 정보 완전성과 안정적인 ID tie-break로 동결한다. 어떤 행을 제외했는지 모두 manifest에 보존한다. `exact`, `near`, `task-id`를 구분하며 그룹 대표만 남겼다고 raw 데이터도 삭제하지 않는다.

## 5. 점수에서 성공 여부로 변환

성공 판정 경계와 난이도 경계는 다르다. 먼저 질문별 각 후보 결과를 성공/실패/결측으로 변환한다.

| 원천·metric | 성공 판정 |
| --- | --- |
| RouterBench exact-match 계열 | `performance == 1.0` |
| RouterBench GSM8K partial-credit, MBPP, MT-Bench의 현재 원점수 계약 | `performance >=0.70` |
| xRoute `em_mc`, `GSM8K`, `MATH`, `code_eval` | `performance ==1.0` |
| xRoute `f1_score` | `performance >=0.70` |
| 등록하지 않은 metric 또는 null | 결측; 자동 실패나 임의의 0.70 판정 금지 |

같은 GSM8K 이름이라도 원천의 채점 형식이 다르므로 metric 계약을 먼저 확인한다. 연속 원점수는 버리지 않고 `quality`로도 보존한다. `0`은 관측된 실패이며 결측이 아니다. RouterBench v2에서 제외했던 `unresolved`를 통합 전체 성공률 방식에서도 자동 제외하지 않는다. 유효한 전부 실패 결과는 통합 난이도에서 hard가 될 수 있다.

xRoute는 `(query key, model_name)` 중복 실행을 오류로 처리한다. 18개 후보 중 유효 결과가 12개 이상이어야 통합 라벨 대상이며, 미만은 사유와 함께 제외한다. 범위 밖 `[0,1]` 점수는 조용히 clipping하지 않는다.

```text
overall_solve_rate(q) = accepted_model_count(q) / valid_model_count(q)
```

RouterBench도 실제 유효 모델 수를 분모로 검증한다. upstream에서 만들어진 `overall_solve_rate`를 읽을 때는 재계산 결과와 일치해야 한다. 분모가 0이면 라벨을 만들 수 없다.

## 6. 통합 난이도 T1/T2

난이도는 두 source 모두 관측된 후보 집단의 solve rate로 정의한다.

```text
rate >= t_easy                   → easy
t_hard <= rate < t_easy          → medium
rate < t_hard                    → hard
```

후보 grid는 `t_hard ∈ {0.18,0.27,0.36}`, `t_easy ∈ {0.55,0.64,0.73}`의 9개다. split을 먼저 봉인하고 train만 이용한 5-fold group OOF TF-IDF+LogisticRegression probe로 고른다. vectorizer도 fold train에만 fit한다.

```text
J_threshold = .25×source_macro_F1 + .15×balanced_accuracy
            + .15×QWK_01 + .15×min(source별 macro_F1)
            + .10×(1−ordinal_MAE/2) + .10×(1−hard→easy 비율)
            + .10×candidate-model bootstrap label agreement
QWK_01 = (QWK+1)/2
```

후보는 train의 각 class가 10% 이상·500행 이상, source별 세 class 존재, easy>medium>hard 평균 solve rate, 후보 모델 bootstrap 1,000회 평균 label agreement `>=.85`를 통과해야 한다.

T1은 최고 점수 후보다. 차이 `<.005`의 근접 후보에서는 core-category red 수, 기존 anchor `(0.27,0.64)` 여부, 점수 순으로 고른다. T2는 남은 후보 중 T1과 train 라벨이 5% 이상 다른 최고 점수 후보다. 없으면 남은 최고 후보를 사용하고 실제 불일치율을 명시한다. exact tie에는 변경 경계 수가 작은 후보와 사전 고정한 후보 ID 순서를 사용한다. 최종 tie-break를 구현·manifest에도 동일하게 남긴다.

2026-09-20 확인한 기존 통합 artifact는 T1 `(0.27,0.55)`, T2 `(0.36,0.55)`이고 train 라벨 불일치율은 약 5.83%다. 이 값은 해당 artifact의 결과이지 모든 새 dataset version의 정답이 아니다. source·정제·split이 바뀌면 train-only 선택을 다시 수행한다. test나 RouterArena에 맞춰 threshold를 수정하지 않는다.

T1은 주 실험, T2는 TF-IDF와 ModernBERT seed 42의 민감도 진단이다. 전체 모델×T2 sweep은 기본 범위에 없다. T1/T2를 모델별로 다시 고르지 않으며 T2 결과로 주 결과의 winner를 교체하지 않는다.

이 라벨은 보편적인 인간 인지 난이도가 아니라 **해당 후보 집단의 관측 성공률에 기반한 조작적 정의**다. GPT 블라인드 인지 주석과 통제 문항쌍은 [평가](evaluation.md)의 보조 분석으로 분리한다. 그 주석으로 학습 라벨을 사후 수정하지 않는다.

한 source만 사용하는 공식 3-action 비교는 그 source의 train에서 같은 grid·OOF·gate로 T1을 별도 선정한다. 단일 source에서는 source macro/minimum 항을 해당 source의 값으로 계산한다. 통합 T1을 가져와 다른 source의 학습 정보를 사용하는 대신 독립 manifest를 만들며, 두 T1 값이 다를 수 있다. 현 probe의 두 source 전용 분기를 일반화하는 구현 보완이 필요하다.

## 7. Split, 자연분포와 coverage

| 원천 | train/validation | 최종 test |
| --- | --- | --- |
| RouterBench | 기존 duplicate-group 80/10/10 중 train 80%, validation 10% 보존 | 기존 test 10% |
| xRoute generic | 공식 train을 group-aware 약 90/10; seed 42 | 공식 test 전량에서 오염·중복을 감사 |

xRoute validation은 category와 사전 grid 전체의 solve-rate 경계 strata를 사용한다. 특정 T1을 고른 뒤 그 라벨에 유리하게 split을 바꾸지 않는다. 통합 train은 원천 train을 concatenate하고 일반 shuffle한다. source를 50:50으로 복제하거나 class를 1:1:1로 맞추지 않는다.

**최종 주 비교의 공통 가중치 정책은 eligible 행의 자연분포, effective sample weight 1, class weight 없음이다.** 기존 confidence/weight metadata는 감사·운영 호환 목적으로 보존한다. TF-IDF, kNN 투표, neural loss에 서로 다른 숨은 confidence weight를 적용하지 않는다. confidence weighting·balanced sampler는 별도 명명한 ablation에서만 사용한다. 이 규칙은 초기 라우팅 비교에 적용하며 운영 feedback sampler는 [재학습](retraining-pipeline.md)에 별도로 명시한다.

coverage는 source×category×label의 pre/post-split 행 수와 비율로 확인한다.

| 상태 | train 300행 이상인 core category |
| --- | --- |
| green | 모든 class 비율 ≥15%, 최대 ≤70%, 각 cell ≥100 |
| yellow | 모든 class 비율 ≥5%, 최대 ≤85%, 각 cell ≥30 |
| red | 위 yellow 조건 미달 |

300행 미만은 `small_category`로 표시한다. 평가 cell이 모두 ≥100이면 confirmatory, 모두 ≥30이면 exploratory, 하나라도 <30이면 해당 category의 세 class 우열 주장은 `insufficient_coverage`로 둔다. 자연분포를 인위적으로 고쳐 green으로 만들지 않는다.

split 때문에만 red가 생겼다면 고정 test를 건드리지 않고 개발 partition을 새 version으로 재생성한 뒤 OOF부터 다시 한다. 원래부터 red인 category는 `keep-natural` 또는 별도 데이터 보강 등의 resolution을 남긴다. 같은 label의 red가 두 source에 걸쳐 core category의 20% 이상이면 metric·결측·histogram을 train에서 재검토한다. resolution이 없으면 full sweep을 시작하지 않는다.

현재 기존 통합 manifest의 행 수는 train 31,417 / validation 3,907 / RouterBench test 3,472 / xRoute test 3,729다. 이후 정제 보완·RouterArena overlap 감사·feedback 분할의 행 수로 복사하지 않는다.

## 8. 난이도 label과 실제 model action

RouterBench와 xRouteBench의 후보는 각각 11개, 18개로 서로 다른 실제 모델이다. 같은 easy/medium/hard 라벨이 같은 모델 집합을 뜻하지 않는다. 세 후보를 쓰는 [라우팅 실험](routing-experiments.md)은 source 안에서 train-only 비용–품질 Pareto frontier를 만들고 저비용·knee·최고 품질의 실제 모델 세 개를 봉인한다.

outcome projection에는 각 action의 `model_id`, `quality`, `success`, `actual_cost`, `decision_cost`를 보존한다. tier 평균을 실제 1회 LLM 호출 결과로 사용하지 않는다. 비용은 benchmark 원천의 당시 모델 가격·관측 token 사용량으로 계산하며 현재 Claude 가격으로 바꾸지 않는다.

- `decision_cost`: source train 평균으로 정한 고정 비용; 추론 시 사용 가능.
- `actual_cost`: 해당 응답에서 관측된 실제 비용; evaluator만 사용.
- 선택된 세 후보 중 결과가 빠진 행: 현재 계약은 dataset 생성 실패. 평균 대입·다른 모델 대체 금지.

통합 symbolic-tier 실험과 한 source의 실제 세 후보 실험을 다른 experiment ID로 관리한다. Claude service mapping은 외부 검증 대상이며 공개 결과와 능력이 같다는 전제가 아니다.

## 9. 운영 호환과 구현 보완

기존 운영 v2는 low 성공률≥.67→easy, 아니면 medium≥.67→medium, 아니면 high≥1/3→hard이고 unresolved를 학습에서 제외한다. v3 standalone은 overall solve rate의 `(hard=.27,easy=.64)`다. 이들은 기존 checkpoint와 feedback base artifact를 재현하기 위한 계약이며 최종 통합 T1과 섞지 않는다. 운영 모델의 학습 데이터 변경은 별도 version과 승격 절차를 거친다.

현재 소스: [통합 전처리](../../ml/src/llm_router_labeling/routing_data/preprocessing.py), [기존 RouterBench 정제](../../ml/src/llm_router_labeling/training/routerbench_preprocessing.py), [중복 정규화](../../ml/src/llm_router_labeling/shared/text_normalization.py), [outcome projection](../../ml/src/llm_router_labeling/objective_comparison/data.py).

데이터 정제·결측 분모·누수 검사·가중치·source별 runner의 구현 차이는 [실행 차이 목록](execution-and-artifacts.md#7-현재-구현과-최종-명세의-차이)에서 관리한다.

문서 통합은 위 코드 수정이나 데이터 재생성을 수행한 기록이 아니다. 실행 gate와 보존 정책은 [실행·산출물](execution-and-artifacts.md), 외부 평가·인지 주석은 [평가](evaluation.md), feedback의 별도 분할은 [재학습](retraining-pipeline.md)을 따른다.
