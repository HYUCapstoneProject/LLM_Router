# 통합 라우팅 데이터와 최종 모델군 비교·평가 보고서

기준일: 2026-09-16
최종 통합 갱신: 2026-09-16
상태: **실행 기준 문서** — 비교 대상·데이터·threshold 탐색 공간·선정 절차와
운영 연결 경계를 동결한다. 실험 결과로 승자를 확정하기 전까지 현재 production
ModernBERT를 자동 교체하지 않는다.

관련 문서:

- `llm-routing-benchmark-dataset-analysis.md`: 벤치마크 원점수와 공식 라벨 생성 근거
- `routerbench-data-labeling-spec.md`: 이미 구현된 RouterBench v2·v3 계약
- `routerarena-routing-evaluation-spec.md`: RouterArena 독립 외부평가 계약
- `modernbert-training-spec.md`: 현재 ModernBERT 학습·checkpoint 실행 계약
- `training-reports/modernbert-v2-balanced-training-report.md`: 시간·성능 실측 기준선
- `gcp-rollout-and-serving-integration-plan.md`: 현행 Vertex/GCS 구현과 cloud smoke 세부사항
- `../../../LLM_Router-serving/docs/api-contract.md`: Spring–FastAPI HTTP 단일 계약

이 문서는 RouterBench와 xRouteBench를 함께 쓰는 다음 데이터 버전, threshold 후보군과
선정 목적함수, Colab 모델 비교, ModernBERT 전용 GCP 파이프라인, FastAPI 반영과 API 경계를
한 곳에서 통합한다. 기존 v2/v3 artifact와 ModernBERT run은 재현용으로 보존하되, **새 통합 실험에서
충돌하는 내용은 이 문서를 우선한다.** 진행 체크박스는 만들지 않으며 실제 상태는
`workstream-checklist.md`에서만 관리한다.

## 0. 최종 결정 요약

### 0-0. 전체 실험·운영 flow

아래 순서를 바꾸지 않는다. 특히 threshold와 모델을 고른 뒤 RouterArena를 보고 되돌아가면
외부평가가 다시 validation으로 변하므로, 되돌아갈 때는 새 실험 버전과 새 holdout이 필요하다.

```text
[A. 계약 봉인]
원천 revision·checksum 고정
  -> RAG/비텍스트 제외
  -> 질문 정규화·cross-source duplicate group 생성
  -> train/validation/test/external split 봉인

[B. 라벨 threshold 선택: train만 사용]
candidate-model score acceptance 적용
  -> query별 overall_solve_rate 계산
  -> 9개 (hard, easy) threshold 후보 생성
  -> group OOF TF-IDF probe + label 안정성 목적함수
  -> 후보별 train category×label coverage 계산
  -> 주 기준 T1과 민감도 분석용 T2 동결
  -> 선택된 T1/T2의 pre/post-split coverage audit와 resolution 기록
  -> split-induced red면 새 split version으로 돌아가 threshold 선택부터 재실행
  -> 통합 dataset version과 manifest 게시

[C. Colab 모델 비교: validation까지만 선택에 사용]
공통 dataset artifact를 Google Drive에 봉인
  -> 모델별 Colab notebook에서 TF-IDF 기준선과 5개 neural model 실행
  -> 5개 neural model smoke/pilot
  -> T1·seed 42 full sweep
  -> T2는 TF-IDF와 ModernBERT seed 42 민감도 분석
  -> validation 상위 2개만 seed 17/73
  -> provisional winner 동결

[D. 손대지 않은 평가와 학술적 결론]
RouterBench test + xRouteBench official test
  -> RouterArena aligned/unseen/full label-alignment
  -> paired bootstrap·오류/길이/도메인/효율 분석
  -> 동급 encoder 결론 + 유형 결론 + production Pareto 권고

[E. ModernBERT 운영 후보 승격]
Colab에서 통합 데이터로 ModernBERT 초기학습
  -> Drive checkpoint checksum·재개·최종 _SUCCESS 검증
  -> 검증된 checkpoint를 package-v1로 export
  -> CPU deployment-compatibility/latency smoke
  -> deploy gate
  -> candidate/approved 게시 + CURRENT.json 원자 갱신
  -> FastAPI load-before-swap, 실패 시 직전 known-good 유지

[F. 운영 feedback]
Spring이 PII 제거된 feedback-inline-v1 전송
  -> FastAPI가 Parquet/manifest/replay를 GCS에 생성
  -> Vertex ModernBERT 재학습·평가·gate·승격
  -> 3-call cycle, 5 major마다 pristine rebase
```

실험용 **Colab notebook suite**, Colab ModernBERT 운영 초기학습, GCP 재학습 pipeline을
분리한다. 다중 모델 비교와 ModernBERT 최초 checkpoint 학습은 Colab에서 실행한다.
KFP v2 IR/Vertex의 GPU 학습은 **ModernBERT feedback 재학습만** 담당하고, 평가·gate·승격도
ModernBERT artifact에만 적용한다. 매 escalation 때 여섯 모델을 다시 학습하거나 비교
winner를 자동 배포하지 않는다.

### 0-1. 실험 질문

이 실험은 **질문 텍스트만 보고 `easy/medium/hard` 중 하나를 고르는 라우터**를 비교한다.
후보 LLM의 설명·가격·응답은 온라인 입력으로 요구하지 않는다. 학습 시에도 공개
benchmark에 이미 들어 있는 정답 성능만 사용하므로, 별도의 후보 모델 설명 생성이나
Claude 대량 호출이 필요 없다.

### 0-2. 최종 비교군

| 구분 | 방법·checkpoint | 파라미터 | 실험에서 답할 질문 | 최종 용도 |
| --- | --- | ---: | --- | --- |
| 비신경 기준선 | TF-IDF + multinomial LogisticRegression | 해당 없음 | 고가 Transformer가 단어·출처 단서보다 실제로 나은가 | 필수 기준선 |
| 경량 encoder | `microsoft/MiniLM-L12-H384-uncased` | 33M | 품질 손실이 작다면 지연·메모리를 크게 줄일 수 있는가 | 연구 비교·차기 전환 권고 후보 |
| 고전 encoder control | `FacebookAI/roberta-base` | 약 125M | ModernBERT의 이점이 현대 구조 때문인가 | 원인 분석용 필수 control |
| 현행 encoder | `answerdotai/ModernBERT-base` | 149M | 현행 기준선과 긴 문맥·코드 사전학습의 이점이 있는가 | production 기준·후보 |
| 강한 동급 encoder | `microsoft/deberta-v3-base` | 약 184M(86M backbone+98M embedding) | disentangled attention이 약한 난이도 신호를 더 잘 잡는가 | 연구 비교·차기 전환 권고 후보 |
| 소형 생성형 LLM | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 양방향 encoder가 아닌 causal LM이 난이도를 더 잘 설명하는가 | 연구 control·차기 전환 권고 후보 |

비슷한 크기의 **RoBERTa 125M–ModernBERT 149M–DeBERTa-v3 184M** 비교에서 구조 차이를
해석한다. MiniLM은 압축 효과, Qwen은 모델 유형 차이를 확인하는 별도 축이다. 크기가
세 배 이상 다른 Qwen의 결과를 동급 파라미터 비교라고 해석하지 않는다.

### 0-3. 최종 데이터와 평가 단계

| 단계 | 데이터 | 사용 목적 | 모델·threshold 선택에 사용 |
| --- | --- | --- | --- |
| threshold 탐색 | 두 source의 train만 사용한 group OOF | 9개 solve-rate 경계 중 T1·T2 선택 | 예 |
| coverage audit | pre/post-split source×category×label count | split 오류·부족 cell·해석 한계 판정 | OOF 근접 후보 tie-break만 |
| 학습 | RouterBench text-only train + xRouteBench `llmrouter_generic` train | 공통 3단계 라우터 학습 | 예 |
| 내부 검증 | 두 source의 duplicate-group validation | LR·checkpoint·seed finalist 선택 | 예 |
| 내부 최종시험 | RouterBench test + xRouteBench 공식 test | 동분포·교차 source 성능 확인 | 아니오, 1회만 |
| 외부 정렬평가 | RouterArena `RB/service-aligned` | 학습 범위와 비슷한 외부 난이도 라벨 정렬 | 아니오 |
| 외부 OOD 진단 | RouterArena `RB-unseen` | 번역·음악·체스·장문독해 일반화 | 아니오 |
| 전체 참고 | RouterArena full/clean | 전체 분포 참고 및 공식 방식과 구분 | 아니오 |

OpenHermes는 질문은 풍부하지만 query×candidate-model correctness가 없어 동일한 solve-rate
라벨을 만들 수 없다. 새 LLM 호출·judge 전처리를 하지 않는다는 제약에 따라 **이번
지도학습 데이터에는 넣지 않는다**. RouterArena는 계속 evaluation-only이며, 점수를 보고
threshold·학습률·모델을 다시 고르지 않는다.

### 0-4. 실행 단위 요약

- **학습 비율:** class나 source 비율을 인위적으로 맞추지 않고, 정제 후 train 전량을
  합친 자연분포를 사용한다. 예상 source 비율은 약 RouterBench 87% : xRouteBench 13%이며
  실제 값은 manifest가 소유한다.
- **라벨 경계:** 기존 `0.27/0.64`를 기본 anchor로 포함한 9개 후보를 train-only OOF로
  평가해 공식 비교용 T1과 민감도 분석용 T2를 함께 동결한다. `performance >= 0.70`이라는
  연속 점수 acceptance와 혼동하지 않는다.
- **카테고리 coverage:** 자연분포를 1:1:1로 바꾸지 않고 green/yellow/red로 진단한다.
  red는 원인과 resolution을 manifest에 기록하며 평가 표본이 부족한 cell은 우열 주장에 쓰지 않는다.
- **모델 선택:** validation에서만 한다. test와 RouterArena는 선택 후 1회 평가한다.
- **실행 환경:** 여섯 비교군과 ModernBERT 운영 초기학습은 Colab에서 실행한다. GCP에
  게시해 실행하는 KFP/Vertex GPU 학습은 ModernBERT feedback 재학습에만 사용한다.
- **운영 승격:** 이번 단계의 escalation 재학습·자동 승격은 ModernBERT만 허용한다.
  학술적 최고 모델과 production Pareto 권고 모델이 달라도 다른 family를 `CURRENT.json`에
  올리지 않는다.

## 1. 먼저 확정할 문서 책임

| 질문 | 소유 문서 | 이 문서의 역할 |
| --- | --- | --- |
| RouterBench/RouterArena 원점수와 라벨은 무엇인가 | 벤치마크 분석 보고서 | 반복 설명하지 않고 참조 |
| 현재 RouterBench v2/v3는 어떻게 재현하는가 | RouterBench 라벨 명세 | 기존 버전 보존 |
| RouterArena를 어디까지 의사결정에 쓰는가 | RouterArena 평가 명세 | 학습·threshold 선택 금지 유지 |
| 현재 ModernBERT는 어떻게 실행하는가 | ModernBERT 학습 명세 | Colab 비교용 run과 GCP 운영 run의 공통 학습 계약을 참조 |
| RouterBench+xRouteBench를 어떻게 합치고 무엇을 비교하는가 | **이 문서** | 새 실험의 단일 설계 문서 |

따라서 별도 “epoch 문서”, “모델 후보 문서”, “데이터 정제 문서”를 각각 만들지 않는다.
수치가 확정되면 run별 결과만 `training-reports/`에 추가한다.

## 2. 데이터 범위와 사용 금지

### 2-1. 포함 확정

1. **RouterBench 핵심 text-only 범위**
   - 기존 canonicalization과 query extraction을 재사용한다.
   - v2/v3의 원본 performance와 duplicate metadata를 보존한다.
2. **xRouteBench `llmrouter_generic` routing train**
   - 공식 데이터: [ulab-ai/xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench)
   - train은 4,487개 query, 최대 18개 candidate-model 실행 행/query다.
   - `task_name + task_id + normalized query`를 query key로 pivot한 뒤 한 행으로 만든다.

통합 전 상한은 RouterBench 34,778 query + xRouteBench train 4,487 query다. 필터,
split 우선순위와 cross-source 중복 제거 전 숫자이므로 최종 학습 행 수로 인용하지 않는다.

위 두 원천은 이제 단순 후보가 아니라 **통합 학습 원천으로 확정**한다. 단,
xRouteBench를 실제로 내려받아 metric별 acceptance mapping과 12/18 유효 모델 조건을
통과한 행 수는 새 preprocessing manifest가 소유한다. 문서의 39,265개 상한을 실제
학습 건수처럼 사용하지 않는다.

### 2-2. 제외

- RouterBench `RAG`와 `needs_retrieval=true`
- xRouteBench `memory_locomo`, `memory_longmemeval` — 공식적으로 RAG top-k=5 문맥 사용
- xRouteBench timeseries, video, multimodal, personalized config
- 빈 질문, 파싱 실패, performance 범위가 0~1 밖인 행
- 응답·정답·모델명·가격이 질문 text에 섞여 label leakage를 일으킨 행
- xRouteBench 공식 test 3,729 query — 학습·threshold·class-ratio 선택에 사용 금지
- RouterArena 전체 — 계속 최종 transfer 관찰 전용
- OpenHermes — 호환되는 query×model correctness가 없어 이번 라벨을 계산할 수 없음

RAG를 제거하는 이유는 검색 필요성과 언어적 난이도가 섞이기 때문이다. 현행 serving이
질문 text만으로 3단계를 분류하므로 이미지·영상·시계열·개인화 입력도 같은 target으로
합치지 않는다. 나중에 modality-aware router를 만들면 별도 dataset version으로 다룬다.

OpenHermes를 제외하는 이유는 데이터 품질이 낮아서가 아니다. OpenHermes 질문마다
여러 capability의 후보 모델을 실행하고 동일 evaluator로 채점해야 RouterBench/xRouteBench와
같은 solve-rate를 만들 수 있다. 이번 실험은 그 API 비용과 대규모 judge 전처리를 하지
않으므로, 자체 난이도 추정이나 다른 벤치마크의 라벨을 억지로 붙이지 않는다.

### 2-3. RouterArena 평가 slice

RouterArena 행을 원본에서 삭제하지 않고 다음 세 집합으로 분리 보고한다. 분류 기준은
모델 예측이나 Difficulty 라벨을 보기 전에 source 이름과 production 범위만으로 고정한다.

| slice | 현재 clean artifact 기준 | 구성 | 해석 |
| --- | ---: | --- | --- |
| `rb_service_aligned` | 7,347 | 아래 1,028행을 제외한 text reasoning/knowledge/math/code | 주 외부 정렬 지표 |
| `rb_unseen` | 1,028 | WMT19 257, 음악 240, 체스 148, NarrativeQA 383 | OOD 진단 |
| `full_clean` | 8,375 | 기존 exact-training-overlap 22, duplicate 2, missing-question 1 제외 | 전체 참고 지표 |

현재 artifact의 label 분포는 `rb_service_aligned`가 easy 3,818 / medium 2,253 /
hard 1,276이고, `rb_unseen`이 easy 154 / medium 186 / hard 688이다. unseen은 hard가
66.9%이므로 accuracy만 비교하면 hard 편향 모델이 유리해진다. 세 slice 모두 macro-F1,
balanced accuracy와 confusion matrix를 필수로 보고하는 이유다.

`NarrativeQA`는 외부 검색 RAG는 아니지만 장문 context 독해이므로, 질문 자체의 순수
reasoning 난이도를 비교하는 주 slice에서는 분리한다. 새로운 RouterBench+xRouteBench
통합 artifact를 만들면 cross-source 중복 수가 달라질 수 있으므로 위 숫자를 그대로
복사하지 않고 동일 규칙으로 manifest의 실제 행 수를 다시 계산한다. 공식 RouterArena
리더보드 결과를 낼 때는 custom clean/slice가 아니라 요구되는 8,400행 전체를 사용하고,
이를 이 문서의 label-alignment 결과와 같은 점수라고 부르지 않는다.

## 3. 정제·결측·중복 계약

### 3-1. 공통 schema

```text
sample_id
source_dataset              # routerbench | xroutebench
source_split
category                    # source-local coarse category; model input에는 넣지 않음
source_task
query_text
model_input_text
valid_model_count
accepted_model_count
overall_solve_rate          # 0~1
difficulty_label            # easy | medium | hard
label_confidence
sample_weight
duplicate_group_id
split                       # train | validation | test | external_test
dataset_version
label_rule_version
preprocessing_version
```

RouterBench의 `category`는 기존 coarse `domain`(`knowledge`, `commonsense`, `math`, `code`,
`conversation`)을 쓰고 `source_task`에는 `eval_name`을 둔다. xRouteBench는 공식
`task_name`을 category로, `task_id`를 source task 식별에 사용한다. RouterArena는 공식
source/category를 평가용으로 보존한다. 서로 다른 source의 category 명칭을 억지로 공통
taxonomy에 매핑하지 않고 coverage는 항상 `source × category` 안에서 계산한다.

모델 응답, 정답, candidate model 이름별 점수와 비용은 감사용 metadata에만 둔다.
`training.parquet`에는 `model_input_text`, label, weight, category, split과 식별자만 넣으며,
category는 audit/evaluation metadata일 뿐 모델 입력에 붙이지 않는다.

### 3-2. xRouteBench query pivot와 결측

- `(query key, model_name)`은 유일해야 하며 중복 실행 행이 있으면 자동 선택하지 않고
  validation error로 기록한다.
- `performance`가 null인 모델은 실패로 바꾸지 않고 분모에서 제외한다.
- 18개 중 12개 미만만 유효하면 low confidence로 제외한다. 12는 후보군 2/3 이상을
  요구하는 이번 실험의 고정값이다.
- `em_mc`, `GSM8K`, `MATH`, `code_eval`처럼 correctness형 score는 `1.0`을 성공으로,
  F1 등 연속 score는 `>=0.70`을 성공으로 둔다. metric 이름과 acceptance rule이
  매핑되지 않으면 묵시적으로 0.70을 적용하지 않고 그 실행을 결측 처리한다.
- query별 `overall_solve_rate = accepted_model_count / valid_model_count`를 계산한다.

RouterBench도 같은 의미의 전체 solve rate를 다시 만들되, 기존 v2 tier별 rate와 v3
artifact를 삭제하거나 덮어쓰지 않는다.

### 3-3. 중복 제거와 split 우선순위

1. 두 학습 source와 두 외부평가 source 모두 Unicode NFKC, case-fold,
   공백·benchmark wrapper 정리를 적용한다. RouterArena는 text/hash만 오염 검사에 쓰고
   Difficulty 라벨은 보지 않는다.
2. normalized exact hash로 exact duplicate를 묶는다.
3. token 3-gram SimHash 후보에 Jaccard `>=0.90`을 적용해 near duplicate group을 만든다.
4. 동일 원천 benchmark의 `task_id`가 같으면 text가 조금 달라도 같은 group 후보로 본다.
5. cross-source group에는 **RouterArena/xRouteBench external test > RouterBench test >
   validation > train** 우선순위를 적용한다. 상위 split과 겹친 행은 가능하면 평가셋이
   아니라 train에서 제외해 평가 누수를 막는다.
6. 같은 split 안에서 중복이면 정보가 더 완전한 행을 대표로 남기고 나머지는 metadata에
   `duplicate_content`로 보존한다.

xRouteBench train은 group-aware 90/10으로 train/validation을 나눈다. RouterBench의
기존 고정 group split은 유지한다. xRouteBench 공식 test는 `external_test`로 두고,
RouterBench test와 합산 점수 하나만 내지 말고 source별 점수와 두 source macro average를
함께 낸다. 평가 질문과 겹쳐 제거된 train 행 수를 source·split별로 manifest에 기록한다.

## 4. 3단계 라벨 threshold 후보군과 목적함수

### 4-1. 성공 판정 경계와 난이도 경계

이 실험에는 이름이 비슷하지만 역할이 다른 두 경계가 있다.

1. **모델 실행의 성공 판정 경계:** F1·judge 같은 연속 `performance`를 성공/실패로
   바꾸는 `0.70`. 이는 query별 solve rate를 계산하기 위한 경계이며 이번 실험에서는 고정한다.
2. **질문 난이도 경계:** solve rate를 easy/medium/hard로 나누는 `(t_hard, t_easy)`.
   이것만 아래 후보군에서 선택한다.

두 데이터셋의 candidate model pool과 가격이 다르므로 RouterBench v2의 low/medium/high
tier 라벨을 xRouteBench에 그대로 복사할 수 없다. 통합 버전은 두 source 모두 계산 가능한
`overall_solve_rate`를 **후보 모델 집단이 맞힌 비율**로 정의하고 상대 난이도의
대리변수로 사용한다.

```python
if overall_solve_rate >= t_easy:
    label = "easy"
elif overall_solve_rate >= t_hard:
    label = "medium"
else:
    label = "hard"
```

### 4-2. 후보군

RouterBench v3에서 사용한 `(0.27, 0.64)`를 anchor로 두고, 후보 모델 한두 개 차이에
해당하는 이웃 경계만 비교한다. 큰 전역 grid를 쓰지 않아 우연히 잘 맞는 라벨을 찾는
multiple-testing 위험과 GPU 낭비를 줄인다.

| 후보 축 | 값 | RouterBench 11개가 모두 유효할 때의 의미 |
| --- | --- | --- |
| `t_hard` | `0.18`, **`0.27`**, `0.36` | medium의 최소 성공 수가 각각 약 2, 3, 4개 |
| `t_easy` | `0.55`, **`0.64`**, `0.73` | easy의 최소 성공 수가 각각 약 7, 8, 9개 |

교차곱 9개가 최종 후보군이다. xRouteBench는 최대 18개이고 질문마다 유효 모델 수가
12~18로 달라질 수 있으므로, 정수 성공 수로 다시 고정하지 않고 비율로 판정한다. 후보
config에는 소수점 값뿐 아니라 `candidate_id`, source별 실제 label count와 경계에 걸린
질문 수도 저장한다.

기존 `0.27/0.64` 결정은 폐기하지 않고 **anchor 후보**로 재해석한다. 새 통합 source에서는
model pool이 달라졌으므로 threshold를 그대로 정답으로 가정하는 것보다, 제한된 후보 안에서
재검증하고 동결하는 편이 학술적으로 타당하다.

### 4-3. threshold 선택 목적함수

먼저 전체 데이터를 duplicate group 단위 train/validation/test로 봉인한다. threshold
선택은 **train만** 대상으로 5-fold group OOF TF-IDF+LogisticRegression probe를 돌린다.
validation, test, xRouteBench 공식 test, RouterArena Difficulty는 이 단계에서 보지 않는다.

모든 후보가 먼저 4-5의 sanity gate를 통과해야 하며, 통과 후보의 점수는 다음과 같다.

```text
J_threshold =
    0.25 * OOF source_macro_F1
  + 0.15 * OOF balanced_accuracy
  + 0.15 * OOF QWK_01
  + 0.15 * min(OOF RouterBench macro_F1, OOF xRouteBench macro_F1)
  + 0.10 * (1 - ordinal_MAE / 2)
  + 0.10 * (1 - hard_to_easy_rate)
  + 0.10 * candidate_model_bootstrap_label_agreement
```

`QWK_01=(QWK+1)/2`로 바꿔 0~1 범위에 맞춘다. 모든 항은 높을수록 좋다. 이 목적함수는
단순 accuracy 대신 세 클래스 균형, 순서가 있는 오류, source 중 하나의 붕괴, 가장 위험한
`hard -> easy`, candidate-model 표본 변화에 대한 라벨 안정성을 함께 본다.

선택 규칙은 다음과 같다.

1. sanity gate를 통과한 후보 중 평균 `J_threshold`가 가장 높은 후보를 공식 비교용
   **T1**으로 고른다.
2. 최고점과 `J_threshold` 차이가 `0.005` 미만인 후보들은 사전에 정한 동률 허용 범위로 보고,
   4-6의 core-category `red` 셀이 가장 적은 후보를 우선한다. red 셀 수도 같으면 더
   단순하고 기존 v3와 연결되는 anchor `(0.27, 0.64)`를 우선하고, 그다음 높은 점수를 쓴다.
3. T1을 제외하고 sanity gate를 통과했으며 T1과 train 라벨 불일치율이 `>=5%`인 후보 중
   `J_threshold`가 가장 높은 것을 민감도 분석용 **T2**로 고른다. 가능한 경우 T1에서
   한 경계만 한 grid step 이동한 후보를 동점 우선한다.
4. 5% 조건을 만족하는 후보가 없으면 남은 후보 중 `J_threshold`가 가장 높은 것을 T2로
   고르고 실제 불일치율이 5% 미만임을 보고한다. T2를 만들기 위해 9개 밖으로 탐색 공간을
   넓히지 않는다.
5. 동률이면 bootstrap agreement, `hard -> easy`, 두 source label 분포
   Jensen–Shannon divergence가 작은 순으로 고른다.
6. T1·T2 config, 후보별 train OOF prediction SHA-256과 상호 라벨 불일치율을 함께 봉인하고,
   neural model별로 threshold를 다시 고르지 않는다.

T1은 전 비교군의 순위, provisional winner와 production 판단에 쓰는 유일한 주 기준이다.
T2는 먼저 TF-IDF와 ModernBERT의 seed 42에만 적용한다. T2에서 두 모델의 상대 순위가
T1과 달라지거나 source별 macro-F1 또는 `hard -> easy`의 절대 차이가 `>=0.01`일 때만
T1 validation 상위 2개 모델의 seed 42까지 확장한다. T2 결과로 T1 winner를 사후 교체하지
않고, 임계값 정의에 대한 민감도 결과로 별도 표에 보고한다. test와 RouterArena를 보고
T2 적용 범위를 결정하지 않는다. T2 대상이 validation까지 완료되면 selection lock 뒤
동일한 test와 RouterArena에 한 번 추론해 외부 일반화도 별도로 기록한다.

이 점수는 “자연의 참 난이도”를 찾아내는 공식이 아니다. 관측된 후보 모델 집단의
정답률로 만든 라벨이 **안정적이고 질문 텍스트에서 예측 가능하며 두 source에서 함께
작동하는지** 고르는 연구용 조작적 정의다.

### 4-4. 원점수 acceptance

- exact-match, multiple-choice, unit-test correctness는 `performance == 1.0`만 성공이다.
- F1·judge rating 등 명시된 연속 metric은 `performance >= 0.70`을 성공으로 둔다.
- metric 이름과 acceptance rule이 매핑되지 않으면 0.70을 임의 적용하지 않고 그
  모델 실행을 결측 처리한다.
- 유효 candidate model 수가 xRouteBench는 12/18 미만이면 질문 전체를 제외한다.
- 비용·모델 설명·token price는 라벨 계산에 넣지 않는다. 이번 target은 비용 최적화
  reward가 아니라 관측 solve rate 기반 난이도다.

### 4-5. 라벨 sanity gate

새 통합 artifact는 학습 전에 다음 조건을 모두 통과해야 한다.

- 전체 train에서 각 클래스 10% 이상이면서 500행 이상
- source별 세 클래스가 모두 존재
- easy의 평균 solve rate > medium > hard
- candidate model bootstrap 1,000회 재표본의 label agreement 평균 `>=0.85`
- domain/source별 행 수와 라벨 비율을 artifact에 저장

gate를 통과하는 후보가 없으면 RouterArena를 보며 경계를 조정하지 않고 결측·metric
mapping·중복 처리 오류를 먼저 수정한다. 9개 밖으로 후보를 넓히려면 새 label-rule
version을 만들고 threshold 선택용 OOF를 처음부터 다시 실행한다.

### 4-6. 카테고리×난이도 coverage audit

전체 클래스 sanity gate와 별개로, 9개 후보의 train coverage를 먼저 계산해 4-3의
근접 후보 tie-break에 사용하고, 선택된 T1/T2에는 pre/post-split
`source × split × category × label` 행 수와 비율을 모두 계산한다. 이 audit은 카테고리마다
easy/medium/hard를 1:1:1로 만드는 목표가 아니다. 난이도는 모든 카테고리에서 같은
solve-rate threshold를 뜻해야 하므로 category별 threshold를 만들거나 다수 클래스를
버려 비율을 맞추지 않는다.

전체 train의 학습 친화적인 권장 범위는 각 클래스 `20~45%`지만 이는 선택 gate가 아니다.
실제 hard gate는 4-5의 클래스별 `>=10%`이면서 `>=500행`이다. train eligible 행이
`>=300`인 카테고리를 **core category**로 정하고 다음 coverage 상태를 적용한다.

| 범위 | `green` 권장 | `yellow` 허용·주의 | `red` 부족 |
| --- | --- | --- | --- |
| core-category train | 모든 label 비율 `>=15%`, 최다 label `<=70%`, 각 cell `>=100행` | green은 아니지만 모든 label `>=5%`, 최다 label `<=85%`, 각 cell `>=30행` | 어느 label이 `<5%` 또는 `<30행`, 혹은 최다 label이 `>85%` |
| validation/test의 category×label cell | 각 cell `>=100행` | 각 cell `30~99행` | 어느 cell이 `<30행` 또는 label 부재 |

`coverage-report.parquet`에는 최소한 `threshold_id`, `phase(pre/post_split)`, `source`,
`split`, `category`, `label`, `category_total`, `cell_count`, `cell_share`, `coverage_status`,
`reason`, `resolution`, `split_version`, `policy_version`을 저장한다.

train eligible 행이 300 미만인 카테고리는 비율만으로 판정하지 않고 `small_category`로
기록한다. 원래 category 이름과 count는 보존하며, 의미상 인접한 상위 category로 묶을지는
별도 taxonomy 결정으로만 허용한다.

전처리 결과는 다음 순서로 처리한다.

1. 라벨 직후 pre-split과 group-aware split 뒤 post-split coverage를 모두 저장한다.
2. pre-split은 yellow 이상인데 post-split만 red이면 duplicate group을 쪼개지 않는 범위에서
   `category × label` 분포를 더 잘 보존하도록 새 split version을 생성하고, 해당 version의
   train-only 9개 threshold OOF와 T1/T2 선택부터 다시 실행한다. 이미 선택한 threshold에
   맞춰 split만 사후 조정하지 않는다. RouterBench 고정 test와 xRouteBench 공식 test는
   이동하지 않으며, 재생성 가능한 train/validation partition만 대상으로 한다.
3. pre-split부터 한 카테고리만 red이면 T1을 category별로 바꾸지 않고 자연분포를 유지한다.
   해당 cell은 `insufficient_coverage`로 기록하고, 운영상 중요한 카테고리일 때만 같은
   candidate-model pool과 threshold로 라벨링 가능한 데이터를 새 dataset version에 보강한다.
4. 같은 label의 red가 전체 core category의 `>=20%`이면서 RouterBench와 xRouteBench에
   각각 최소 한 category 이상 존재하면
   `coverage_systemic_warning`을 발생시킨다. full model 학습 전에 T1/T2와 0.005 이내
   근접 후보의 solve-rate histogram, metric mapping과 결측 처리를 train만 사용해 재검토한다.
   RouterArena나 test를 보고 threshold를 바꾸지 않는다.
5. category red는 자동적인 전체 dataset 폐기 사유가 아니지만, `keep-natural`, `resplit`,
   `collect-new-version`, `merge-taxonomy` 중 resolution을 manifest에 기록하기 전에는 full
   sweep을 시작하지 않는다. 전체 class gate 실패는 계속 hard failure다.

평가에서는 validation/test/RouterArena를 oversampling·downsampling하지 않는다. 한
category의 세 label cell이 모두 100건 이상일 때만 category-level 결과를 confirmatory로
해석한다. 모두 30건 이상이지만 하나라도 100건 미만이면 95% 신뢰구간을 붙인 exploratory
결과로 보고한다. 하나라도 30건 미만이면 해당 category의 3-class macro-F1과 label별
우열 주장을 `insufficient_coverage`로 두고, count·비율·QWK 또는 전체 category accuracy만
참고값으로 남긴다. 이 coverage 상태는 모델 winner를 사후 교체하는 기준이 아니다.

### 4-7. 모델 선정 목적함수와 production 판단

threshold를 동결한 뒤 모델 비교에는 별도 validation 품질 점수를 쓴다.

```text
Q_model =
    0.45 * validation source_macro_F1
  + 0.15 * validation balanced_accuracy
  + 0.15 * validation QWK_01
  + 0.15 * (1 - hard_to_easy_rate)
  + 0.10 * (1 - ECE)
```

`Q_model`은 학술적 품질 요약값이며 latency를 섞지 않는다. 먼저 7-5의 seed·bootstrap
규칙으로 통계적 품질을 판단하고, 차이가 실질적으로 작을 때 p95 latency, peak memory,
artifact size로 production Pareto **권고 모델**을 고른다. 이번 운영 재학습 모델은
ModernBERT로 고정하므로 최고 `Q_model`, Pareto 권고, 실제 운영 ModernBERT를 보고서에
각각 명시한다.

## 5. 학습 데이터 비율

### 5-1. split과 source 혼합

| 원천 | train | validation | 최종평가 |
| --- | --- | --- | --- |
| RouterBench text-only | 기존 duplicate-group `80%` | 기존 `10%` | 기존 test `10%` |
| xRouteBench `llmrouter_generic` | 공식 train을 group-aware `90%` | 공식 train의 `10%` | 공식 test 3,729개 전량 |

통합 학습은 두 원천 train을 단순 concatenate한 뒤 전체 행을 shuffle한다. source를 50:50으로
oversampling하거나 RouterBench를 잘라내지 않는다. RAG 800개 제거 전 RouterBench와
xRouteBench 상한으로 계산하면 실제 train source 비율은 약 **87:13**이 예상되지만,
cross-source 중복·결측 제거 후 확정 비율과 행 수는 preprocessing manifest에 기록한다.
즉 `87:13`은 목표 quota가 아니라 실행 전 예상값이다.

source 규모가 다른 영향을 숨기지 않기 위해 validation checkpoint 선택은 합친 전체 F1이
아니라 RouterBench macro-F1과 xRouteBench macro-F1의 산술평균인 `source_macro_f1`을 쓴다.
이렇게 하면 작은 xRouteBench를 학습에서 복제하지 않으면서도 검증에서 무시되지 않는다.
duplicate group을 한 split에만 두는 제약을 최우선으로 하되, 가능한 해 중에서는
`source × category × label` 분포 편차가 가장 작은 group-aware split을 선택한다. 4-6의
post-split red가 pre-split에 없었다면 현재 split version을 실패로 간주하고, 재생성 가능한
train/validation partition만 다시 만든 뒤 threshold 선택부터 재실행한다.

### 5-2. class 비율

validation, test, external_test뿐 아니라 train도 **선정된 threshold가 만든 자연분포 전량**을
유지한다. v2에서 1:1:1 전역 균형화가 도메인 내부 easy 신호를 훼손했으므로 class
downsampling, oversampling, balanced sampler와 class-weighted loss를 주 비교에 사용하지
않는다. 모든 neural model은 같은 일반 shuffle과 unweighted cross-entropy 또는 동등한
label-token loss를 사용한다.

class weighting, source-balanced sampler와 soft-cap은 최종 winner에 대한 사후 ablation으로만
허용한다. 기본 ablation은 `natural`, `source 80:20 weighted sampler`, `class-weighted loss`
세 가지이며, 그 결과는 architecture 주 순위와 별도 표에 싣는다. RouterArena 결과를 보고
어느 ablation을 채택할지는 고르지 않는다.

4-6의 yellow/red coverage는 주 비교 train을 재균형화하라는 뜻이 아니다. 특히 한
카테고리의 medium이 적다는 이유로 easy/hard를 버리거나 medium을 복제하지 않는다.
주 비교는 자연분포 전량을 유지하고, 부족 cell의 추가 수집·sampler·loss 변경은 별도
dataset version 또는 사후 ablation으로만 수행한다.

## 6. 최종 모델군과 비교 가설

아래 다섯 neural checkpoint와 비신경 기준선 하나를 **필수 비교군**으로 확정한다.

| 역할 | checkpoint | 크기·문맥 | 학습·추론 형태 | 확인할 가설 |
| --- | --- | --- | --- | --- |
| 비신경 기준선 | TF-IDF + LogisticRegression | 해당 없음 | 3-class 선형 분류 | 표층 단어·출처 단서의 상한 |
| 경량 encoder | `microsoft/MiniLM-L12-H384-uncased` | 33M, 512 | sequence classification | 비슷한 품질이면 serving 비용을 줄일 수 있는가 |
| 고전 control | `FacebookAI/roberta-base` | 약 125M, 512 | sequence classification | ModernBERT 이점인지 일반 encoder fine-tuning 효과인지 |
| 현행 기준 | `answerdotai/ModernBERT-base` | 149M, 최대 8,192 | sequence classification | 현대 encoder·코드 pretraining 이점이 있는가 |
| 강한 동급 encoder | `microsoft/deberta-v3-base` | 약 184M total, 512 | sequence classification | disentangled attention이 약한 난이도 신호를 더 잘 잡는가 |
| 소형 생성형 LLM | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B, 최대 32,768 | label-token causal LM | 생성형 문맥 해석이 encoder보다 이득인가 |

공식 참고:

- [ModernBERT-base model card](https://huggingface.co/answerdotai/ModernBERT-base)
- [DeBERTa-v3-base config](https://huggingface.co/microsoft/deberta-v3-base/blob/main/config.json)
- [RoBERTa-base config](https://huggingface.co/FacebookAI/roberta-base/blob/main/config.json)
- [MiniLM-L12-H384 model card](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Qwen은 Arch-Router 등에서 관찰된 소형 생성형 router 계열을 최소 비용으로 대표한다.
Qwen2.5-1.5B 대신 0.5B를 고른 이유는 T4 한 장에서 전체 실험을 반복할 수 있고, router
자체 지연이 Claude 호출 절감분을 잠식하지 않게 하기 위해서다. Qwen의 0.49B는 encoder
세 모델과 동급 크기가 아니므로 품질이 높더라도 **decoder가 우월하다**고만 해석하고
파라미터 효율 우위라고 해석하지 않는다.

ModernBERT-large, Qwen 1.5B 이상, ELECTRA, DeBERTa-v3-small, GNN·bandit router는 이번
비교군에서 제외한다. 같은 역할의 후보를 더 늘리면 제한된 T4 반복 횟수가 줄고,
질문×후보모델 correctness 외에 그래프·온라인 feedback 같은 추가 입력을 요구하는 방법은
현재 “질문 하나→3단계” serving 계약과 공정 비교할 수 없다.

## 7. 공정 학습 계약

### 7-1. 공통 조건

- 같은 versioned training/validation/test artifact와 duplicate groups
- primary 비교는 모든 모델 `max_length=512`; ModernBERT 8,192는 별도 secondary 결과
- 같은 seed 42, effective batch 32, FP16, AdamW, weight decay 0.01
- 같은 example exposure: 최대 3 epoch
- epoch당 같은 네 지점(25/50/75/100%)에서 validation
- RouterBench와 xRouteBench validation macro-F1의 산술평균인 `source_macro_f1`로
  best checkpoint 선택
- `early_stopping_patience=4 evaluation points`, `min_delta=0.003`
- RouterBench/xRouteBench test와 RouterArena는 설정·winner 동결 뒤 고정된 전 비교군에
  모델별 1회 실행
- 각 모델의 tokenizer revision, 실제 token 수, truncation, GPU, runtime, wall time 기록
- neural 비교의 validation ECE와 `Q_model`에는 5-fold cross-fitted scalar temperature
  calibration을 사용하고, test 실행 직전 각 checkpoint의 최종 temperature를 전체
  validation에 한 번 적합한다. raw confidence 지표도 함께 보존한다.

batch가 메모리에 맞지 않으면 micro-batch만 줄이고 gradient accumulation으로 effective
batch 32를 유지한다. 그래도 OOM이면 max length를 몰래 줄이지 않고 해당 모델의
`resource_incompatible` 결과로 남긴다.

### 7-2. learning rate와 epoch 형평성

모든 architecture에 같은 learning rate 하나를 강제하면 특정 모델에 불리하다. 대신
10% train의 1 epoch pilot에서 각 모델에 동일한 3개 후보 `{1e-5, 2e-5, 3e-5}`를 주고,
동일한 primary metric으로 하나를 선택한다. full run은 선택된 learning rate와 최대
3 epoch를 사용한다.

모델이 epoch 2에 끝나더라도 early stopping에 따른 정상 종료다. epoch 수를 맞추기 위해
성능이 나빠지는 epoch 3 checkpoint를 최종본으로 강제하지 않는다. 반대로 3 epoch에서
계속 개선 중이면 5 epoch 추가 실험은 “training-to-convergence” 보조 결과로 분리하며
1차 공정 순위에는 넣지 않는다.

### 7-3. encoder와 Qwen의 출력 계약

네 encoder는 `AutoModelForSequenceClassification(num_labels=3)`과 unweighted
cross-entropy로 학습한다. class 순서는 `0=easy, 1=medium, 2=hard`로 고정한다.

Qwen은 다음 instruction 뒤에 정답 label 한 단어만 이어 붙여 causal-LM loss로
full fine-tuning한다. reasoning 문장이나 후보 LLM 설명은 넣지 않는다.

```text
Classify the difficulty of the user question as easy, medium, or hard.
Return exactly one label.

Question:
{model_input_text}

Label:
```

추론 때 자유 생성 결과를 정규식으로 고르지 않는다. `easy`, `medium`, `hard` 세
label-token sequence의 길이 정규화 conditional log-likelihood를 각각 계산하고 가장 큰
label을 선택한다. 세 점수를 softmax해 confidence를 만들므로 ECE도 encoder와 같은 방식으로
계산할 수 있다. full FP16 fine-tuning이 T4에서 OOM이면 4-bit/LoRA로 조용히 바꾸지 않고
`resource_incompatible`로 기록한다. QLoRA는 별도 후속 실험이다.

### 7-4. epoch가 없는 기준선

TF-IDF+LogisticRegression에는 epoch를 억지로 정의하지 않는다. 주 기준선은 lowercase,
word `(1,2)`-gram, `min_df=2`, `sublinear_tf=true`, `max_features=100000`, multinomial
LogisticRegression `C=1.0`, `class_weight=None`, `max_iter=2000`, seed 42로 고정한다.

- 같은 train/validation/test와 input text를 사용한다.
- 이 고정 기준선을 보고 neural model 설정을 반복 조정하지 않는다.
- 품질 지표 외에 fit time, peak RAM/VRAM, p50/p95 latency와 artifact size를 함께 낸다.
- neural fine-tuning과 “동일 계산량”이라고 주장하지 않고 **품질-비용 Pareto baseline**으로
  비교한다.

### 7-5. 검증·통계·winner 규칙

accuracy 하나로 결정하지 않는다. 최소 보고 항목은 accuracy, macro/weighted F1,
class별 precision/recall/F1, balanced accuracy, QWK, ordinal MAE, under/over-routing,
`hard -> easy`, ECE, p50/p95 latency, peak VRAM, 학습 시간이다. 모두 0~1인 지표와
시간·메모리 단위를 표에서 분리한다.

절차를 다음 순서로 고정한다.

1. seed 42의 validation `Q_model`로 neural 상위 2개를 고른다.
2. 두 finalist만 seed 17/73을 추가해 총 3 seeds 평균·표준편차를 계산한다.
3. validation 예측을 source×label로 층화해 paired bootstrap 10,000회로 두 finalist와
   TF-IDF 차이의 95% 신뢰구간을 계산한다.
4. 평균 `Q_model`이 가장 높은 모델을 provisional winner로 동결한다.
5. `Q_model` 차이가 0.005 미만이면 severe `hard -> easy` 비율이 낮은 모델, 다시 동점이면 p95
   latency, peak memory, artifact size 순으로 작은 모델을 고른다.
6. 그 뒤에만 고정된 전 비교군을 RouterBench test, xRouteBench 공식 test,
   RouterArena 세 slice에서 모델별 1회 평가한다.

test와 RouterArena는 이미 고른 winner를 **확인하거나 승격을 거부**하고 후보별 외부
특성을 비교할 수는 있지만, runner-up으로 다시 고르는 데 사용하지 않는다. test를 보고
후보를 바꾸려면 새 실험 버전과 새 untouched holdout이 필요하다. winner가 majority-class accuracy와 TF-IDF
macro-F1를 모두 넘지 못하거나 TF-IDF 대비 macro-F1 차이의 bootstrap 신뢰구간이 0을
포함하면 neural model 우위를 주장하지 않고 “데이터·라벨 병목”으로 결론낸다.

### 7-6. RouterArena label-alignment 평가

RouterArena에서는 실제 Claude 답변을 생성하지 않고 각 비교 모델의 3단계 예측과 공식
Difficulty를 비교한다. 따라서 결과 이름은 `RouterArena difficulty-label alignment`로
고정하며 Claude answer accuracy나 비용 절감이라고 부르지 않는다.

세 slice 각각에 accuracy, macro-F1, balanced accuracy, QWK, ordinal MAE,
`hard -> easy`, 3×3 confusion matrix와 source/category별 macro-F1를 낸다. 주 외부 지표는
`rb_service_aligned` macro-F1이고, `rb_unseen`은 OOD 진단, `full_clean`은 전체 참고다.
RouterArena Difficulty는 42개 이종 모델 중 정답 모델 수로 만든 상대 난이도이며 특정
Claude tier의 최소 충분성을 뜻하지 않는다.

### 7-7. Claude 연결 후 별도 end-to-end 검증

ModernBERT production 승격 전에는 Spring이 실제로 설정한 `easy/medium/hard -> Claude model ID`
snapshot을 고정하고, RouterArena 또는 서비스 허용 범위에서 최소 450건을 난이도×도메인으로
층화 표본한다. `always-cheap`, `always-middle`, `always-strong`, router 네 정책을 같은
prompt/scorer로 비교하고 다음을 보고한다.

- task별 실제 answer correctness와 95% bootstrap 신뢰구간
- 1,000 query당 실제 API 비용과 입력·출력 token 수
- always-strong 대비 품질 보존율과 비용 절감률
- 가장 저렴한 정답 모델 대비 selection regret
- `hard -> easy` 실제 실패와 escalation 필요율

이 단계 전에는 RouterArena 라벨 일치도가 높더라도 Claude 서비스 성능이 검증됐다고
결론내리지 않는다. tier→Claude model mapping은 Spring이 소유하므로 mapping 변경만으로
라우터를 재학습하지 않고, 평가 report에 policy version을 함께 기록한다.

### 7-8. 결과 원인 분석

최고 모델 이름만 보고하지 않고 같은 질문에 대한 후보별 예측을 `sample_id`로 결합해
다음 분석을 수행한다.

- **동급 architecture:** RoBERTa/ModernBERT/DeBERTa의 paired correctness 차이를
  source×label bootstrap으로 비교한다.
- **길이:** 0~128, 129~256, 257~512, 512 초과 원문 길이 bucket별 성능과 truncation을
  비교해 ModernBERT 장문 이점 주장 여부를 판단한다.
- **내용 유형:** knowledge, commonsense, math, code, conversation과 xRoute task별
  class recall을 비교한다.
- **출처 shortcut:** 질문을 보지 않고 source 최빈 label만 내는 baseline과 TF-IDF를
  함께 두고, wrapper 제거 전후 성능 차이로 출처 문체 의존을 측정한다.
- **오류 중첩:** 두 모델이 모두 틀린 질문, 한 모델만 맞힌 질문, severe `hard -> easy`
  질문을 각각 50건까지 정성 검토한다.
- **효율:** macro-F1 대 p95 latency, peak memory, artifact size Pareto plot을 만든다.
- **calibration:** confidence bin reliability diagram과 ECE를 label·source별로 비교한다.

한 checkpoint씩만 비교하므로 결과의 원인을 architecture 하나로 단정하지 않는다.
동급 크기, 같은 데이터·노출·길이·optimizer 계약과 paired error 분석이 함께 같은 방향을
가리킬 때만 “해당 구조가 이 태스크에 유리하다”는 근거로 해석한다.

관찰 결과별 1차 해석은 다음처럼 미리 정해 두어, 결과를 본 뒤 설명을 맞추지 않는다.

| 관찰 | 지지되는 해석 | 결론 전에 확인할 반증 |
| --- | --- | --- |
| TF-IDF가 neural과 동률 또는 우세 | 난이도 라벨이 표층 어휘·출처 문체에 크게 의존하거나 텍스트만으로 예측 가능한 신호가 약함 | wrapper 제거 후에도 유지되는지, source-only baseline보다 나은지 |
| MiniLM이 최고점과 `0.005` 이내이며 훨씬 빠름 | 3단계 분류에는 큰 encoder 표현력이 필요하지 않아 압축 모델이 production Pareto | hard recall·OOD·calibration이 악화되지 않는지 |
| RoBERTa와 ModernBERT가 동률 | 긴 문맥/현대 pretraining 이점이 공통 512-token 주 실험에는 거의 없음 | 512 초과 secondary와 code slice에서 ModernBERT가 이기는지 |
| ModernBERT가 특히 code·장문에서 우세 | 코드/긴 문맥 pretraining과 효율적 attention이 이 데이터 특성에 도움을 줬을 가능성 | 같은 길이 bucket paired CI, truncation 차이 |
| DeBERTa가 동급 encoder에서 우세 | disentangled content/position 표현이 약한 난이도 단서를 잡는 데 유리할 가능성 | vocab·tokenization·LR 차이가 원인인지 |
| Qwen이 encoder보다 유의하게 우세 | instruction-tuned causal LM의 생성형 의미 표현이 label-token 판단에 추가 신호를 줌 | 0.49B 규모 효과와 모델 유형 효과를 구분할 수 없음을 명시, CPU SLO 확인 |
| validation만 좋고 test/RA aligned가 급락 | source shortcut, threshold 과적합 또는 분포 이동 | source/category별 confusion, dedup·오염 report |

학술 결론은 “A가 항상 B보다 우월”이 아니라 **이 데이터 계약·파라미터 범위·입력 길이에서
관찰된 차이**로 한정한다. 원인 주장은 위 사전 가설, paired 신뢰구간, slice 분석 중 최소
두 종류가 같은 방향을 보일 때만 채택한다.

### 7-9. ModernBERT 데이터 충원 비교

데이터 충원 실험은 조건을 늘리지 않고 T1에서 다음 두 run만 비교한다.

| run ID | 학습 데이터 | threshold | seed | 역할 |
| --- | --- | --- | ---: | --- |
| `MB-RB-T1` | RouterBench text-only train | T1 | 42 | 충원 전 control |
| `MB-MIX-FULL-T1` | RouterBench text-only + xRouteBench `llmrouter_generic` train 전량 | T1 | 42 | 유일한 데이터 충원 조건 |

`MB-MIX-FULL-T1`은 6개 모델 T1 full sweep의 ModernBERT run과 동일하므로 별도로 다시
학습하지 않고 그 artifact를 재사용한다. 비교를 성립시키기 위해 `MB-RB-T1`은 같은
ModernBERT revision, tokenizer, max length, optimizer, epoch·early-stopping 계약으로
실행한다. 두 run은 동일한 통합 validation/test와 같은 T1 라벨로 평가한다.

`MB-MIX-N`, 데이터 규모별 부분집합, T2 기반 충원 run, 조건별 threshold 재선정과
equal-step 보조 실험은 이번 범위에서 제외한다. 따라서 이 비교는 xRouteBench 행 추가에
따른 **데이터 양·출처 다양성·optimizer step 증가가 결합된 실용적 충원 효과**를 측정한다.
순수한 데이터 규모의 인과효과로 표현하지 않는다. 기존 RouterBench v2 ModernBERT는
라벨·균형화·threshold가 달라 역사적 참고치로만 두며 `MB-RB-T1`을 대신하지 않는다.

## 8. Colab notebook 실행 순서와 자원 계획

ModernBERT v2의 실측은 18,768 train, 3 epoch, RouterBench/RouterArena 관찰 포함
20,884초(5시간 48분)였다. 통합 train은 RAG·중복 제거 후 약 29K~31K로 예상하므로
아래 값은 **200 optimizer-step pilot 전의 계획 범위**이며 deadline이 아니다.

| 작업 | 예상 시간 | 비고 |
| --- | ---: | --- |
| 데이터 pivot·정제·dedup·9개 threshold OOF | CPU 2~5시간 | 최초 1회, download 제외 |
| 다섯 neural 모델 200-step calibration pilot | GPU 2~4시간 합계 | 실제 throughput·VRAM 측정 |
| MiniLM 3 epoch | GPU 3~5시간 | 33M 경량 후보 |
| RoBERTa-base 3 epoch | GPU 6~9시간 | dynamic padding 기준 |
| DeBERTa-v3-base 3 epoch | GPU 8~12시간 | 큰 vocab/relative attention 반영 |
| ModernBERT-base 3 epoch | GPU 8~10시간 | 기존 실측을 데이터 비율로 보정 |
| ModernBERT `MB-RB-T1` control | GPU 5~7시간 | 데이터 충원 비교용 RouterBench-only 1회 |
| Qwen2.5-0.5B 3 epoch | GPU 10~18시간 | label-token causal LM, pilot로 교체 |
| 1차 full sweep 합계 | GPU 35~54시간 | 세션 중단·Drive sync 제외 |
| T1 sweep + `MB-RB-T1` | GPU 40~61시간 | 데이터 충원 control 포함, 병렬화 전 GPU 시간 |
| T2 민감도 | GPU 8~10시간 + CPU | ModernBERT 1회와 TF-IDF; 확장 조건 충족 시 별도 추가 |
| finalist 추가 2 seeds | GPU 12~24시간/모델 | 두 finalist면 합계 24~48시간, early stopping에 따라 감소 가능 |

위 시간은 Colab T4 실측을 데이터 규모로 보정한 초기값이다. 다중 모델 비교의 공식
실행 환경은 **Google Colab notebook**이며 KFP pipeline이나 Vertex AI CustomJob을 만들지
않는다. Colab GPU 종류와 세션 지속 시간은 보장되지 않으므로 한 세션에서 full sweep을
시도하지 않고 모델별 notebook이 자신의 checkpoint를 Google Drive에 원자적으로 동기화한다.
pilot 뒤 각 모델의 ETA는 다음 식으로 교체한다.

```text
ETA_full = pilot_wall_time
         * planned_optimizer_steps / pilot_optimizer_steps
         * 1.15
         + measured_validation_time
```

`1.15`는 checkpoint·Drive sync·Colab 런타임 편차 여유다. 각 notebook은 로컬 임시 SSD에
먼저 checkpoint를 완결하고 checksum과 `_SUCCESS`를 만든 뒤 Drive staging directory를
원자 rename한다. 새 세션은 model/config/dataset fingerprint가 같은 마지막 정상 checkpoint만
재개한다. 세션 수가 달라도 최종 example exposure와 optimizer step은 공통 계약을 지켜야 한다.

### 8-1. notebook 구성

아래 파일은 `ml/notebooks/router_model_comparison/` 아래에 둔다. 각 모델 notebook은 다른
모델을 조건문으로 전환하는 거대한 단일 notebook이 아니라, 공통 Python helper와 artifact
schema만 공유하는 독립 실행 단위다.

| 순서 | notebook | 역할 |
| ---: | --- | --- |
| 00 | `00_prepare_dataset_and_thresholds.ipynb` | source 정제·dedup·split·9개 threshold OOF, pre/post-split coverage audit·resolution, 공통 dataset manifest 생성 |
| 01 | `01_tfidf_logreg.ipynb` | TF-IDF+LogisticRegression 기준선과 validation prediction 생성 |
| 02 | `02_minilm.ipynb` | MiniLM smoke/pilot/full train과 validation prediction 생성 |
| 03 | `03_roberta.ipynb` | RoBERTa-base 동일 절차 수행 |
| 04 | `04_modernbert.ipynb` | `MB-MIX-FULL-T1`, `MB-RB-T1`과 T2 민감도 run 수행; production publish 금지 |
| 05 | `05_deberta.ipynb` | DeBERTa-v3-base 동일 절차 수행 |
| 06 | `06_qwen.ipynb` | Qwen label-likelihood 학습·validation prediction 생성 |
| 07 | `07_select_evaluate_and_report.ipynb` | validation winner 동결, 전 모델 test/RouterArena 1회 평가, bootstrap·Pareto 보고서 생성 |

`00`이 만든 dataset SHA-256과 `experiment-config.json`을 모든 notebook이 시작할 때 검증한다.
`01~06`은 첫 단계에서 validation까지만 생성한다. `07`이 `selection-lock.json`을 쓴 뒤에만
봉인된 checkpoint를 순차 로드해 RouterBench test, xRouteBench test, RouterArena 세 slice를
평가한다. 이로써 노트북이 분리돼도 test를 보고 학습 설정을 바꾸는 누수를 막는다.

### 8-2. 실행 순서

1. RouterBench+xRouteBench+RouterArena text hash를 모아 contamination report를 만든다.
2. duplicate-group split을 봉인하고, train-only 9개 threshold OOF와 후보별 train
   coverage를 계산해 T1·T2를 동결한다.
3. `00`에서 선택된 T1/T2의 pre/post-split coverage를 판정한다. split-induced red면 새
   split version으로 2단계부터 다시 실행한다. 모든 red resolution이 기록되면 T1·T2 label column을 가진
   통합 train/validation, 봉인된 두 test manifest와 coverage report를 Drive에 게시한다.
4. `01`에서 TF-IDF 기준선을 실행하고 `02~06`에서 각 모델 100행 smoke와 200-step pilot으로
   호환성·VRAM·ETA를 확인한다.
5. T1 seed 42 full run은 Colab Pro에서 서로 다른 runtime·Drive run directory를 사용해
   최대 두 개씩 병렬 실행한다. 권장 wave는 `MiniLM + RoBERTa`,
   `ModernBERT MB-MIX-FULL + DeBERTa`, `Qwen + ModernBERT MB-RB`이며 두 번째 GPU가
   배정되지 않으면 동일 순서의 단일 실행으로 전환한다. TF-IDF는 CPU에서 병행할 수 있다.
6. TF-IDF와 ModernBERT의 T2 seed 42 민감도 run을 실행하고 4-3의 확장 조건을 판단한다.
   T2는 데이터 충원 비교에는 적용하지 않는다.
7. validation 상위 2개 notebook에만 T1 seed 17/73을 추가하고 `07`에서 provisional winner와
   `selection-lock.json`을 동결한다.
8. `07`에서 고정된 전 비교군의 T1과 사전 확정된 T2 대상에 RouterBench test,
   xRouteBench test, RouterArena 세 slice를 모델·threshold별 1회 실행한다. T2 결과는
   winner 변경에 사용하지 않는다.
9. 비교 결과와 무관하게 운영 초기 후보는 동일 데이터 SHA를 입력으로 받은 기존
   `modernbert_full_colab.ipynb`의 통합-data profile에서 별도 학습한다. Drive의 최종
   `_SUCCESS` checkpoint를 package-v1로 검증·게시하고 gate를 통과하면 Claude 450건
   end-to-end 평가를 실행한다. Vertex initial-training job은 제출하지 않는다.

TF-IDF, MiniLM, 동급 encoder 세 모델을 Qwen보다 먼저 실행하면 생성형 실험이 늦어져도
핵심 “동급 파라미터 구조 비교”와 “경량화 가능성” 결론을 먼저 확보할 수 있다. 그러나
최종 보고서에서 Qwen 행을 누락한 채 전체 모델군 비교가 끝났다고 표시해서는 안 된다.

## 9. 버전·산출물·최종 표 계약

### 9-1. 실행 전에 config로 고정할 값

- dataset/label/preprocessing version과 입력 artifact SHA-256
- 다섯 checkpoint의 immutable Hugging Face revision과 license snapshot
- 9개 threshold 후보, 선택된 T1·T2 `t_hard/t_easy`, 상호 라벨 불일치율, OOF fold와 목적함수 version
- continuous acceptance `0.70`, xRouteBench minimum valid `12`
- category coverage policy version, core-category 최소 300행, train cell `100/30행`과
  `15/5%` green/yellow 경계, evaluation cell `100/30행` 경계
- natural train, max length 512, effective batch 32, seed 42/17/73
- 모델별 pilot에서 고른 learning rate와 선택 근거
- RouterArena slice source 목록과 현재 production scope version
- Colab runtime lock, notebook git commit, accelerator 관찰값과 Drive experiment root
- 모델별 notebook filename·실행 시각·Drive checkpoint/prediction checksum
- `selection-lock.json`의 validation 결과 digest와 생성 전 test 미열람 확인

값을 바꾸면 새 config·Drive experiment/run version을 만들고 RouterBench v2/v3 및 기존
ModernBERT run을 덮어쓰지 않는다. 운영 ModernBERT Colab 초기학습은 production config와
Drive checkpoint checksum을 기록하고, 이후 GCP feedback 재학습 run은 별도의 image digest,
Vertex project/region/service account와 GCS URI를 자신의 manifest에 기록한다.

### 9-2. 모델별 필수 산출물

- training config, tokenizer/model revision, input fingerprint
- best checkpoint와 validation observation 전부
- cross-fitted calibration metric과 최종 `calibration.json`(temperature, fit split hash)
- confusion matrix, class/source/category별 metric JSON
- wall time, peak RAM/VRAM, p50/p95 batch-1 latency, artifact size
- truncation count와 Qwen label likelihood/logit export
- seed별 prediction file과 paired-bootstrap 비교 결과
- threshold OOF prediction, 후보별 목적함수 항, T1/T2 선택 근거와 상호 라벨 불일치율
- pre/post-split `source × category × label` coverage report, systemic warning과 resolution
- ModernBERT `MB-RB-T1` 대 `MB-MIX-FULL-T1`의 공통 evaluation manifest와 paired 결과

### 9-3. 최종 결과표 형식

| 모델 | params | val Q_model | val source-macro-F1 | RB test macro-F1 | xRoute test macro-F1 | RA aligned macro-F1 | RA unseen macro-F1 | QWK | hard→easy | p95 ms | peak VRAM | artifact MB | train h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF+LR | - |  |  |  |  |  |  |  |  |  |  |  |  |
| MiniLM-L12-H384 | 33M |  |  |  |  |  |  |  |  |  |  |  |  |
| RoBERTa-base | ~125M |  |  |  |  |  |  |  |  |  |  |  |  |
| ModernBERT-base | 149M |  |  |  |  |  |  |  |  |  |  |  |  |
| DeBERTa-v3-base | ~184M |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen2.5-0.5B-Instruct | 0.49B |  |  |  |  |  |  |  |  |  |  |  |  |

주 순위표는 T1만 사용한다. 그 아래에 두 성능 보조표와 coverage 표를 분리한다.

| threshold 민감도 대상 | T1/T2 val macro-F1 | T1/T2 RB test macro-F1 | T1/T2 xRoute test macro-F1 | T1/T2 RA aligned macro-F1 | hard→easy 변화 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TF-IDF+LR |  |  |  |  |  |  |
| ModernBERT-base |  |  |  |  |  |  |

| ModernBERT 데이터 조건 | train rows | RB val/test macro-F1 | xRoute val/test macro-F1 | QWK | hard→easy | train h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MB-RB-T1` |  |  |  |  |  |  |
| `MB-MIX-FULL-T1` |  |  |  |  |  |  |

T2 표는 민감도만 판단하며 주 순위를 다시 정하지 않는다. ModernBERT 데이터 표는 충원에
따른 실용적 전후 차이를 보여 주며 순수 데이터 규모 효과라고 표기하지 않는다.

데이터 전처리 직후에는 다음 coverage 표를 먼저 채우고, `red` 행의 resolution이 모두
기록된 뒤 모델 성능표를 해석한다.

| source | split | category | total | easy n/% | medium n/% | hard n/% | coverage | resolution |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
|  |  |  |  |  |  |  |  |  |

category별 모델 metric에는 `confirmatory`, `exploratory`, `insufficient_coverage`를 함께
표시한다. 표본 부족 행은 순위 비교나 architecture 우열 근거에 포함하지 않는다.

결론은 하나의 최고 accuracy가 아니라 다음 세 문장으로 분리한다.

1. **동급 encoder 최고:** RoBERTa/ModernBERT/DeBERTa 중 품질 차이와 이유.
2. **production Pareto 권고:** 품질 허용 범위 안에서 가장 빠르고 작은 모델. 이번 자동
   재학습·승격 대상은 ModernBERT뿐이라는 운영 제약을 함께 표시.
3. **유형 비교 결론:** Qwen의 추가 계산량이 encoder 대비 유의한 품질 향상을 주는지.

모든 neural model이 TF-IDF를 통계적으로 넘지 못하면 “ModernBERT 대체 모델 없음”이
아니라 **질문 텍스트→현재 난이도 라벨 자체의 예측 가능성 부족**을 최종 결론으로 쓴다.

## 10. Colab 비교와 ModernBERT-only GCP 파이프라인

### 10-1. 실행 시스템을 세 경계로 분리

| 실행 시스템 | 대상 | 목적 | serving 변경 |
| --- | --- | --- | --- |
| Colab notebook suite | TF-IDF, MiniLM, RoBERTa, ModernBERT, DeBERTa, Qwen | 데이터·threshold 연구, 모델 학습·평가·학술 보고 | 없음 |
| Colab 운영 초기학습 | **ModernBERT만** | 동일 dataset SHA로 최초 serving checkpoint 학습·Drive 보존 | qualification 통과 후만 |
| GCP의 KFP v2/Vertex pipeline | **ModernBERT만** | feedback 재학습, 평가·gate·package·승격 | gate 통과 후만 |

다중 모델 비교용 `routing-data-build-v1` 또는 `router-model-comparison-v1` KFP pipeline은
만들지 않는다. 기존 `modernbert-router-initial-training` IR은 코드·compile regression용
역사적 artifact로만 남기고 제출하지 않는다. production GPU 실행은
`modernbert-router-feedback-retraining`만 사용한다. KFP component image, active compiled IR,
Vertex CustomJob config에는 MiniLM·RoBERTa·DeBERTa·Qwen 또는 초기학습 분기를 넣지 않는다.

### 10-2. Colab notebook workflow

```text
00_prepare_dataset_and_thresholds
  -> source schema/checksum 검증
  -> RAG/비텍스트 제거 + xRoute pivot
  -> exact/near duplicate group + split 봉인
  -> train-only 9개 threshold OOF + 후보별 train coverage
  -> T1/T2 선택 + 선택 후보의 pre/post-split coverage audit
  -> split-induced red면 새 split version과 threshold 선택부터 재실행
  -> 모든 red resolution 기록
  -> Drive에 immutable dataset/experiment manifest 게시

01_tfidf + 02_minilm + 03_roberta + 04_modernbert + 05_deberta + 06_qwen
  -> 각 notebook 독립 smoke/pilot
  -> T1 seed 42 train + validation output, 최대 2개 runtime 병렬
  -> TF-IDF/ModernBERT T2 seed 42 민감도
  -> ModernBERT MB-RB-T1 control과 MB-MIX-FULL-T1 비교
  -> T1 validation finalist만 seed 17/73
  -> Drive checkpoint와 공통 metrics/predictions/runtime schema

07_select_evaluate_and_report
  -> validation-only provisional winner + selection-lock
  -> 전 모델 RB/xRoute test + RA 3 slice 1회 평가
  -> paired bootstrap·원인 분석·Pareto report
  -> academic winner + production Pareto recommendation report
```

notebook은 orchestration service가 아니다. 사용자가 순서대로 실행하며 각 notebook은 선행
artifact의 checksum을 검증하고 이미 완료된 동일 run은 건너뛴다. Colab 세션 중단 복구는
Drive checkpoint가 담당하고, notebook 실행 이력·accelerator·runtime package version은
`run-manifest.json`에 남긴다. 비교 checkpoint를 GCS production 경로로 복사하지 않는다.

### 10-3. ModernBERT 초기학습과 GCP/KFP 재학습 workflow

```text
승인된 통합 dataset/config로 Colab ModernBERT 초기학습
  -> Drive checkpoint checksum + _SUCCESS
  -> package-v1 assemble/import + CPU qualification/gate
  -> models/router-modernbert candidate/approved 게시
  -> CURRENT.json 갱신과 FastAPI 최초 적재

이후 feedback-inline-v1 수신
  -> Vertex AI Pipelines에서 feedback-retraining KFP v2 IR 실행
  -> CPU feedback 검증·merge
  -> Vertex T4 CustomJob에서 ModernBERT만 재학습
  -> CPU 평가·deploy gate·package-v1 검증
  -> models/router-modernbert candidate/approved 게시
  -> CURRENT.json 갱신과 FastAPI load-before-swap
```

Colab의 `04_modernbert.ipynb` checkpoint도 연구 비교 결과일 뿐 운영 승격 입력이 아니다.
운영 최초 후보는 동일 dataset SHA와 고정 production config를 사용해
`modernbert_full_colab.ipynb`에서 별도 학습한다. 두 notebook의 목적과 run ID를 섞지 않는다.
운영 초기 checkpoint는 checksum·package·qualification gate를 통과한 뒤에만 게시하며,
Colab 비교 winner를 자동으로 `CURRENT.json`에 게시하는 연결은 존재하지 않는다.

### 10-4. 비교 결과와 운영 적용의 경계

| 유형 | Colab 비교 | Colab 운영 초기학습 | GCP KFP/Vertex 재학습·자동 승격 | 결과의 용도 |
| --- | :---: | :---: | :---: | --- |
| TF-IDF+LR | 포함 | 제외 | 제외 | 표층 기준선·CPU Pareto 참고 |
| MiniLM/RoBERTa/DeBERTa | 포함 | 제외 | 제외 | ModernBERT 대체 가능성에 대한 학술적 근거와 차기 권고 |
| ModernBERT | 포함 | **유일하게 포함** | **유일하게 포함** | 현행 기준선, 최초 serving checkpoint와 feedback 재학습 |
| Qwen 0.5B | 포함 | 제외 | 제외 | 생성형 유형 비교와 차기 권고 |

다른 모델이 학술적 winner이거나 Pareto 권고로 나와도 그 checkpoint를 `candidate/`,
`approved/`, `CURRENT.json`에 올리지 않는다. 보고서에는 `academic_winner`,
`pareto_recommendation`, `deployed_family=modernbert`를 분리 기록한다. 실제 family 전환은
후속 별도 결정·serving adapter·새 GCP pipeline 계약을 요구한다.

## 11. Colab Drive와 GCS artifact 보존 계약

### 11-1. 비교 실험용 Google Drive

다중 모델 비교 산출물은 GCS가 아니라 다음 연구용 Drive 경로에 둔다.

```text
MyDrive/LLM_Router/router-model-comparison/<experiment-id>/
  dataset/
    manifest.json
    contamination-report.json
    coverage/{coverage-report.parquet,summary.json,resolution.json}
    threshold/{candidates.parquet,metrics.json,selected-t1-t2.json}
    train/training.parquet
    validation/validation.parquet
    test/{routerbench.parquet,xroutebench.parquet}
    external/routerarena-{aligned,unseen,full}.parquet
  experiment-config.json
  runs/<model-id>/<threshold-id>/<data-variant>/<seed>/
    checkpoint/
    metrics.json
    predictions.parquet
    runtime.json
    run-manifest.json
  selection/{selection-lock.json,provisional-winner.json}
  reports/model-comparison.json
```

Drive에는 staging directory로 먼저 복사하고 checksum 검증 뒤 final name으로 rename한다.
최종 보고서와 작은 metric/prediction summary는 저장소의 `docs/ml/training-reports/`에
커밋할 수 있지만 model weight와 원문 포함 전체 Parquet는 Git에 커밋하지 않는다.

### 11-2. ModernBERT 운영용 GCS

정확한 bucket 이름은 환경 설정이 소유하고 Spring에는 노출하지 않는다. GCS에는 승인된
통합 dataset과 **ModernBERT pipeline에 필요한 artifact만** 둔다.

```text
gs://<artifact-bucket>/
  sources/{routerbench,xroutebench,routerarena}/<source-version>/source-manifest.json
  datasets/routing-integrated/<dataset-version>/
    manifest.json
    coverage/{summary.json,resolution.json}
    train/training.parquet
    validation/validation.parquet
    test/{routerbench.parquet,xroutebench.parquet}
    external/routerarena-{aligned,unseen,full}.parquet
  models/router-modernbert/
    <model-version>/candidate/
    <model-version>/approved/
    CURRENT.json
  datasets/continual-feedback/<candidate-version>/
    feedback/{events.parquet,manifest.json}
    replay/{events.parquet,manifest.json}
  retraining-runs/<batch-id>/observability-v1/
  retraining-control/validated-batches/
  pipeline-root/
```

Colab `00` 산출물을 운영 입력으로 채택할 때 dataset manifest와 파일별 SHA-256이 같은지
검증해 GCS의 새 immutable version으로 한 번 게시한다. GCS에는 비교 모델 checkpoint,
비교 notebook runtime 또는 `provisional-winner.json`을 올리지 않는다.

보존 원칙은 다음과 같다.

- Drive의 최종 비교 metric·prediction·report와 GCS의 승인 dataset/ModernBERT model은 서로
  다른 보존 경계다.
- Colab pilot·중간 checkpoint는 Drive 연구 경로에서 관리하고, GCP `pipeline-root`는
  ModernBERT KFP 임시 output과 package staging만 담아 1일 TTL 대상으로 삼는다.
- ModernBERT weight는 current, previous known-good, 현재 feedback cycle anchor를 보호하고
  나머지는 승격 뒤 정리한다. GCS soft delete를 rollback 수단으로 간주하지 않는다.
- 모든 directory artifact는 파일별 SHA-256 manifest와 `_SUCCESS`가 모두 생긴 뒤에만 읽는다.
- 질문 원문이 든 dataset/feedback Parquet에는 최소 권한을 적용하며 credential은 어느
  notebook·manifest·log에도 넣지 않는다.

## 12. Escalation과 continual retraining 전략

### 12-1. 책임 경계

```text
Spring
  feedback 대상 선택 + PII 제거 + feedback-inline-v1 POST
FastAPI
  schema/의미/중복 검증 + 정규화 + Parquet/manifest/replay + GCS + PipelineJob 제출
Vertex pipeline
  feedback prepare/merge + ModernBERT 재학습 + 평가/gate/package/publish
FastAPI watcher
  CURRENT 감지 + checksum/label/adapter 검증 + load-before-swap
```

Spring은 `manifestUri`, `eventsUri`, `eventCount`, GCS SDK와 재학습용 GCS IAM을 갖지 않는다.
FastAPI는 event 1건 이상을 받으며, 호출 시점·batch 크기는 Spring 정책이다. 같은
`batchId`와 같은 내용은 같은 run을 반환하고, 같은 ID의 다른 내용과 동시 pipeline은
`409`로 거부한다.

### 12-2. label 의미 차이와 편향 통제

초기 공개 데이터 label은 **후보 모델 집단의 solve rate**이고, escalation `finalLabel`은
**실제 서비스에서 사용자가 만족한 더 강한 tier**다. 둘은 관련 있지만 같은 측정값이
아니다. 또한 escalation은 under-routing 실패만 수집하므로 자연 traffic의 대표 표본도 아니다.

따라서 새 feedback adapter는 `label_origin=service_escalation`을 파생하고 기존
`router_model_version`을 보존하며 다음을 gate한다.

- `userSatisfied=true`, PII 제거, strict tier increase, label 충돌 없음
- base validation/test와 exact/near overlap인 feedback 제외
- source별 feedback count와 `easy->medium`, `easy->hard`, `medium->hard` 분포 보고
- 공개 base rehearsal을 항상 포함하고, feedback sample weight는 별도 config로 고정
- base validation의 macro-F1·hard recall·`hard->easy`가 허용 범위를 벗어나면 승격 거부

현행 `feedback_retraining_preprocessing.json`은 RouterBench v2-balanced와 ModernBERT에
고정되어 있으므로 통합 dataset과 threshold가 정해지면 새 config version을 만들어야 한다.
기존 config를 파일명만 유지한 채 내용만 바꾸지 않는다.

### 12-3. version cycle

기본 정책은 현재 구현된 3-call cycle을 유지한다.

```text
n.0 anchor + B1                       -> n.1 minor
n.0 anchor + B1+B2                    -> n.2 minor
n.0 anchor + B1+B2+B3                 -> (n+1).0 major consolidation
5번째 major마다 pristine ModernBERT + compact replay -> pristine rebase
```

각 단계는 gate까지 성공한 run만 count한다. 실패·거부·취소·동일 batch retry는 count하지
않는다. `successfulHttpRequest`와 `validatedUniqueBatch` policy 전환은 developer API에서
다음 major boundary에 예약하며, 과거 run을 소급 계산하지 않는다.

base ModernBERT revision은 model manifest에 고정한다. architecture 재평가는 매 escalation이
아니라 데이터 drift, 반복 gate 실패 또는 정기 major checkpoint에서 새 Colab
`router-model-comparison` experiment로 수행한다. 결과가 달라져도 GCP continual pipeline은
ModernBERT만 재학습하며, family 전환은 별도 후속 결정으로 남긴다.

## 13. 수정해야 할 코드와 구현 순서

현재 코드를 기준으로 한 상태다. `구현됨`은 로컬 단위·계약 코드가 있다는 뜻이며,
`cloud 미검증`은 실제 Vertex/GCS end-to-end 완료를 뜻하지 않는다.

### 13-1. `LLM_Router` ML 저장소

| 우선순위 | 대상 | 현재 | 필요한 변경·완료 조건 |
| --- | --- | --- | --- |
| P0 | 새 `routing_data` preprocessing package/CLI | 미구현 | Colab `00`에서 호출할 xRoute pivot, metric mapping, RAG filter, cross-source dedup, split, 9개 threshold OOF, T1/T2 선택, pre/post-split category coverage audit와 resolution manifest 구현 |
| P0 | 공통 evaluator | 부분 구현 | Colab `07`이 호출할 source macro-F1, BA, QWK, ordinal MAE, hard→easy, ECE, prediction export와 category cell count 기반 confirmatory/exploratory/insufficient 판정을 통합 |
| P0 | Colab 공통 training helper | ModernBERT 전용 | MiniLM·RoBERTa·ModernBERT·DeBERTa가 공유할 seed/exposure/checkpoint/output 계약 구현; notebook별 checkpoint는 Drive에 저장 |
| P0 | TF-IDF baseline | 미구현 | 고정 config, group OOF threshold probe, validation/test prediction과 runtime export |
| P0 | Qwen label-likelihood adapter | 미구현 | causal-LM 학습, 세 label conditional likelihood, confidence/ECE, OOM 명시 처리 |
| P0 | Colab notebook suite | 미구현 | `00~07` notebook, coverage report·red resolution gate, T1 주 비교, TF-IDF/ModernBERT T2 민감도, `MB-RB-T1` control, 공통 bootstrap/runtime lock, Drive 원자 checkpoint·resume, selection-lock 구현 |
| P0 | ModernBERT 운영 초기학습 Colab profile | 기존 v2 notebook 구현 | `modernbert_full_colab.ipynb`가 통합 dataset/config를 검증하고 Drive checkpoint·resume·최종 `_SUCCESS`와 package import용 manifest를 생성 |
| P1 | ModernBERT 재학습 Vertex artifact 전달 | 부분 구현 | feedback 재학습 CustomJob에만 immutable `gs://` 입력/출력, `_SUCCESS`, checksum, resume state와 MLMD lineage 연결 |
| P1 | 비교 artifact schema | 미구현 | Drive 연구 checkpoint에 model family/revision, config, latency, prediction checksum을 남기되 GCS production package로 게시하지 않음 |
| P1 | ModernBERT qualification/gate | RouterArena 단일 평가 | 통합 validation safety gate + RA aligned veto + CPU latency/resource gate, `calibration.json` checksum을 ModernBERT 승격에 적용 |
| P2 | notebook 재현성 | 기존 ModernBERT Colab notebook만 존재 | 각 notebook이 동일 repo commit/runtime lock을 설치하고 공통 helper를 호출하며 실행 순서·artifact digest를 기록 |

먼저 P0의 data contract와 evaluator를 끝내야 모델 성능 숫자가 서로 비교 가능하다. 다른
family의 checkpoint는 Drive 연구 artifact로만 저장하고 KFP·Vertex·ModernBERT
package/publish code에 연결하지 않는다.

### 13-2. `LLM_Router-serving` FastAPI 저장소

| 우선순위 | 대상 파일·영역 | 현재 | 필요한 변경·완료 조건 |
| --- | --- | --- | --- |
| 유지 | `inference.py`, `main.py` | ModernBERT sequence classifier 적재 | 이번 범위에서는 다른 비교 모델 adapter를 추가하지 않음 |
| 유지 | `model_source.py` | ModernBERT package v1 label/checksum 검증 구현됨 | 통합 ModernBERT package의 새 dataset/label metadata fixture를 추가하고 v1 계약 유지 |
| 유지 | `config.py` router max length | ModernBERT 8,192 | 비교 모델별 max length를 serving 설정에 추가하지 않음 |
| P0 | classify response·calibration | tier/confidence/version와 Claude hint, raw softmax 구현됨 | API shape은 유지; 통합 ModernBERT package의 `calibration.json` temperature를 logit에 적용하고 legacy package는 raw-softmax fallback을 명시적으로 표시 |
| 유지 | model manager/watcher | checksum 검증 후 ModernBERT load-before-swap 구현됨 | 통합 ModernBERT package로 swap 실패/rollback 회귀 test 실행 |
| P0 | retraining config | RouterBench v2-balanced base URI·config | 통합 dataset URI, 선택 threshold/label version, ModernBERT training/package metadata의 일관성 검증 |
| P1 | `/escalate`, 상태 API | inline artifact·Vertex 제출·상태/current/policy 구현됨 | 새 feedback config와 통합 base lineage를 전달; 기존 멱등성·3-call 계약 유지 |
| P1 | cloud integration | fake GCS/Vertex test 통과 | 실제 inline 1행→Vertex ModernBERT→CURRENT→activation E2E와 rollback 검증 |
| P2 | 문서/OpenAPI/README | endpoint 계약 구현됨 | ModernBERT-only 재학습 경계와 최신 구현 상태 반영, YAML/JSON 동등성 contract test 실행 |

온라인 API가 비교 architecture를 노출할 필요는 없다. FastAPI는 ModernBERT만 적재하고,
운영 추적은 immutable `modelVersion`과 package manifest가 담당한다. 따라서 비교 실험
때문에 새 `/router/*` endpoint나 다른 model loader를 만들지 않는다.

### 13-3. 구현 순서와 완료 gate

1. data build + threshold selector + category coverage audit + 공통 evaluator를 구현하고
   green/yellow/red와 split-induced red를 포함한 작은 fixture로 재현한다.
2. `00~07` Colab notebook과 공통 helper, TF-IDF/네 encoder/Qwen adapter를 연결한다.
3. `00`의 pre/post-split coverage report와 모든 red resolution을 봉인하고, split-induced
   red면 새 split version에서 T1/T2 선택부터 재실행한다.
4. 각 notebook의 100행 Colab smoke와 Drive checkpoint 저장·새 세션 resume를 통과한다.
5. 모델별 200-step pilot 후 최대 두 runtime으로 T1 seed 42 full comparison과
   `MB-RB-T1` control을 실행한다.
6. TF-IDF/ModernBERT T2 seed 42 민감도와 필요 시 T1 상위 2개의 T2 확장을 실행한 뒤,
   T1 finalist에만 seed 17/73을 추가한다.
7. `selection-lock.json`, academic winner와 Pareto 권고를 T1 기준으로 동결한다.
8. 동일 dataset SHA와 production config로 `modernbert_full_colab.ipynb` 운영 초기학습을
   완료하고 최종 Drive checkpoint checksum·`_SUCCESS`·resume 증거를 동결한다.
9. 기존 package-v1로 Colab ModernBERT checkpoint import/qualification smoke → GCS candidate/approved →
   `CURRENT.json` hot-swap을 검증한다.
10. ModernBERT feedback-retraining KFP v2 IR compile test와 Vertex 100행 재학습 cloud smoke를
   통과한다.
11. 마지막으로 inline feedback 1행 E2E를 실행해 prepare부터 activation까지 확인한다.

각 단계가 실패하면 그 단계 이전 production ModernBERT와 `CURRENT.json`은 바꾸지 않는다.

## 14. FastAPI API 문서

정확한 request/response/error schema의 단일 기준은
`LLM_Router-serving/docs/api-contract.md`와 같은 디렉터리의
`backend-api-v1/openapi.yaml`·`openapi.json`이다. 이 절은 실험·파이프라인 관점의 색인이다.

### 14-1. 현행 endpoint와 구현 상태

| Method | Path | 인증 | 성공 | 의미 | 상태 |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/router/embed` | internal bearer | `200` | BGE 1,024차원 L2-normalized embedding | 구현됨 |
| `POST` | `/router/classify` | internal bearer | `200` | `easy/medium/hard`, confidence, model hint/version | 구현됨 |
| `GET` | `/health` | 없음 | `200`/`503` | BGE와 router 모두 적재됐을 때만 ready | 구현됨 |
| `POST` | `/escalate` | internal bearer | `202` | inline feedback artifact화와 Vertex PipelineJob 제출 | 구현됨 |
| `GET` | `/retraining-runs/{batchId}` | internal bearer | `200` | 단계·필터링·gate 결과·activation 조회 | 구현됨 |
| `GET` | `/router/model/current` | internal bearer | `200`/`503` | desired/serving/known-good와 swap 상태 | 구현됨 |
| `GET/PUT` | `/developer/retraining/version-policy` | developer bearer+feature flag | `200` | 3-call count mode 조회·다음 major 경계 변경 예약 | 구현됨 |

### 14-2. 핵심 payload 경계

`POST /router/classify`는 현재 ModernBERT serving 계약으로 유지한다. 다른 다섯 비교 모델은
offline evaluator만 사용하고 이 endpoint에 적재하지 않는다.

```json
{
  "message": "Explain why the sky appears blue."
}
```

```json
{
  "tier": "medium",
  "model": "sonnet",
  "confidence": 0.87,
  "modelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d"
}
```

`model`은 기존 client 호환용 tier hint다. 실제 Claude API model ID와 mapping version은
Spring이 소유한다. `confidence`는 ModernBERT validation에서 검증된 calibrated probability여야
하며, offline 비교 모델의 raw logit이나 Qwen 생성 확률을 이 API 값으로 보내지 않는다.

`POST /escalate`는 `feedback-inline-v1`의 `batchId`, `schemaVersion`, `requestedAt`,
`events[]`만 받는다. `events.length >= 1`이며 각 event는 strict tier increase,
`userSatisfied=true`, `piiRedacted=true`, UTC timestamp, 고유 `eventId`를 만족해야 한다.
`manifestUri`, `eventsUri`, `eventCount`는 보내면 `422`다.

`202 Accepted`는 GCS artifact 작성과 PipelineJob 제출 성공일 뿐 학습·승격 성공이 아니다.
Spring은 응답의 `statusUrl`을 polling하고 `result.outcome=promoted`와
`activation.status=active`를 별도로 확인한다.

### 14-3. 오류·fallback와 변경 규칙

- `/router/embed` 또는 `/router/classify`의 모델이 미적재면 `503`; `/health`도 두 모델 중
  하나가 미적재면 `503`이다.
- FastAPI timeout/503 시 어느 tier로 fallback할지는 Spring의 versioned routing policy가
  소유한다. FastAPI가 몰래 `easy`를 반환하지 않는다.
- 모든 오류는 `error`, `code`, `requestId`를 반환하고 Spring은 `code`로 분기한다.
- 이번 비교 실험은 HTTP API 변경을 만들지 않는다. 향후 별도 승인으로 model family를
  전환하더라도 필드·enum·의미가 유지되면 API v1을 유지할 수 있다.
- production ModernBERT package-v1 metadata 변경을 Spring payload에 전파하지 않는다.

## 15. 실행 승인 기준

이 문서를 “실행 가능”으로 간주하려면 순서대로 다음 증거가 필요하다.

1. 통합 preprocessing/T1·T2 threshold fixture, category coverage fixture와 deterministic checksum
2. pre/post-split coverage report, systemic warning 여부와 모든 red resolution
3. `00~07` notebook bootstrap과 여섯 비교군 100행 Colab smoke·공통 prediction schema
4. Drive checkpoint checksum·원자 저장·새 Colab 세션 resume 증거
5. T1 full Colab comparison, T2 민감도, `MB-RB-T1` 대 `MB-MIX-FULL-T1` 비교 보고서와
   academic provisional winner freeze
6. 통합 dataset으로 실행한 Colab ModernBERT 운영 초기학습의 최종 checkpoint·resume 증거
7. Colab checkpoint→package-v1→qualification→candidate/approved→CURRENT→FastAPI 최초 적재
8. **ModernBERT feedback 재학습에 한해서** KFP compile 및 Vertex
   PipelineJob→T4 CustomJob→GCS checkpoint→CPU evaluation 100행 cloud lineage
9. FastAPI activation/rollback E2E
10. Spring inline feedback→3-call lineage 중 최소 1회 실제 E2E

현재 시점에는 로컬 ModernBERT 학습·평가·승격 코드와 FastAPI API가 상당 부분 구현돼 있지만,
**통합 RouterBench+xRouteBench data builder, T1/T2 threshold selector, category coverage audit,
`00~07` Colab 비교 notebook,
공통 comparison helper, Qwen adapter, 통합-data ModernBERT Colab 초기학습과 Vertex feedback
재학습 cloud 실행은 아직 미구현 또는
미검증**이다. 다중 모델 비교 KFP pipeline은 미구현 과제가 아니라 만들지 않기로 확정했다.
FastAPI의 다른 model-family loader는 이번 범위에 포함하지 않는다. 따라서 이 문서는
실행 설계를 확정한 것이며, 새 비교 결과가 이미 존재한다고 해석하지 않는다.
