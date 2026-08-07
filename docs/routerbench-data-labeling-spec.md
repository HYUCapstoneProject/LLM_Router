# RouterBench Data Labeling Specification and Checklist

기준일: 2026-08-08
상위 문서: `docs/mvp-scope-and-checklist.md`
작업 브랜치: `feat/data-labeling-training`

## 1. 목표와 완료 정의

RouterBench 질문을 **만족스러운 답을 만들 수 있는 최소 모델 성능 티어**에 따라 `easy`, `medium`, `hard`로 재라벨링한다. 라벨 산출물은 ModernBERT 학습에서 바로 읽을 수 있는 Parquet과 검증용 JSON으로 만든다.

데이터 기준선 작업은 다음 조건을 모두 만족하면 완료한다.

- 공식 RouterBench 0-shot/5-shot 파일을 checksum으로 검증해 읽을 수 있다.
- 모델 capability tier를 데이터에서 재현 가능하게 산정한다.
- 질문별 tier success rate와 3-class label을 생성한다.
- 고정 seed로 약 100개 smoke sample을 생성한다.
- smoke 산출물의 행 수, 중복, 스키마, 클래스 분포를 자동 검증한다.
- smoke와 같은 판정 코드로 전체 핵심 eval을 라벨링하고 versioned artifact를 만든다.
- 전체 라벨에 중복 그룹, 신뢰도 가중치, group-aware split 전처리를 적용한다.
- 로컬 실행 명령과 산출 위치가 문서화되어 있다.

100개 smoke test는 정확도 결론을 내리는 실험이 아니라 데이터 계약과 전체 실행 경로를 확인하는 테스트다. 라벨 품질 판단과 임계값 확정은 전체 데이터 분포 및 Claude 표본 검증 단계에서 수행한다.

Smoke 표본은 eval별 최소 1행을 먼저 선택하고, 여유 행으로 가능한 각 도메인이 최소 10%를 차지하도록 보충한 뒤 나머지를 고정 seed로 채운다. 이는 원본 분포 추정용 표본이 아니라 여러 전처리·판정 경로를 실행하기 위한 coverage 표본이다.

## 2. 입력 데이터 계약

### 공식 파일

| 용도 | 파일 | 예상 크기 | SHA-256 |
| --- | --- | ---: | --- |
| 0-shot | `routerbench_0shot.pkl` | 99,567,659 bytes | `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d` |
| 5-shot | `routerbench_5shot.pkl` | 171,091,324 bytes | `fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe` |

공식 배포본은 Pandas Pickle이다. Pickle은 역직렬화 시 코드를 실행할 수 있으므로 다음 조건을 모두 만족할 때만 읽는다.

1. 다운로드 URL이 설정 파일에 고정되어 있다.
2. 파일 크기와 SHA-256이 일치한다.
3. 실행자가 `--trust-pickle` 옵션으로 신뢰를 명시한다.
4. 읽은 원본은 수정하지 않고 이후 산출물은 Parquet으로 저장한다.

### 예상 wide schema

```text
sample_id
prompt
eval_name
<model-name>
<model-name>|model_response
<model-name>|total_cost
oracle_model_to_route_to
```

`<model-name>` 컬럼은 질문별 모델 성능 점수다. 모델 컬럼은 접미사 규칙과 numeric dtype을 함께 검사해 식별한다.

## 3. Canonicalization

- MMLU, HellaSwag, GSM8K, ARC-Challenge, WinoGrande는 5-shot row를 우선한다.
- MBPP, MT-Bench, RAG는 0-shot row를 우선한다.
- 논문의 핵심 8개 계열에 매핑되지 않은 보조 eval은 초기 학습 범위에서 제외하고 이름과 행 수를 metric에 남긴다.
- 최종 행의 유일 키는 `sample_id`다.
- 0-shot/5-shot에 같은 `sample_id`가 있으면 설정된 source 우선순위로 한 행만 남긴다.
- `prompt`가 문자열로 직렬화된 list이면 안전한 literal parser로 복원한다.
- query extractor는 5-shot의 마지막 미응답 `Answer:` 블록을 기준으로 대상 질문만 추출하며, 원문과 추출문을 각각 `prompt`와 `query_text`로 남긴다.

RAG는 난이도와 검색 필요성을 혼동할 수 있으므로 `needs_retrieval`을 별도 필드로 기록한다.

## 4. 모델 capability tier

모델 가격이나 파라미터 수가 아니라 관측 성능으로 tier를 정한다.

1. `eval_name × model` 평균 performance를 계산한다.
2. eval별 모델 performance percentile rank를 계산해 평가 척도와 태스크 크기 차이를 줄인다.
3. 모델별 percentile rank의 macro average를 capability score로 사용한다.
4. capability score 오름차순으로 모델을 약 1/3씩 `low`, `medium`, `high`로 나눈다.
5. capability score가 같으면 평균 비용이 낮은 모델을 먼저 배치한다.

산정 결과는 다음 필드를 가진 `model_tiers.json`으로 고정한다.

```text
tier_config_version
dataset_checksums
ranking_metric
models.low
models.medium
models.high
model_statistics
```

## 5. 성공 판정과 라벨 규칙

### 모델 응답 성공

- exact-match 계열: `performance == 1.0`
- judge/연속 점수 계열: `performance >= acceptance_threshold[dataset]`
- 초기 continuous 기본 임계값: `0.70`
- 결측값: 실패가 아니라 해당 tier의 유효 모델 수에서 제외

```text
tier_success(q, t)
  = t에서 성공한 모델 수 / t에서 유효한 모델 수
```

### 3-class 판정

```python
if low_success_rate >= 0.67:
    label = "easy"
    reason = "low_tier_consensus"
elif medium_success_rate >= 0.67:
    label = "medium"
    reason = "medium_tier_consensus"
elif high_success_rate >= 0.50:
    label = "hard"
    reason = "strong_only"
else:
    label = "hard"
    reason = "unsolved"
```

low와 medium의 합의 임계값을 높게 두어 under-routing을 보수적으로 제한한다. `strong_only`와 `unsolved`는 운영 라벨은 같지만 분석 필드는 분리한다.

### 신뢰도

- `high`: 라벨을 결정한 tier에 유효 모델이 2개 이상이고 비단조/결측 이상이 없다.
- `medium`: 라벨은 결정 가능하지만 일부 결측 또는 tier 성공률 비단조가 있다.
- `low`: 유효 모델 부족, 단일 모델 성공, 기타 모순으로 안정적인 판정이 어렵다.

## 6. 출력 데이터 계약

```text
sample_id: string
query_text: string
prompt: string
eval_name: string
domain: string
shot_source: string
low_success_rate: float
medium_success_rate: float
high_success_rate: float
overall_solve_rate: float
cheapest_correct_tier: string | null
difficulty_label: easy | medium | hard
label_reason: string
label_confidence: high | medium | low
label_weight: float
needs_retrieval: bool
dataset_version: string
tier_config_version: string
label_rule_version: string
```

산출물:

```text
ml/artifacts/routerbench-smoke-100/labeled.parquet
ml/artifacts/routerbench-smoke-100/metrics.json
ml/artifacts/routerbench-smoke-100/model_tiers.json

ml/artifacts/routerbench-labeled/
  <dataset-version>__<tier-version>__<label-rule-version>/
    labeled.parquet
    metrics.json
    model_tiers.json
```

## 7. 학습 전처리 계약

라벨 산출물은 분석 가능한 원본에 가깝게 유지하고, ModernBERT 학습에 직접 넣는
projection은 별도의 결정적 전처리 단계에서 생성한다. 전처리 설정이 바뀌면
`preprocessing_version`을 올리고 기존 split을 덮어쓰지 않는다.

### 7-1. 학습 텍스트 정제

- `query_text`는 감사와 재처리를 위해 변경하지 않는다.
- Unicode는 NFKC로 정규화하고 제어 문자, 행 끝 공백, 연속 빈 줄을 정리한다.
- MMLU 계열의 `Please answer with ...`, `Print only a single choice ...`처럼
  RouterBench 평가 형식에만 필요한 앞뒤 지시문을 제거해 `model_input_text`를 만든다.
- 문제 본문, 선택지, 코드, MT-Bench의 선행 대화 문맥은 유지한다.
- 정제 후 빈 문자열이 되면 조용히 제외하지 않고 전처리를 실패시킨다.
- 문자·단어 길이는 metadata에 기록하되, tokenizer 기반 truncation은 tokenizer가
  확정된 학습 컴포넌트에서 수행한다. 긴 입력은 질문이나 선택지 보존 여부를 별도
  검증한다.

### 7-2. 중복 및 split 누수 방지

1. 정제 텍스트를 case-folding하고 단어 단위로 정규화해 exact duplicate를 묶는다.
2. token 3-gram SimHash로 근접 후보를 찾고 Jaccard 기준으로 near duplicate를 묶는다.
3. 동일·유사 질문 cluster에 안정적인 `duplicate_group_id`를 부여한다.
4. `difficulty_label × domain` 분포를 고려하되 group 전체를 하나의
   train/validation/test split에만 배치한다.
5. seed, split 비율, 유사도 임계값을 설정과 metric에 남긴다.

초기 split은 `80/10/10`, seed는 `42`, near-duplicate Jaccard 임계값은 `0.90`이다.
validation/test에는 oversampling이나 class balancing을 적용하지 않는다.

### 7-3. 신뢰도와 학습 표본 정책

| label confidence | 초기 sample weight | 처리 |
| --- | ---: | --- |
| `high` | 1.0 | 포함 |
| `medium` | 0.5 | 포함하되 가중치 축소 |
| `low` | 0.0 | 학습 projection에서 제외, metadata에는 보존 |

`hard/strong_only`와 `hard/unsolved`는 모두 hard로 유지하지만 metadata와 평가에서는
분리한다. `unsolved`의 추가 감쇠 또는 제외는 전체 분포와 실제 Claude 검증 후 별도
실험 설정으로 결정한다. RAG 포함 여부도 `include_retrieval` 설정으로 고정한다.

### 7-4. 학습 projection과 분석 metadata 분리

모델 성능률과 cheapest-correct 정보는 라벨을 직접 노출하므로 모델 입력으로
전달하지 않는다.

```text
training.parquet
  sample_id
  model_input_text
  difficulty_label
  sample_weight
  split
  duplicate_group_id

metadata.parquet
  query_text, eval_name, domain, label_reason, label_confidence
  tier/overall success rate, cheapest-correct tier
  needs_retrieval, shot_source, text length, exclusion reason
  dataset/label/tier/preprocessing versions
```

trainer는 `model_input_text`만 feature로, `difficulty_label`만 target으로,
`sample_weight`만 loss weight로 읽는다. success rate, model name, model response,
cost는 training projection에 존재해서는 안 된다.

### 7-5. 코드와 Kubeflow의 책임 경계

Python 패키지는 로컬과 GKE에서 동일하게 실행할 수 있는 데이터 의미론을 담당한다.

- 텍스트 정제와 검증
- 중복 cluster와 group-aware split 산정
- 신뢰도 필터와 sample weight 적용
- 학습/metadata projection 및 metric 생성

Kubeflow Pipeline은 이 로직을 다시 구현하지 않고 독립 컴포넌트를 조립한다.

- 원본 및 설정 Artifact 전달, ML Metadata 계보 기록
- 입력 checksum과 parameter 기반 step caching
- 라벨링 CPU component와 전처리 CPU component 분리
- ModernBERT tokenizer/train GPU component 분리
- retry, resource request, GCS output 경로와 버전 관리
- threshold/split 정책 비교 시 parameterized run 또는 병렬 실험 구성

split은 전처리 component가 한 번 생성한 versioned artifact를 downstream에서
재사용한다. 학습 Pod가 실행될 때마다 split을 다시 계산하지 않는다.

전처리 산출물:

```text
ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet
ml/artifacts/routerbench-preprocessed-smoke-100/metadata.parquet
ml/artifacts/routerbench-preprocessed-smoke-100/metrics.json

ml/artifacts/routerbench-training/
  <dataset-version>__<label-rule-version>__<preprocessing-version>/
    training.parquet
    metadata.parquet
    metrics.json
```

## 8. Smoke test 승인 기준

- 정확히 100행이 생성된다.
- `sample_id` 중복이 없다.
- 필수 컬럼과 dtype 검증을 통과한다.
- 모든 success rate가 `[0, 1]` 또는 null이다.
- 모든 label이 `easy`, `medium`, `hard` 중 하나다.
- 모든 모델이 정확히 하나의 capability tier에 포함된다.
- 동일 seed로 실행하면 같은 `sample_id`와 라벨이 생성된다.
- 클래스, eval, confidence, reason 분포가 `metrics.json`에 기록된다.
- 원본 Pickle과 생성 artifact는 Git 추적 대상이 아니다.

## 9. 실행 체크리스트

### A. 명세와 안전성

- [x] 라벨 의미와 3-class 판정식을 문서화한다.
- [x] 공식 파일 URL, 크기, checksum을 고정한다.
- [x] Pickle 신뢰 경계를 명시한다.
- [x] smoke 승인 기준을 정한다.

### B. 구현

- [x] Python ML 패키지와 의존성 파일을 추가한다.
- [x] checksum 검증 및 중단 재개 downloader를 구현한다.
- [x] wide schema/model column detector를 구현한다.
- [x] 0-shot/5-shot canonicalizer와 target query extractor를 구현한다.
- [x] capability tier calibration을 구현한다.
- [x] tier-consensus labeler와 confidence 산정을 구현한다.
- [x] Parquet/JSON writer를 구현한다.
- [x] synthetic unit test를 작성한다.
- [x] benchmark wrapper 정제와 `model_input_text` 생성을 구현한다.
- [x] exact/near duplicate grouping과 group-aware split을 구현한다.
- [x] confidence weight/filter와 학습 projection 분리를 구현한다.
- [x] 전처리 CLI와 synthetic unit test를 작성한다.

### C. 100행 smoke

- [x] 공식 0-shot/5-shot 파일을 다운로드하고 checksum을 확인한다.
- [x] 실제 eval/model/schema 분포를 확인한다.
- [x] 고정 seed로 100행을 라벨링한다.
- [x] smoke 승인 검사를 통과한다.
- [x] 분포와 이상치를 검토하고 이 문서에 결과를 기록한다.

### D. 전체 데이터와 다음 단계

- [x] smoke 전용이 아닌 전체 데이터 라벨링 CLI와 versioned 출력 경로를 추가한다.
- [x] 전체 데이터의 현재 threshold 기준 라벨 분포를 계산하고 검토한다.
- [ ] continuous threshold sensitivity를 계산한다.
- [x] domain/label을 고려한 train/validation/test group split을 구현한다.
- [x] 전체 라벨 산출물에 전처리를 실행하고 split 분포를 승인한다.
- [ ] eval별 acceptance threshold override와 최종 threshold를 확정한다.
- [ ] `model_tiers.json`에 생성 코드 commit을 기록한다.
- [ ] 전체 split의 크기·라벨·도메인 편차에 대한 자동 승인 기준을 추가한다.
- [ ] 확정 tokenizer로 길이/truncation 통계와 긴 입력 보존 검사를 실행한다.
- [ ] ModernBERT 로컬 training smoke test를 수행한다.
- [ ] Claude Haiku/Sonnet/Opus 표본 검증으로 프록시 라벨을 보정한다.
- [ ] 동일 config로 GCS/Kubeflow/GKE 학습 파이프라인을 연결한다.

## 10. 미확정 사항

- continuous performance의 최종 acceptance threshold
- RAG를 초기 3-class 학습에 포함할지 여부
- capability tier의 모델 수를 균등 분할할지 Pareto frontier로 보정할지 여부
- `medium_confidence` 행의 최종 sample weight

이 값들은 smoke test만으로 확정하지 않고 전체 분포와 Claude 검증 결과를 근거로 버전업한다.

## 11. 100행 labeling smoke 결과

실행 설정:

```text
dataset_version: withmartian-routerbench-2024-03-27
label_rule_version: minimum-sufficient-tier-v1
tier_config_version: routerbench-capability-tier-v1
sample_size: 100
seed: 42
sampling: eval별 최소 1행 + 가능한 도메인별 최소 10행 + seeded fill
```

입력 및 범위 확인:

| 항목 | 결과 |
| --- | ---: |
| 공식 배포본 canonical 질문 | 36,497 |
| MVP 핵심 데이터셋 범위 | 34,778 |
| 범위 밖 보조 eval 행 | 1,719 |
| 핵심 범위 eval_name | 65 |
| 모델 | 11 |

공식 Pickle의 모델 performance 컬럼은 값은 숫자지만 dtype이 `object`였다. 따라서 컬럼 이름에 대응하는 `|model_response` 또는 `|total_cost` sibling이 있고, 모든 non-null 값을 숫자로 변환할 수 있을 때만 모델 performance 컬럼으로 인정하도록 detector를 보강했다.

논문이 설명한 핵심 범위 외에 중국어 태스크, `abstract2title`, `accounting_audit`, `bias_detection`, `consensus_summary`, `test-match` 등 21개 보조 eval 이름이 배포본에 포함되어 있었다. 초기 MVP 라벨링에서는 이를 제외하고 metric에 이름과 행 수를 보존한다.

산정된 capability tier:

| 티어 | 모델 |
| --- | --- |
| low | Mistral-7B, Claude Instant v1, Code Llama 34B, WizardLM-13B |
| medium | Claude v1, Llama-2-70B, Claude v2, GPT-3.5 Turbo |
| high | Mixtral-8x7B, Yi-34B, GPT-4 |

100행 결과:

| 항목 | 분포 |
| --- | --- |
| label | easy 42, medium 15, hard 43 |
| reason | low consensus 42, medium consensus 15, strong only 13, unsolved 30 |
| confidence | high 64, medium 36, low 0 |
| domain | knowledge 59, commonsense 11, math 10, code 10, conversation 10 |
| shot source | 5-shot 80, 0-shot 20 |

검증 결과:

- 100행, 필수 스키마, success rate 범위, 유효 label 검사를 통과했다.
- `sample_id` 중복, `query_text` 중복, 빈 `query_text`는 각각 0건이었다.
- 동일 seed로 독립 실행한 두 Parquet의 `sample_id`, query, label, reason, confidence, success rate와 metric JSON이 일치했다.
- 5-shot 예시 제거 후 smoke query 길이는 평균 529.99자였고, 0-shot은 평균 242.00자였다.
- MT-Bench 다중 턴 질문은 후속 질문만 남기지 않고 선행 문맥과 함께 보존했다.
- 원본 Pickle, 가상환경, smoke artifact가 `.gitignore` 적용 대상임을 확인했다.

Smoke 결과는 파이프라인 실행 가능성을 확인했지만 threshold와 tier 경계를 확정하지는 않는다. 특히 GPT-3.5와 Mixtral이 medium/high 경계에 있으므로 전체 데이터 sensitivity 분석에서 경계 안정성을 확인해야 한다.

## 12. 100행 preprocessing smoke 결과

`routerbench-training-preprocess-v1`, seed `42`, split `80/10/10`으로 labeling smoke
산출물 100행을 전처리했다.

| 항목 | 결과 |
| --- | ---: |
| 학습 포함 | 100 |
| 제외 | 0 |
| benchmark wrapper 제거 | 70 |
| duplicate group | 100 |
| exact duplicate row | 0 |
| 승인된 near-duplicate pair | 0 |
| train / validation / test | 80 / 10 / 10 |
| 빈 model input | 0 |
| 한 group이 차지한 split 최대 수 | 1 |

초기 실행에서는 GSM8K의 서로 다른 10개 sample이 같은 질문으로 추출되는 이상이
발견됐다. 공식 5-shot prompt가 `[demonstration block, target question]` 형태인데 기존
extractor가 첫 list item의 마지막 demonstration을 target으로 오인한 것이 원인이었다.
별도 list item의 demonstration block 다음 항목을 target으로 선택하도록 extractor와
회귀 테스트를 수정한 뒤 labeling과 preprocessing smoke를 다시 생성했다. 수정 후
100개 `query_text`와 duplicate group은 모두 유일했다.

`training.parquet`은 계약된 6개 컬럼만 포함하며 success rate, cost, model 정보가 없는
것을 확인했다. `metadata.parquet`은 원문, label 근거, tier 성공률, 텍스트 길이와 버전을
보존한다.

## 13. 전체 데이터 labeling 및 preprocessing 결과

2026-08-08에 smoke와 같은 `minimum-sufficient-tier-v1` 규칙으로 MVP 핵심 범위
34,778행 전체를 실행했다. `label`은 범위 전체를 `sample_id` 순으로 결정적으로
처리하며, `preprocess --profile full`은 config와 라벨 산출물의 dataset/tier/rule
version이 정확히 일치할 때만 진행한다.

전체 labeling 결과:

| 항목 | 결과 |
| --- | ---: |
| 입력 / 라벨 완료 | 34,778 / 34,778 |
| easy / medium / hard | 12,765 / 5,348 / 16,665 |
| high / medium / low confidence | 23,941 / 10,823 / 14 |
| low / medium consensus | 12,765 / 5,348 |
| strong only / unsolved | 4,894 / 11,771 |
| 중복 sample_id / 빈 query | 0 / 0 |
| 중복 query text | 7 |

전체 preprocessing 결과:

| 항목 | 결과 |
| --- | ---: |
| 학습 포함 / 제외 | 34,764 / 14 |
| 제외 사유 | low confidence weight 0: 14 |
| train / validation / test | 27,808 / 3,478 / 3,478 |
| duplicate group / duplicate row | 34,769 / 9 |
| near-duplicate 승인 pair | 2 |
| benchmark wrapper 제거 | 26,821 |
| model input 단어 수 min / median / max | 7 / 58 / 797 |

split은 학습 포함 행 기준 정확히 80/10/10이며, 각 duplicate group은 하나의 split에만
속한다. RouterArena full도 이 전체 `training.parquet`을 기준으로 다시 전처리했고,
정규화 텍스트가 정확히 겹치는 22행을 평가에서 제외했다. 따라서 RouterArena full
평가 가능 행은 8,375행이다.
