# LLM Router MVP Scope — Historical Baseline

기준일: 2026-07-08

이 문서는 2026-07-08 당시 stub 상태를 기준으로 MVP와 최종 목표를 구분한 과거 범위
초안이다. 현행 ML 구조, 백엔드 협의 사항과 작업 상태는
`docs/ml/workstream-checklist.md`를 유일한 기준으로 사용한다.
상세 일정은 `docs/timeline.md`, 배경 설계는 `docs/졸업프로젝트 개요.md`를 기준으로 한다.

## 1. 현재 stub 목록

| 영역 | 파일 | 현재 상태 | 구현 방향 |
| --- | --- | --- | --- |
| Router | `backend/src/main/java/com/llmrouter/service/RouterService.java` | 항상 `HEAVY`를 반환한다. | FastAPI ModernBERT 라우터 호출 결과를 `LIGHT/MEDIUM/HEAVY`로 매핑한다. |
| Semantic Cache | `backend/src/main/java/com/llmrouter/service/CacheService.java` | 항상 cache miss를 반환하고 저장은 no-op이다. | bge embedding 생성, pgvector 유사도 검색, 응답 저장/갱신을 구현한다. |
| LLM 호출 | `backend/src/main/java/com/llmrouter/service/LlmService.java` | 모델명만 포함한 더미 응답을 반환한다. | Anthropic Claude API 호출로 교체하고 모델별 timeout/retry/fallback을 둔다. |
| Chat orchestration | `backend/src/main/java/com/llmrouter/service/ChatService.java` | cache -> route -> llm -> cache 흐름은 있으나 실제 하위 구현이 stub이다. | 캐시 hit 메타데이터, 실패 fallback, 비용/latency 로그, 에스컬레이션 연결을 추가한다. |
| API | `backend/src/main/java/com/llmrouter/controller/ChatController.java` | `POST /api/chat`만 있다. | MVP는 유지하고, 최종 목표에서는 상위 모델 재요청 API를 추가한다. |
| Frontend API | `frontend/src/api/chatApi.ts` | `/api/chat` 호출은 가능하지만 스트리밍은 mock 형태다. | MVP는 non-streaming으로 유지하고, 메타데이터 표시와 오류 처리를 안정화한다. |
| Frontend state | `frontend/src/hooks/useChat.ts` | 메시지 송수신과 latency/cache/model 표시용 상태가 있다. | 에스컬레이션 버튼과 재요청 결과 반영을 추가한다. |

## 2. MVP 범위

MVP 목표는 "한 번의 사용자 질문이 실제 캐시, 라우터, Claude API를 거쳐 응답되는 end-to-end 경로"를 만드는 것이다. Kubeflow 재학습과 완전한 자동 배포는 MVP 밖으로 둔다.

### 포함

- Spring Gateway의 `/api/chat` 요청을 실제 Anthropic Claude API까지 연결한다.
- pgvector 기반 semantic cache를 붙인다.
- 캐시 미스 시 라우터가 `haiku`, `sonnet`, `opus` 중 하나를 선택한다.
- ModernBERT가 아직 준비되지 않았을 경우 규칙 기반 또는 mock FastAPI 라우터로 API 계약을 먼저 고정한다.
- 응답에는 `model`, `cacheHit`, `latency`가 포함된다.
- 프론트는 기존 채팅 UI에서 모델, 캐시 hit 여부, latency를 안정적으로 보여준다.
- 로컬 실행 기준으로 backend, frontend, PostgreSQL/pgvector, router-serving이 함께 동작한다.

### 제외

- 사용자 피드백 기반 자동 재학습 루프
- Kubeflow Pipelines 전체 자동화와 MLMD 계보 추적
- GCE/Cloud Run/GKE 운영 배포 완성
- adaptive cache threshold
- 대규모 반복 실험과 최종 보고서용 그래프 freeze
- streaming 응답

## 3. 최종 목표

최종 목표는 "비용을 줄이면서 품질 저하를 제한하는 LLM 라우팅 시스템과 이를 지속 개선하는 MLOps 루프"를 실증하는 것이다.

- 캐시 hit이면 LLM 호출 없이 유사 질문의 검증된 답변을 반환한다.
- 캐시 miss이면 ModernBERT 라우터가 질문 난이도를 easy/medium/hard로 분류하고 Claude Haiku/Sonnet/Opus로 연결한다.
- 사용자가 불만족하면 상위 모델로 에스컬레이션하고, 최종 만족 티어를 정정 라벨로 기록한다.
- 정정 라벨이 누적되면 Kubeflow Pipeline이 재학습, 평가, 배포 게이트를 수행한다.
- RouterBench 재라벨링, RouterArena 일반화 테스트, QQP cache threshold 평가가 재현 가능하게 실행된다.
- GCE Gateway/pgvector, FastAPI Router, Cloud Run Frontend, GKE/Kubeflow 학습 파이프라인이 배포 가능한 형태로 정리된다.
- 최종 보고서에는 비용 절감, 품질 유지, cache 효과, latency, 재학습 효과가 정량 결과로 들어간다.

## 4. 평가 지표 초안

### 라우터 품질

| 지표 | 정의 | MVP 목표 | 최종 목표 |
| --- | --- | --- | --- |
| 3-class accuracy | RouterBench 유래 easy/medium/hard 라벨과 예측 일치율 | baseline 측정 | ModernBERT v0/v1 비교 |
| under-routing rate | 실제 필요 티어보다 낮게 보낸 비율 | 측정 가능하게 로그화 | always-Opus 대비 품질 손실 한도 내 관리 |
| over-routing rate | 실제 필요 티어보다 높게 보낸 비율 | 측정 가능하게 로그화 | 비용 증가 원인으로 분석 |
| macro F1 | 클래스 불균형을 고려한 분류 성능 | baseline 측정 | cutoff/라벨링 방식별 비교 |
| RouterArena transfer score | RouterBench 학습 모델의 RouterArena 일반화 성능 | 제외 | 최종 평가 포함 |

### 비용/품질 트레이드오프

| 지표 | 정의 | MVP 목표 | 최종 목표 |
| --- | --- | --- | --- |
| cost per 1K queries | 라우팅 분포와 cache hit을 반영한 1,000쿼리당 추정 비용 | 계산식 구현 | always-Opus 대비 약 3배 절감 검증 |
| quality retention | always-Opus 대비 정답/선호/평가 점수 유지율 | 소규모 수동 평가 | 벤치마크 기반 반복 측정 |
| model distribution | haiku/sonnet/opus 호출 비율 | 로그 확인 | 비용 분석 표에 포함 |

### Semantic Cache

| 지표 | 정의 | MVP 목표 | 최종 목표 |
| --- | --- | --- | --- |
| cache hit rate | 전체 요청 중 캐시 hit 비율 | 로컬 테스트에서 동작 확인 | 실사용/벤치 데이터에서 보고 |
| false-hit rate | 의미가 다른 질문을 hit으로 잘못 처리한 비율 | 수동 샘플 확인 | QQP threshold sweep으로 측정 |
| threshold curve | 임계값별 hit rate와 false-hit rate 곡선 | 제외 | 0.60-0.90 구간 실험 |
| cache latency | cache lookup에 걸린 시간 | API 응답에 포함 | p50/p95 보고 |

### 시스템/운영

| 지표 | 정의 | MVP 목표 | 최종 목표 |
| --- | --- | --- | --- |
| end-to-end latency | 요청부터 응답까지 전체 시간 | API 응답에 포함 | cache hit/miss별 p50/p95 |
| fallback rate | 라우터/LLM/cache 실패로 fallback한 비율 | 로그화 | 운영 안정성 지표 |
| retraining trigger count | 정정 라벨 N건 도달로 재학습이 시작된 횟수 | 제외 | Kubeflow 루프 검증 |
| deploy gate pass/fail | 새 모델이 기존 모델 대비 지표를 통과했는지 | 제외 | MLMD와 함께 기록 |

## 5. RouterBench 3단계 데이터 라벨링 명세

### 5-1. 라벨의 의미

RouterBench의 `easy/medium/hard`는 추상적인 지식 난이도가 아니라 **질문에 만족스러운 답을 생성할 수 있는 최소 모델 성능 티어**를 뜻한다. 저성능 모델군으로 충분할수록 쉬운 질문으로 본다.

| 라벨 | 판정 의미 | 운영 매핑 |
| --- | --- | --- |
| `easy` | low-capability 모델군으로 안정적으로 해결 가능 | Claude Haiku |
| `medium` | low 모델군은 불충분하고 medium-capability 모델군이 필요 | Claude Sonnet |
| `hard` | medium 모델군까지 불충분하여 high-capability 모델이 필요하거나 모든 모델이 실패 | Claude Opus |

RouterBench 모델은 Claude 제품군의 프록시이므로, 이 매핑은 클래스 및 도메인별 Claude 샘플 검증으로 보정한다. 품질 손실 위험이 비용 증가보다 크므로 라벨 경계와 모델 선택 임계값은 under-routing에 더 큰 페널티를 주도록 조정한다.

### 5-2. 입력 데이터 정규화

- MMLU, HellaSwag, GSM8K, ARC-Challenge, WinoGrande는 논문의 5-shot 결과를 사용한다.
- MBPP, MT-Bench, RAG는 논문의 0-shot 결과를 사용한다.
- 0-shot과 5-shot 파일을 단순 연결하지 않고 `sample_id` 기준으로 canonical row 하나만 유지한다.
- ModernBERT 입력에는 few-shot 예시가 아니라 실제 대상 질문만 들어가도록 `prompt`에서 query text를 분리한다.
- 동일 질문과 그 변형은 같은 split에 배치하여 train/validation/test 누수를 방지한다.
- RAG는 모델 추론 난이도와 검색 필요성을 혼동할 수 있으므로 `needs_retrieval`을 별도 기록한다. 초기 3-class 학습 포함 여부는 실험 설정으로 고정하고 별도 결과를 보고한다.

### 5-3. 모델 성능 티어 산정

1. 11개 모델의 데이터셋별 평균 `performance`를 계산한다.
2. 객관식, 수학, 코드, 오픈엔디드의 점수 체계 차이를 줄이기 위해 데이터셋별로 성능을 정규화한다.
3. 정규화된 성능의 도메인 macro average로 모델의 종합 capability를 산정한다.
4. 종합 capability를 기준으로 모델을 `low`, `medium`, `high` 그룹으로 나눈다.
5. 비용은 capability를 대신하지 않으며, 성능이 유사한 모델 사이의 tie-breaker로만 사용한다.

모델 이름을 코드에 직접 고정하지 않고 산정 결과를 `model_tiers.json`으로 저장한다. 파일에는 tier 구성, 산정 metric, 데이터 버전, 생성 코드 commit을 기록하고 `tier_config_version`으로 추적한다.

### 5-4. 모델 응답의 성공 판정

- exact-match 계열은 `performance == 1`을 성공으로 본다.
- judge 또는 연속 점수 계열은 `performance >= acceptance_threshold[dataset]`을 성공으로 본다.
- 연속 점수 임계값은 초기 후보 `0.60`, `0.70`, `0.80`을 비교하고 Claude 검증 및 quality retention 결과로 확정한다.
- 결측 응답은 오답으로 단정하지 않고 해당 tier의 유효 모델 수에서 제외하며, 유효 모델 수가 부족하면 `low_confidence`로 기록한다.

질문 `q`와 모델 티어 `t`의 성공률은 다음과 같이 정의한다.

```text
tier_success(q, t)
  = t에서 성공한 모델 수 / t에서 유효한 모델 수
```

### 5-5. easy/medium/hard 판정 규칙

초기 합의 임계값은 low와 medium 티어에 `0.67`, high 티어에 `0.50`을 적용한다.

```python
if low_success_rate >= 0.67:
    difficulty_label = "easy"
    label_reason = "low_tier_consensus"
elif medium_success_rate >= 0.67:
    difficulty_label = "medium"
    label_reason = "medium_tier_consensus"
elif high_success_rate >= 0.50:
    difficulty_label = "hard"
    label_reason = "strong_only"
else:
    difficulty_label = "hard"
    label_reason = "unsolved"
```

`strong_only`와 `unsolved`는 운영상 모두 가장 안전한 티어로 보내기 위해 `hard`로 합치되, 분석과 재학습에서는 반드시 구분한다. `unsolved`는 Opus가 해결할 수 있다는 보장이 아니라 현재 선택지 중 가장 강한 모델을 사용해야 한다는 의미다.

### 5-6. 보조 신호와 라벨 신뢰도

- `solve_rate`: 전체 유효 모델 중 성공한 비율로, 주 라벨의 일관성과 이상치를 점검한다.
- `cheapest_correct_tier`: 정답을 낸 가장 저렴한 모델의 티어로, 동률 해소와 비용 분석에 사용한다.
- solve-rate 삼분위: 클래스 균형을 확인하는 대조군으로만 사용하고, 클래스 수를 억지로 1/3씩 맞추는 기준으로 사용하지 않는다.
- low 티어 한 모델만 성공하거나, 상위 티어로 갈수록 성공률이 감소하거나, 0/5-shot 결과가 크게 다르면 `ambiguous`로 표시한다.

초기 ModernBERT baseline은 `high_confidence` 라벨을 우선 사용한다. `medium_confidence`는 낮은 sample weight를 적용할 수 있고, `ambiguous`는 학습에서 제외하되 분석 데이터로 보존한다.

### 5-7. 라벨 산출 스키마와 버전 관리

```text
sample_id
query_text
domain
low_success_rate
medium_success_rate
high_success_rate
overall_solve_rate
cheapest_correct_tier
difficulty_label
label_reason
label_confidence
label_weight
needs_retrieval
dataset_version
tier_config_version
label_rule_version
```

라벨 산출물은 Pickle 원본을 그대로 학습 입력으로 사용하지 않고 검증 가능한 Parquet 형식으로 저장한다. 데이터 checksum, row count, 클래스/도메인 분포, 제외 사유별 count를 metric JSON에 남긴다. 임계값, 모델 티어, RAG 포함 여부가 바뀌면 `label_rule_version`을 올리고 재라벨링한다.

### 5-8. 학습 파이프라인 연결 범위

이 작업 흐름은 다음 단계를 한 브랜치에서 순차적으로 구축한다.

```text
ingest -> canonicalize -> calibrate tiers -> relabel -> split
       -> local smoke train -> evaluate
       -> GCS artifact upload -> Kubeflow/GKE train -> deploy gate
```

로컬 smoke test는 소량 샘플로 전처리, tokenizer, forward/backward, checkpoint 저장 및 평가 metric 생성을 검증한다. GCP/GKE 단계는 동일한 데이터·라벨·학습 설정을 사용하고 실행 환경만 확장하며, 라벨링 CPU 작업과 ModernBERT GPU 학습 작업을 별도 컴포넌트로 분리한다.

### 5-9. RouterBench 학습 전처리

라벨링 산출물의 `query_text`는 감사용 원문으로 유지하고, RouterBench 객관식 평가에만
필요한 앞뒤 답변 형식 문구를 제거한 `model_input_text`를 별도로 만든다. 문제 본문,
선택지, 코드와 대화 문맥은 보존한다. 문자/단어 길이는 기록하되 tokenizer 기반
truncation은 ModernBERT tokenizer가 확정된 학습 컴포넌트에서 수행한다.

정규화된 동일 질문과 token 3-gram 기반 near duplicate를 cluster로 묶고,
`duplicate_group_id` 단위로 train/validation/test를 분할한다. split은 label과 domain
분포를 함께 고려하며 같은 group이 여러 split에 나타나면 파이프라인을 실패시킨다.

초기 표본 정책은 high confidence weight `1.0`, medium `0.5`, low `0.0`이다.
low confidence는 학습에서는 제외하지만 metadata에는 보존한다. `hard/strong_only`와
`hard/unsolved`, RAG 포함 여부는 분석 가능한 별도 필드와 versioned config로 유지한다.

학습 projection에는 `model_input_text`, `difficulty_label`, `sample_weight`와 split/group
식별자만 둔다. tier 성공률과 cheapest-correct tier는 별도 metadata artifact에 두고,
모델별 원시 성능·응답·비용은 라벨 산출물에만 남겨 라벨 누수를 방지한다.

Kubeflow는 전처리 규칙 자체를 소유하지 않는다. 동일한 Python 전처리 component를
로컬과 GKE에서 실행하고, Kubeflow는 Artifact 전달, caching, ML Metadata 계보,
retry, CPU/GPU resource 분리, GCS 경로와 parameterized 실험을 담당한다. 생성된 split
artifact는 downstream 학습에서 재사용하며 학습 Pod 내부에서 다시 만들지 않는다.

## 6. 현행 작업 추적

이 절에 있던 구현 체크리스트는 2026-07 시점의 초안이라 현재 코드와 크게 달라졌다.
ML 진행 상태와 온라인 팀 협의 agenda는
`docs/ml/workstream-checklist.md`로 통합했다. 팀 전체의 frontend,
backend와 배포 일정은 GitHub issue/milestone에서 관리한다.

## 7. MVP 완료 기준

- 사용자가 프론트에서 질문을 보내면 `/api/chat`이 실제 backend로 호출된다.
- cache miss 요청은 router 선택 결과에 따라 Claude API를 호출한다.
- 같은 또는 유사한 질문을 다시 보내면 cache hit으로 응답한다.
- 응답 UI에 모델명, cache hit 여부, latency가 보인다.
- backend 로그에서 request id 기준으로 cache, router, LLM 호출 흐름을 추적할 수 있다.
- RouterBench 소량 샘플의 canonicalization, 3-class relabeling, split, ModernBERT 로컬 smoke train이 한 명령 흐름으로 완료된다.
- 로컬 재현 절차가 문서화되어 새 팀원이 30분 안에 실행할 수 있다.

## 8. 최종 완료 기준

- 버전이 고정된 RouterBench 라벨 산출물로 로컬과 GKE의 ModernBERT 학습이 재현 가능하다.
- always-Opus 대비 비용/품질 트레이드오프 결과가 있다.
- QQP 기반 cache threshold sweep 결과가 있다.
- RouterArena 일반화 평가 결과가 있다.
- 에스컬레이션 기록이 DB에 저장되고 정정 라벨로 재학습 입력에 들어간다.
- Kubeflow Pipeline이 학습, 평가, artifact 업로드, 배포 게이트를 수행한다.
- 최종 보고서와 발표자료에 들어갈 표, 그래프, 데모 시나리오가 freeze되어 있다.
