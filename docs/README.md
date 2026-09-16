# 문서 안내

프로젝트 전체에 공통으로 적용되는 문서는 `docs/`에, ML 데이터·학습·평가·Kubeflow에
특화된 문서는 `docs/ml/`에 둔다.

## 공통 프로젝트 문서

| 문서 | 역할 |
| --- | --- |
| `졸업프로젝트 개요.md` | 전체 시스템 구상, 데이터·평가·인프라와 최종 목표 |
| `mvp-scope-and-checklist.md` | 2026-07 시점의 역사적 MVP 범위와 완료 기준 |
| `timeline.md` | 전체 일정, 기능별 작업 분해와 마일스톤 |
| `team rules.md` | branch, commit, PR, API, 실험과 비용 운영 규칙 |
| `github setup.md` | GitHub Project, milestone, issue와 label 구성 |

## ML 문서

| 문서 | 역할 |
| --- | --- |
| `ml/workstream-checklist.md` | 현행 ML 구조·상태와 유일한 작업 체크리스트 |
| `ml/backend-infrastructure-agreement.md` | 확정 cloud 실행 구조 통보와 나머지 온라인·인프라 협의 계약 |
| `ml/backend-ml-integration-meeting-guide.md` | 확정 구조 통보, 회의 순서, 결정 선택지와 현장 기록표 |
| `ml/gcp-rollout-and-serving-integration-plan.md` | GCP 설정 runbook과 미구현 코드·모델 전달 경로 구현 명세 |
| `ml/continual-retraining-strategy.md` | 3-call version cycle, compact replay, 5-major rebase와 inline feedback 경계 |
| `ml/development-questions-and-decisions.md` | 개발 과정의 질문, 선택 근거와 미해결 고민 |
| `ml/llm-routing-literature-review.md` | LLM routing 논문 목록·분류와 조사 범위 |
| `ml/llm-routing-benchmark-dataset-analysis.md` | RouterBench·RouterArena 등 원점수, 라벨 생성과 공식 평가 비교 |
| `ml/llm-routing-model-specs-report.md` | 라우팅 모델 아키텍처·성능과 논문별 데이터 규모 조사 |
| `ml/routerbench-data-labeling-spec.md` | RouterBench 검증·라벨링·학습 전처리 계약 |
| `ml/routerarena-evaluation-preprocessing-spec.md` | RouterArena 입력 조립·품질 필터·누수 방지 계약 |
| `ml/routerarena-routing-evaluation-spec.md` | checkpoint 추론, 예측 그룹 분포와 평가 metric 계약 |
| `ml/modernbert-training-spec.md` | ModernBERT runtime, 학습, checkpoint와 Kubeflow 계약 |
| `ml/integrated-routing-data-and-model-comparison-plan.md` | **신규 실험 실행 기준:** 전체 flow, Colab 모델 비교, 데이터·threshold·목적함수, ModernBERT-only Vertex/GCS, FastAPI·API 계약 |
| `ml/training-reports/` | 완료된 학습 run별 결과 관측과 진단 보고서 |
| `ml/model-promotion-log.md` | candidate 조립·검증·승격 이력과 잔여 위험 |

## 문서 관리 원칙

- 현재 ML 진행 상태와 체크박스는 `ml/workstream-checklist.md`에서만 관리한다.
- ModernBERT 초기학습과 다중 모델 비교는 Colab에서 실행한다. Vertex AI Pipelines CPU →
  Vertex AI Custom Training GPU 구조는 ModernBERT feedback 재학습에만 사용하고 GKE는
  사용하지 않는다. 비교용 또는 초기학습용 KFP/Vertex job을 제출하지 않는다.
- 팀 합의가 필요한 값은 `ml/backend-infrastructure-agreement.md`에 결정자와 결정일을
  기록한다.
- 세부 명세는 알고리즘·schema·artifact 계약을 설명하며 별도 진행 체크리스트를 두지
  않는다.
- 제안 상태 문서는 기존 승인 config를 바꾸지 않는다. 실험값을 승인할 때 versioned
  config와 artifact를 새로 만들고 해당 명세의 현행 계약을 갱신한다.
- `mvp-scope-and-checklist.md`는 역사적 기준이고 현재 구현 상태로 해석하지 않는다.
- 파일을 이동하거나 이름을 바꾸면 저장소 전체의 문서 참조도 같은 변경에서 갱신한다.
