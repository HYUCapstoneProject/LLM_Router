# LLM Router Team Rules

이 문서는 LLM Router 졸업프로젝트의 팀 운영 규칙을 한 페이지에 정리한 것이다. 규칙의 목적은 작업을 빠르게 진행하면서도, 나중에 보고서·데모·실험 결과를 재현할 수 있게 만드는 것이다.

## 1. 기본 원칙

- 7-8월에는 모델 학습 인프라와 end-to-end 통합을 최우선으로 한다.
- 초기 개발은 `LLM_Router` 단일 monorepo에서 진행한다.
- `mlops/`, `infra/`가 독립 실행 단위로 커지면 별도 repo 분리를 검토한다.
- 중요한 결정은 말로만 공유하지 않고 Notion 또는 repo 문서에 남긴다.
- 비용이 발생하는 작업(GCP GPU, Kubeflow, Claude API 대량 실험)은 실행 전에 팀에 공유한다.

## 2. Issue & Project 규칙

- 모든 작업은 GitHub issue로 시작한다.
- issue에는 milestone, `area:*`, `priority:*` label을 붙인다.
- 하나의 issue는 하나의 명확한 산출물을 가져야 한다.
- 큰 작업은 작은 issue로 쪼갠다.
- GitHub Project status는 아래 흐름을 따른다.

```text
Backlog -> Ready -> In Progress -> Review -> Done
```

- 막힌 작업은 `Blocked`로 옮기고, 막힌 이유와 필요한 도움을 issue comment에 적는다.
- `LLM Router Delivery`는 구현·배포·문서 작업을 관리한다.
- `LLM Router Experiments`는 라벨링·학습·평가·결과 freeze 작업을 관리한다.

## 3. Label 규칙

필수 area label:

- `area:frontend`
- `area:gateway`
- `area:cache`
- `area:router-serving`
- `area:ml-training`
- `area:kubeflow`
- `area:infra`
- `area:evaluation`
- `area:docs`

필수 priority label:

- `priority:p0`: 일정 또는 데모를 막는 작업
- `priority:p1`: 이번 milestone 안에 처리해야 하는 작업
- `priority:p2`: 여유가 있을 때 처리하는 개선 작업

## 4. Branch 규칙

- `main`에 직접 push하지 않는다.
- 모든 변경은 branch를 만들고 PR로 반영한다.
- branch 이름은 아래 형식을 따른다.

```text
feat/router-serving
feat/mlops-relabeling
fix/cache-miss
docs/timeline-update
chore/project-setup
```

- branch는 작업 목적이 드러나게 짧게 짓는다.

## 5. Commit 규칙

commit message는 현재 repo 스타일에 맞춰 아래 형식을 사용한다.

```text
[TYPE] 변경 내용 요약
```

허용 type:

- `[FEAT]`: 기능 추가
- `[FIX]`: 버그 수정
- `[DOCS]`: 문서 수정
- `[CHORE]`: 설정, 정리, 의존성 관리
- `[REFACTOR]`: 동작 변경 없는 구조 개선
- `[TEST]`: 테스트 추가 또는 수정
- `[BUILD]`: 빌드, 패키지, 의존성 설정
- `[CI]`: GitHub Actions 등 CI 설정

예시:

```text
[DOCS] timeline 학습 우선 계획 반영
[FEAT] router-serving FastAPI skeleton 추가
[FIX] gateway 캐시 미스 처리 오류 수정
[CHORE] frontend node_modules 추적 제거
```

## 6. PR 규칙

- 한 PR은 하나의 목적만 가진다.
- frontend/backend/mlops/infra 변경이 섞이는 PR은 통합 작업일 때만 허용한다.
- PR 설명에는 아래 항목을 포함한다.

```text
## 목적
## 변경 내용
## 테스트/검증
## 관련 issue
```

- 최소 1명 리뷰 후 merge한다.
- 리뷰에서 논의된 결정은 PR comment 또는 관련 문서에 남긴다.
- 테스트를 못 했으면 못 한 이유를 PR에 명시한다.

## 7. 문서 규칙

- 일정과 milestone은 Notion timeline을 최신본으로 둔다.
- repo에는 구현에 필요한 실행 문서와 기술 문서를 둔다.
- API 계약, 인프라 변경, 모델 실험 결과는 반드시 문서화한다.
- 실행 방법이 바뀌면 README 또는 관련 문서를 같이 수정한다.
- 중요한 기술 결정은 Decision Log에 남긴다.

Decision Log에는 최소한 아래 내용을 적는다.

```text
날짜:
결정:
이유:
대안:
영향:
```

## 8. Secret & 환경변수 규칙

- `.env`, API key, GCP credential, service account key는 절대 commit하지 않는다.
- repo에는 `.env.example`만 둔다.
- 새 환경변수가 생기면 `.env.example`과 실행 문서를 함께 수정한다.
- Anthropic API key, GCP credential, Hugging Face token 등은 팀에서 합의한 안전한 위치에 저장한다.
- 실수로 secret을 commit하면 즉시 팀에 알리고 key rotation을 진행한다.

## 9. API 계약 규칙

- endpoint, request, response, error format은 구현 전에 최소 초안을 문서화한다.
- API breaking change는 issue와 PR에 명시한다.
- frontend와 backend가 동시에 영향을 받는 API 변경은 PR 전에 팀에 공유한다.
- 응답에는 데모에 필요한 정보가 빠지지 않게 한다.

채팅 응답에서 우선 필요한 필드:

```text
response
model
cacheHit
latency
```

추후 에스컬레이션 기능에는 아래 필드를 추가 검토한다.

```text
requestId
initialModel
finalModel
escalationPath
```

## 10. 실험 & 모델 학습 규칙

- 실험은 재현 가능해야 한다.
- 각 실험은 아래 정보를 남긴다.

```text
실험 이름:
날짜:
목적:
데이터셋/버전:
코드 commit hash:
모델:
주요 hyperparameter:
평가 metric:
결과:
다음 액션:
```

- 실패한 실험도 간단히 기록한다.
- RouterBench 재라벨링 기준, train/validation/test split, cutoff 변경은 반드시 기록한다.
- 최종 보고서에 사용할 실험 결과는 `M7 Experiment Results Freeze` 전에 확정한다.
- 모델 artifact 이름은 아래 형식을 따른다.

```text
router-{model}-{dataset}-{date}-{shortCommit}
```

예시:

```text
router-modernbert-base-routerbench-20260731-21c2f1b
```

## 11. Infra & 비용 규칙

- GCP 리소스를 새로 만들 때는 목적, 예상 비용, 종료 방법을 문서화한다.
- GPU, GKE, Kubeflow, Claude API 대량 호출은 실행 전에 팀에 공유한다.
- 실험 후 사용하지 않는 VM, cluster, GPU node pool은 중지하거나 삭제한다.
- budget alert를 설정하고, 비용 이상 징후가 있으면 즉시 공유한다.
- 배포 절차는 `infra/` 문서 또는 runbook에 남긴다.

## 12. Merge 전 체크리스트

PR 작성자는 merge 전에 아래를 확인한다.

- 관련 issue가 연결되어 있다.
- milestone과 label이 붙어 있다.
- secret이나 로컬 파일이 포함되지 않았다.
- 실행 방법이 바뀌었다면 문서를 수정했다.
- 가능한 테스트 또는 검증을 수행했다.
- 테스트를 못 했다면 이유를 PR에 적었다.
