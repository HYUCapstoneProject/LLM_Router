# GitHub Setup

이 문서는 `HYUCapstoneProject/LLM_Router` 단일 repo 기준으로 최초 등록한 GitHub Project, milestone, issue, label 정보를 정리한 것이다.

## Repository

| 항목 | 값 |
| --- | --- |
| Organization | `HYUCapstoneProject` |
| Repository | `LLM_Router` |
| Repository URL | https://github.com/HYUCapstoneProject/LLM_Router |
| 운영 방식 | 초기에는 단일 monorepo로 진행 |

## Projects

| Project No. | Project | 목적 | URL | Items |
| ---: | --- | --- | --- | ---: |
| 2 | LLM Router Delivery | 구현, 배포, 문서, runtime 통합 작업 관리 | https://github.com/orgs/HYUCapstoneProject/projects/2 | 8 |
| 3 | LLM Router Experiments | 데이터셋, 라벨링, 학습, 평가, 실험 결과 관리 | https://github.com/orgs/HYUCapstoneProject/projects/3 | 4 |

## Project Fields

| Project | Field | Type | Options |
| --- | --- | --- | --- |
| LLM Router Delivery | Area | Single select | `frontend`, `gateway`, `cache`, `router-serving`, `ml-training`, `kubeflow`, `infra`, `evaluation`, `docs` |
| LLM Router Delivery | Priority | Single select | `p0`, `p1`, `p2` |
| LLM Router Experiments | Experiment Type | Single select | `dataset`, `relabeling`, `training`, `evaluation`, `cache`, `report` |
| LLM Router Experiments | Priority | Single select | `p0`, `p1`, `p2` |
| LLM Router Experiments | Artifact | Text | 모델/데이터/그래프 산출물 경로 |
| LLM Router Experiments | Metric | Text | 주요 평가 metric |

## Milestones

| No. | Milestone | Due Date | Open Issues | URL |
| ---: | --- | --- | ---: | --- |
| 5 | M0 Training First Kickoff | 2026-07-07 | 0 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/5 |
| 3 | M1 Infra/Data Ready | 2026-07-14 | 3 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/3 |
| 4 | M2 Router v0 | 2026-07-31 | 3 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/4 |
| 1 | M3 Gateway/Cache MVP | 2026-08-03 | 5 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/1 |
| 2 | M4 Local Integrated System | 2026-08-24 | 1 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/2 |
| 6 | M5 Initial Cloud Deployment | 2026-08-31 | 0 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/6 |
| 9 | M6 Stabilization Freeze | 2026-09-14 | 0 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/9 |
| 8 | M7 Experiment Results Freeze | 2026-09-30 | 0 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/8 |
| 7 | M8 Final Submission | 2026-11-01 | 0 | https://github.com/HYUCapstoneProject/LLM_Router/milestone/7 |

## Issues

| Issue | Title | Project | Milestone | Labels | URL |
| ---: | --- | --- | --- | --- | --- |
| #1 | GCP/T4 quota와 budget alert 준비 | LLM Router Delivery | M1 Infra/Data Ready | `area:infra`, `priority:p0` | https://github.com/HYUCapstoneProject/LLM_Router/issues/1 |
| #2 | RouterBench easy/medium/hard 재라벨링 구현 | LLM Router Experiments | M2 Router v0 | `area:ml-training`, `priority:p0` | https://github.com/HYUCapstoneProject/LLM_Router/issues/2 |
| #3 | RouterBench/RouterArena/QQP 접근 및 샘플 스키마 검증 | LLM Router Experiments | M1 Infra/Data Ready | `area:ml-training`, `priority:p0` | https://github.com/HYUCapstoneProject/LLM_Router/issues/3 |
| #4 | KFP skeleton과 GCS artifact 경로 설계 | LLM Router Delivery | M1 Infra/Data Ready | `area:kubeflow`, `priority:p0` | https://github.com/HYUCapstoneProject/LLM_Router/issues/4 |
| #5 | ModernBERT base baseline 학습 | LLM Router Experiments | M2 Router v0 | `area:ml-training`, `area:evaluation`, `priority:p0` | https://github.com/HYUCapstoneProject/LLM_Router/issues/5 |
| #6 | 라우터 평가 및 cutoff 민감도 분석 | LLM Router Experiments | M2 Router v0 | `area:ml-training`, `area:evaluation`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/6 |
| #7 | 최소 API 계약 초안 작성 | LLM Router Delivery | M3 Gateway/Cache MVP | `area:frontend`, `area:router-serving`, `area:gateway`, `area:docs`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/7 |
| #8 | PostgreSQL + pgvector 캐시 스키마 설계 | LLM Router Delivery | M3 Gateway/Cache MVP | `area:cache`, `area:gateway`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/8 |
| #9 | 임베딩 생성 경로 구현 | LLM Router Delivery | M3 Gateway/Cache MVP | `area:cache`, `area:gateway`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/9 |
| #10 | 시맨틱 캐시 조회/저장/갱신 구현 | LLM Router Delivery | M3 Gateway/Cache MVP | `area:cache`, `area:gateway`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/10 |
| #11 | FastAPI router-serving skeleton 추가 | LLM Router Delivery | M4 Local Integrated System | `area:router-serving`, `area:gateway`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/11 |
| #12 | Anthropic Claude API 연동 | LLM Router Delivery | M3 Gateway/Cache MVP | `area:gateway`, `priority:p1` | https://github.com/HYUCapstoneProject/LLM_Router/issues/12 |

## Labels

| Label | 의미 |
| --- | --- |
| `area:frontend` | React UI and frontend deployment |
| `area:gateway` | Spring Gateway and API orchestration |
| `area:cache` | Semantic cache, embeddings, PostgreSQL, pgvector |
| `area:router-serving` | FastAPI router serving and inference path |
| `area:ml-training` | Datasets, relabeling, ModernBERT training |
| `area:kubeflow` | Kubeflow Pipelines, GKE, MLMD |
| `area:infra` | GCP infra, deployment, cost controls |
| `area:evaluation` | Metrics, benchmark evaluation, final result freeze |
| `area:docs` | Documentation, Notion sync, report materials |
| `priority:p0` | Blocks schedule, demo, or critical path |
| `priority:p1` | Should be completed within the current milestone |
| `priority:p2` | Nice-to-have improvement after core path |

## 운영 메모

- 작업 상태의 source of truth는 GitHub Project다.
- 실험 결과와 기술 결정의 source of truth는 Notion이다.
- issue는 모두 하나의 milestone에 연결한다.
- repo는 현재 `LLM_Router` 단일 repo로 시작하고, `mlops/`, `infra/`가 독립 실행 단위로 커질 때 분리 여부를 다시 판단한다.
