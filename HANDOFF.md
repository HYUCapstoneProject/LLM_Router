# LLM Router 개발 인수인계

기준일: 2026-08-30

이 문서는 다른 AI/개발자가 현재 저장소에서 바로 작업을 이어가기 위한 기준 문서다.
세부 일정은 `PLAN.md`, 배포 절차는 `infra/DEPLOY.md`, reranker 근거는
`experiments/results/reranker-20260830/REPORT.md`를 함께 확인한다.

## 1. 담당 범위와 저장소 기준

- 담당 범위: Spring Boot 백엔드, React 프론트엔드, PostgreSQL/pgvector 시맨틱 캐시
- 팀원 연동 범위: FastAPI embedding·routing inference와 데이터 처리·재학습 파이프라인
- 원격 저장소: `https://github.com/HYUCapstoneProject/LLM_Router.git`
- 작업 브랜치: `docs/demo-questions`
- 작업 시작 기준: `9325cba` (`origin/docs/demo-questions`)
- `main` 기준: `e299772`

현재 브랜치는 `main` 위에 시연 질문셋과 라우터 실측 문서 커밋이 추가된 상태다.
배포 스크립트는 안전장치 때문에 `main`의 깨끗한 작업 트리에서만 실행된다. 따라서
이 브랜치에서 바로 배포하지 말고 리뷰·머지 후 최신 `main`에서 실행한다.

## 2. 시스템 구조와 책임

```text
Browser
  → Cloud Run: nginx + React
  → GCE: Spring Boot Gateway
      ├─ PostgreSQL/pgvector: cache_entry, request_log, escalation_log
      ├─ FastAPI Router: BGE embedding, ModernBERT easy/medium/hard classification
      └─ Anthropic: Haiku / Sonnet / Opus
```

Spring Gateway가 cache 조회·저장, 실제 LLM 선택·호출, escalation, 비용·토큰·지연 로그를
담당한다. FastAPI는 embedding과 classification inference만 담당한다. 이 책임 경계를
reranker에서도 유지해 FastAPI는 pair score를 반환하고 최종 hit/miss 정책과 로그는
Spring이 결정하는 방향이다.

현재 캐시 흐름:

```text
질문 원문 → exact match → BGE embedding → pgvector 후보 조회
          → cache hit이면 router/Claude 생략
          → miss이면 ModernBERT 분류 → Claude 호출 → cache 저장
```

라우팅에는 질문 평문을 넣고, embedding은 시맨틱 캐시 후보 검색에 사용한다.
JPA가 아니라 JDBC를 쓰는 이유는 pgvector `vector(1024)`, `<=>` 연산자, HNSW 후보 조회와
항목별 threshold가 어차피 native SQL이기 때문이다.

## 3. 구현된 주요 기능

- `POST /api/chat`: cache hit/miss, FastAPI routing, Claude 호출과 저장
- `POST /api/escalate`: 상위 모델 재요청, 중복 과금 방지, 체인·정정 로그, cache 갱신
- `GET /api/analytics/summary`: 요청·성공·cache hit·모델 비율·토큰·비용·지연 집계
- React 채팅 UI, 모델·tier·token·latency·cache hit 표시, escalation 버튼
- React 누적 분석 대시보드
- FastAPI 장애 시 embed는 cache bypass, classify는 medium fallback
- request ID, model version, confidence, fallback reason, status/error/origin 기록
- GCE 백엔드와 Cloud Run 프론트 반복 배포 스크립트

운영 리소스 이름과 마지막 문서화 상태:

- Cloud Run: `llm-router-frontend`, `asia-northeast3`
- GCE: `llm-router-workspace`, `asia-northeast3-a`
- PostgreSQL container: `llmrouter-postgres`
- Spring container: `llm-router-backend`
- FastAPI Router: 내부 주소 `10.138.0.2:8000`

이 인수인계 작성 시점에는 실제 서버 revision과 health를 다시 조회하지 않았다.
배포 전 `gcloud`, 컨테이너 상태, DB migration `04-router-confidence.sql`, 현재 image digest를
반드시 확인한다.

## 4. Reranker 결정과 직접 평가 결과

### 결정

- pgvector 후보를 cross-encoder로 재검증하는 2단계 구조는 채택한다.
- 생성형 LLM judge를 cache hit마다 호출하는 방식은 채택하지 않는다.
- 운영 모델과 임계값은 아직 확정하지 않았다.
- 현재 최선값은 실제 응답 판정값이 아니라 shadow mode 후보값이다.

### 평가 데이터와 기준

- `experiments/rerank-eval-pairs.csv`: 58쌍
- positive 21, hard negative 37
- 1차 gate: false hit 0건과 positive recall 90% 이상
- 네 후보를 같은 로컬 CPU 환경에서 비교

### 결과

최선 후보는 `cross-encoder/quora-distilroberta-base`와 기존 BGE cosine의 AND 조합이다.

```text
BGE cosine >= 0.8774
AND reranker score >= 0.03581769
```

- TP 20, FP 2, TN 35, FN 1
- precision 90.91%, recall 95.24%, false-hit rate 5.41%, F1 93.02%
- 같은 recall gate의 BGE 단독은 TP 19, FP 7, F1 80.85%
- 남은 false hit: red-black tree 삭제↔삽입, quicksort↔mergesort
- miss: HTTP 약어 질문 paraphrase

엄격한 gate를 통과한 모델은 없다. 따라서 위 임계값을 production cache hit에 하드코딩하면
안 된다. 독립 평가셋 확장, 필요 시 cache equivalence 데이터 fine-tuning, shadow 검증 후
다시 임계값을 고정한다.

원본 점수·revision·latency·RSS:

- `experiments/results/reranker-20260830/REPORT.md`
- 모델별 CSV/JSON: `experiments/results/reranker-20260830/*/`
- GPTCache 조사: `experiments/GPTCACHE_COMPARISON.md`

GPTCache quick-start는 ALBERT embedding + FAISS + vector distance가 기본이다. 선택 가능한
pair evaluator로 `GPTCache/albert-duplicate-onnx`와 기본 Quora cross-encoder가 있다.
공식 ALBERT evaluator도 같은 58쌍에서 실행했지만 recall 95.24% 조건 FP 17건으로
Quora 모델보다 나빴다. GPTCache 라이브러리 자체를 도입하지 않고 현재 Spring/pgvector
구조를 유지한다.

## 5. 다음 구현 순서

운영 모델 확정 전에도 다음 기반 코드는 구현할 수 있다.

1. FastAPI `POST /router/rerank` 요청·응답 계약 확정
2. Spring contract test를 먼저 작성
3. pgvector 조회를 top-k 후보 질문·답변·similarity 반환 형태로 확장
4. reranker batch 호출과 model version·score·latency·decision reason 기록
5. timeout, 5xx, 길이 불일치, NaN은 무조건 cache miss로 처리하는 fail-closed 폴백
6. feature flag와 shadow mode 구현: 점수는 기록하되 기존 cache 결정은 바꾸지 않음
7. 독립 test set 통과 후에만 최종 AND threshold로 활성화

그다음 우선순위는 `PLAN.md`의 다음 항목이다.

- 발표용 Router Only / Router + Cache / Always 모델 ablation
- Random, Rule-based, Oracle, RouteLLM baseline과 cost-accuracy Pareto
- escalation 유효 로그 1,000건 단위 GCS batch export와 FastAPI 재학습 trigger
- 대화 localStorage, 모바일 UI, 인증/TLS 등 운영 마무리

## 6. 평가 코드와 재현

평가 로직은 `experiments/evaluate_reranker.py`, 단위 테스트는
`experiments/test_evaluate_reranker.py`다. 모델 단독 threshold와 BGE+reranker joint
threshold를 모두 탐색하며 비정상 label·NaN도 거부한다.

Windows CPU 예시:

```powershell
python -m venv .venv-reranker
.\.venv-reranker\Scripts\python.exe -m pip install `
  --index-url https://download.pytorch.org/whl/cpu torch
.\.venv-reranker\Scripts\python.exe -m pip install `
  -r experiments\requirements-reranker.txt

.\.venv-reranker\Scripts\python.exe -m unittest discover `
  -s experiments -p test_evaluate_reranker.py -v

.\.venv-reranker\Scripts\python.exe experiments\evaluate_reranker.py `
  --model cross-encoder/quora-distilroberta-base `
  --input experiments\rerank-eval-pairs.csv `
  --output-dir experiments\results\reranker-local `
  --cache-dir .hf-cache --device cpu --batch-size 8
```

GPTCache 공식 ONNX evaluator는 `--backend gptcache-onnx --model
GPTCache/albert-duplicate-onnx`로 실행한다. 모델별 threshold는 revision이 바뀌면 다시
측정해야 하며 서로 다른 모델의 score 값을 직접 비교하면 안 된다.

이번 변경에서 reranker 평가 단위 테스트 13개가 통과했다. 전체 Spring·React 테스트는
reranker 문서/평가 변경 범위 밖이고 대형 캐시로 인한 로컬 저장공간 문제 때문에 다시
실행하지 않았다. 임시 venv와 모델 캐시는 삭제했고 C 드라이브 여유 공간은 약 4.18GB다.
대형 후보를 추가 평가할 때는 GCE workspace를 사용해도 된다.

## 7. 일반 테스트·배포 명령

```bash
# Spring
cd backend
./gradlew test

# React
cd frontend
npm ci
npm test
npm run build

# main 머지 후 배포
./scripts/deploy-backend.sh
./scripts/deploy-frontend.sh
```

백엔드 배포 스크립트는 DB migration을 실행하지 않는다. schema 변경이 있으면
`infra/DEPLOY.md` 절차대로 migration을 먼저 적용한다. 배포 스크립트는 `main`이며
작업 트리가 깨끗할 때만 동작한다.

## 8. 비밀정보와 주의사항

다음 값은 저장소에 커밋하지 않는다.

- `ANTHROPIC_API_KEY`
- `POSTGRES_PASSWORD`
- `GATEWAY_KEY`
- `ROUTER_API_TOKEN`
- GCP service-account key 또는 Secret Manager 출력

운영 profile은 `GATEWAY_KEY`, FastAPI mode는 `ROUTER_API_TOKEN`이 없으면 기동 실패하도록
유지한다. DB 로그 적재 오류는 사용자 응답을 막지 않으므로 배포 후 반드시
`request_log` 실제 행과 backend 로그를 확인한다.

## 9. 핵심 문서 지도

- `PLAN.md`: 전체 진행 상태와 발표·재학습 계획
- `infra/DEPLOY.md`: GCE, Cloud Run, DB migration, smoke test runbook
- `experiments/RERANKER_REQUEST.md`: reranker 결정 기준과 API 초안
- `experiments/results/reranker-20260830/REPORT.md`: 직접 평가 결과
- `experiments/GPTCACHE_COMPARISON.md`: GPTCache 공식 구현·모델 비교
- `experiments/evaluate_reranker.py`: 재현 가능한 평가 도구
- `backend/src/main/resources/application.yml`: 모델·cache·router 운영 설정
