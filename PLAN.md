# LLM Router 작업 계획

> 기준일: 2026-08-16
>
> 범위: React 프론트엔드, Spring Boot 게이트웨이, PostgreSQL/pgvector 캐시, 배포 및 발표용 분석 기능
>
> 팀원 연동 범위: FastAPI 임베딩·라우팅 서버, 데이터 라벨링 및 재학습 파이프라인

## 1. 현재 기준선

### 소스 코드

현재 로컬 `main`에 통합할 범위는 다음과 같다.

- React 채팅 UI와 모델·티어·토큰·지연시간·캐시 히트 표시
- Spring Boot `POST /api/chat`과 Anthropic Claude 호출
- PostgreSQL/pgvector 기반 시맨틱 캐시 조회·저장
- `POST /api/escalate` 재요청, 중복 과금 방지, 체인 기록, 캐시 답변 갱신
- 성공·실패 요청의 `request_log` 적재
- 에스컬레이션 호출 비용을 `origin='escalate'`로 구분해 적재
- GCE 백엔드와 Cloud Run 프론트엔드 반복 배포·롤백 스크립트
- 백엔드 JUnit/Mockito·JaCoCo 및 프론트 Vitest 테스트 기반

### 운영 배포

| 구성요소 | 현재 상태 | 리소스 |
|---|---|---|
| 프론트엔드 | 배포·접속 정상 | Cloud Run `llm-router-frontend`, `asia-northeast3` |
| 백엔드 | 컨테이너 실행 정상 | GCE `llm-router-workspace`, `asia-northeast3-a` |
| DB | PostgreSQL/pgvector 실행 정상 | GCE Docker `llmrouter-postgres` |
| 채팅 경로 | 정상 | Cloud Run → nginx → GCE `/api/chat` |
| 에스컬레이션 경로 | 정상 | Cloud Run → nginx → GCE `/api/escalate` |

운영 서버는 현재 커밋 `0bb0ea5` 기준이다. 이번 통합 커밋을 배포할 때는 백엔드보다 먼저
`infra/postgres/migrations/03-request-log-status.sql`을 적용해야 한다.

## 2. 완료된 기능

### 게이트웨이와 캐시

- [x] 질문 임베딩 → 캐시 조회 → miss 시 라우팅 → Claude 호출 → 캐시 저장
- [x] cache hit 시 라우터와 Claude 호출 생략
- [x] pgvector HNSW top-k 조회와 항목별 threshold 적용
- [x] 요청별 모델, 티어, 토큰, latency, similarity, 캐시 행 ID 기록
- [x] 실패 요청의 `status`, `error_type` 기록
- [x] Gateway 공유 시크릿과 운영 프로파일 fail-fast

### 에스컬레이션

- [x] 상위 모델 재요청 API와 프론트 버튼
- [x] 동일 요청·단계 중복 호출 방지
- [x] easy → medium → hard 체인 기록
- [x] 에스컬레이션 답변과 토큰·지연시간 저장
- [x] 최초 라우터 버전과 cache false-hit 근거 저장
- [x] 만족한 상위 답변으로 기존 캐시 갱신
- [x] 에스컬레이션 비용을 일반 요청 로그에 포함

### 테스트와 배포

- [x] 에스컬레이션 서비스·예약·체인·캐시 갱신 테스트
- [x] 캐시 hit/miss, 라우팅, LLM 호출, 실패 기록 테스트
- [x] 프론트 API·입력·모델 배지 테스트
- [x] 프론트 production build 절차
- [x] GCE 원격 작업공간 bootstrap
- [x] 프론트·백엔드 반복 배포와 백엔드 자동 롤백

## 3. 현재 한계

| 항목 | 현재 상태 | 영향 |
|---|---|---|
| 임베딩 | `stub-hash-v0` | 동일 문자열 외 의미 유사 질문의 캐시 품질을 평가할 수 없음 |
| 라우팅 | `stub-medium-v0` | 모든 cache miss가 medium으로 가므로 모델 절감·정확도 실험 불가 |
| FastAPI 연동 | 계약 및 팀 브랜치만 존재 | 실제 bge-large·ModernBERT 호출 미완료 |
| 분석 화면 | 미구현 | 캐시·라우팅 절감 효과를 발표 화면에서 설명하기 어려움 |
| 실험 실행·저장 | 미구현 | Always Opus 대비 재현 가능한 비교 결과가 없음 |
| 재학습 전달 | 설계 단계 | 에스컬레이션 1,000건 단위 배치 export·trigger 미구현 |
| 대화 유지 | React 메모리 상태 | 새로고침 시 대화가 사라짐 |

## 4. 다음 작업 우선순위

### P0 — 통합본 배포 완료

- [ ] 통합된 로컬 `main`을 원격에 push
- [ ] 운영 DB에 `03-request-log-status.sql` 적용
- [ ] 최신 `main`으로 GCE 백엔드 재배포
- [ ] Cloud Run 프론트 재배포
- [ ] `/api/chat`, `/api/escalate`, 실패 로그 smoke test
- [ ] `request_log.status/error_type/origin` 적재 확인

### P0 — 교수님 시연용 분석 기능

- [ ] `GET /api/analytics/summary?period=today|7d|all`
- [ ] 총 요청 수, 성공률, cache hit rate, LLM 호출 생략 횟수
- [ ] 모델·티어별 호출 비율과 입력·출력 토큰 합계
- [ ] 실제 비용과 Always-Opus 기준 비용 및 절감률
- [ ] 평균·P95 latency와 오류 유형별 발생 건수
- [ ] 프론트 `/analytics` 누적 대시보드와 상단 이동 버튼
- [ ] 원문 질문이나 API key 없이 집계값만 반환하는 보안 테스트

### P0 — 실제 FastAPI Router 통합

- [ ] `POST /router/embed`, `POST /router/classify`, `GET /health` 계약 확정
- [ ] Spring `FastApiRouterClient` 구현
- [ ] embed/classify timeout과 retry 범위 확정
- [ ] embed 실패 시 cache bypass, classify 실패 시 medium fallback
- [ ] `fallback_reason`, confidence, 실제 model version 기록
- [ ] 동일 fixture 기반 Spring↔FastAPI contract test
- [ ] bge-large와 ModernBERT artifact를 GCE에 배포

### P1 — 발표용 고정 실험

- [ ] easy/medium/hard 균형 질문과 정답 tier 100~200개 준비
- [ ] 유사 표현, 무관 질문, hard negative 포함
- [ ] Always Opus / Router Only / Router + Semantic Cache 비교 실행
- [ ] 총 비용, 토큰, 평균·P95 latency, cache hit rate 비교
- [ ] routing confusion matrix와 under-routing 비율 산출
- [ ] false cache hit과 threshold별 품질 산출
- [ ] 데이터셋·모델·threshold·가격표 버전을 포함한 immutable 결과 저장
- [ ] 프론트 `/experiments/:runId` 결과 조회 화면

### P1 — 피드백 배치와 재학습 연동

- [ ] backend·FastAPI 공통 `feedback-export-v0.1` 계약 확정
- [ ] 만족한 최종 에스컬레이션만 선별하고 중복·PII 처리
- [ ] `manifest.json`과 `events.parquet` 생성
- [ ] GCS immutable batch 업로드와 checksum 검증
- [ ] watermark/batchId 기반 멱등성 보장
- [ ] 유효 로그 1,000건 또는 최대 대기 기간 기준 batch 생성
- [ ] FastAPI 데이터 처리 서버에 batch URI 전달
- [ ] 초기에는 사람 승인 후 Kubeflow 재학습 실행

### P2 — 프론트와 운영 마무리

- [ ] 대화 `localStorage` 유지
- [ ] API 실패 재시도와 세분화된 로딩 상태
- [ ] 모바일 화면 점검
- [ ] Cloud Billing budget alert 확인
- [ ] npm audit 영향 분석 후 안전한 범위만 업데이트
- [ ] JWT 또는 사용자 인증 도입 여부 결정
- [ ] GCE 직접 노출 축소와 TLS 적용 검토

## 5. 발표 시나리오

1. 쉬운 질문에서 낮은 티어 모델 선택을 보여준다.
2. 어려운 질문에서 상위 모델 선택을 보여준다.
3. 불만족 버튼으로 에스컬레이션 경로와 개선된 답변을 보여준다.
4. 같은 질문과 유사 질문을 다시 보내 cache hit, 토큰 0, latency 감소를 보여준다.
5. `/analytics`에서 누적 cache hit rate, 모델 분포, 비용·토큰 절감을 보여준다.
6. `/experiments/:runId`에서 Always Opus 대비 라우팅·캐시 효과를 고정 결과로 비교한다.
7. confusion matrix와 under-routing을 함께 보여 절감률만 높인 결과가 아님을 설명한다.

핵심 설명은 다음과 같다.

> 라우팅은 요청마다 필요한 수준의 모델을 선택해 모델 단가를 낮추고, 시맨틱 캐시는
> 충분히 유사한 기존 답변을 재사용해 LLM 호출 자체와 토큰 비용을 제거한다.

## 6. 작업 규칙

- 기능 변경은 실패 테스트 작성 → 최소 구현 → 전체 관련 테스트 순으로 진행한다.
- DB 초기 스키마와 운영 마이그레이션은 항상 함께 수정한다.
- 스키마 마이그레이션은 백엔드 컨테이너보다 먼저 적용한다.
- 비용 실험에는 model ID, 가격표 버전, router/embed model version을 함께 저장한다.
- 공개 분석 API는 원문 질문과 민감정보를 반환하지 않는다.
- `main` 직접 push 대신 기능 브랜치와 PR을 기본으로 사용한다.
