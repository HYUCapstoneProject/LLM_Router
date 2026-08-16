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
- FastAPI bge-large 임베딩·ModernBERT 라우팅 HTTP 연동과 장애 폴백
- `POST /api/escalate` 재요청, 중복 과금 방지, 체인 기록, 캐시 답변 갱신
- 성공·실패 요청의 `request_log` 적재
- 에스컬레이션 호출 비용을 `origin='escalate'`로 구분해 적재
- GCE 백엔드와 Cloud Run 프론트엔드 반복 배포·롤백 스크립트
- 백엔드 JUnit/Mockito·JaCoCo 및 프론트 Vitest 테스트 기반

### 운영 배포

| 구성요소 | 현재 상태 | 리소스 |
|---|---|---|
| 프론트엔드 | 배포·접속 정상 | Cloud Run `llm-router-frontend`, `asia-northeast3` |
| 백엔드 | FastAPI 연동 컨테이너 실행 정상 | GCE `llm-router-workspace`, `asia-northeast3-a` |
| DB | PostgreSQL/pgvector 실행 정상 | GCE Docker `llmrouter-postgres` |
| 채팅 경로 | 정상 | Cloud Run → nginx → GCE `/api/chat` |
| 에스컬레이션 경로 | 정상 | Cloud Run → nginx → GCE `/api/escalate` |
| FastAPI Router | 모델 로드·계약 호출 정상 | GCE `llm-router-serving`, `10.138.0.2:8000` |

운영 Spring 서버는 현재 커밋 `ff8b690` 기준이며
`infra/postgres/migrations/04-router-confidence.sql`까지 적용됐다.

## 2. 완료된 기능

### 게이트웨이와 캐시

- [x] 질문 임베딩 → 캐시 조회 → miss 시 라우팅 → Claude 호출 → 캐시 저장
- [x] cache hit 시 라우터와 Claude 호출 생략
- [x] pgvector HNSW top-k 조회와 항목별 threshold 적용
- [x] 요청별 모델, 티어, 토큰, latency, similarity, 캐시 행 ID 기록
- [x] 실패 요청의 `status`, `error_type` 기록
- [x] Gateway 공유 시크릿과 운영 프로파일 fail-fast
- [x] FastAPI Bearer 인증·`X-Request-ID` 전파·커넥션 재사용
- [x] 임베딩 실패 시 캐시 우회, 분류 실패 시 medium 폴백
- [x] 요청별 실제 모델 버전·confidence·fallback reason 기록

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
| FastAPI 계약 회귀 | Mock HTTP 계약 테스트와 실서버 smoke test 완료, 공통 fixture 미구현 | 양쪽 DTO가 따로 바뀌면 drift 가능 |
| 분석 화면 | 미구현 | 캐시·라우팅 절감 효과를 발표 화면에서 설명하기 어려움 |
| 실험 실행·저장 | 미구현 | Always Opus 대비 재현 가능한 비교 결과가 없음 |
| 재학습 전달 | 설계 단계 | 에스컬레이션 1,000건 단위 배치 export·trigger 미구현 |
| 대화 유지 | React 메모리 상태 | 새로고침 시 대화가 사라짐 |

## 4. 다음 작업 우선순위

### P0 — 통합본 배포 완료

- [x] 통합된 로컬 `main`을 원격에 push
- [x] 운영 DB에 `03-request-log-status.sql` 적용
- [x] 최신 `main`으로 GCE 백엔드 재배포
- [x] Cloud Run 프론트 재배포
- [x] `/api/chat`, `/api/escalate`, 실패 로그 smoke test
- [x] `request_log.status/error_type/origin` 적재 확인

### P0 — 교수님 시연용 분석 기능

- [x] `GET /api/analytics/summary?period=today|7d|all`
- [x] 총 요청 수, 성공률, cache hit rate, LLM 호출 생략 횟수
- [x] 모델·티어별 호출 비율과 입력·출력 토큰 합계
- [x] 실제 비용과 Always-Opus 기준 비용 및 절감률
      → **캐시 히트 행은 제외하고 계산한다.** 토큰이 0 이라 "Opus 였다면"을 만들 근거가
      없다. 캐시 효과는 `llmCallsSkipped` 로 따로 보고하고, 응답의 `cost.note` 에
      "동일 토큰 기준 단가 차이"임을 명시해 화면에 그대로 띄운다.
- [x] 평균 latency (캐시 히트/미스 분리)와 오류 유형별 발생 건수
- [ ] P95 latency — 중간 시연 범위에서 제외. 표본이 적어 의미가 없다
- [x] 프론트 누적 대시보드와 상단 이동 버튼
      → 라우터 라이브러리가 없어 `/analytics` 경로 대신 `App.tsx` 의 view 토글로 구현
- [x] nginx rate limit 분리 — 지표 새로고침이 채팅 요청 한도를 먹지 않게 zone 을 나눔
- [ ] 원문 질문이나 API key 없이 집계값만 반환하는 보안 테스트
      → DTO 에는 질문 필드가 없다. 응답 본문 검증 테스트는 미작성
- [ ] **실 DB smoke test 미실행** — 집계 SQL 을 살아 있는 Postgres 에 한 번 돌려봐야 한다.
      로컬 5432 가 내려가 있어 확인하지 못했다

### P0 — 실제 FastAPI Router 통합

- [x] `POST /router/embed`, `POST /router/classify`, `GET /health` 계약 확정
- [x] Spring `FastApiRouterClient` 구현
- [x] embed/classify timeout과 retry 범위 확정
- [x] embed 실패 시 cache bypass, classify 실패 시 medium fallback
- [x] `fallback_reason`, confidence, 실제 model version 기록
- [ ] 동일 fixture 기반 Spring↔FastAPI contract test
- [x] bge-large와 ModernBERT artifact를 GCE에 배포
- [x] 운영 DB에 `04-router-confidence.sql` 적용 후 Spring 백엔드 재배포

### P1 — 발표용 고정 실험 (A: 우리 시스템 ablation)

우리 시스템을 실제로 돌려 얻는 숫자다. Claude 호출이 발생한다.

- [ ] easy/medium/hard 균형 질문과 정답 tier 100~200개 준비
- [ ] 유사 표현, 무관 질문, hard negative 포함
- [ ] Always Opus / Router Only / Router + Semantic Cache 비교 실행
- [ ] 총 비용, 토큰, 평균·P95 latency, cache hit rate 비교
- [ ] routing confusion matrix와 under-routing 비율 산출
- [ ] false cache hit과 threshold별 품질 산출
- [ ] 데이터셋·모델·threshold·가격표 버전을 포함한 immutable 결과 저장
- [ ] 프론트 `/experiments/:runId` 결과 조회 화면

> 비용 추정: 200문항 × 3구성 ≈ 600회 호출. 문항당 입출력 500토큰 가정 시 약 \$5~8.

### P1 — 발표용 고정 실험 (B: 기존 기법 대비 baseline)

**교수님 지적에 대한 직접적인 답이다.** 2026-08-03 미팅에서 "토큰을 아끼는 기법이 이미
여럿 있는데 그 대비 성능을 비교하라"는 지적이 있었고, 근거는
Notion `06 Decision Log` 2026-08-03 및 `05 Experiment Log` EXP-004 에 있다.

위 A는 **우리 시스템의 ablation**이지 다른 기법과의 비교가 아니다. Haiku 가 Opus 보다
5배 싸므로 무작위로 라우팅해도 비용은 내려간다. Always-Opus 비교만으로는
"우리 라우터가 좋다"가 아니라 "싼 모델이 싸다"를 보여줄 뿐이다.

- [ ] **Random (동일 분포)** — 분류기가 학습한 게 있는지 증명하는 최소 조건
- [ ] **Rule-based (길이/토큰 기반 if문)** — ModernBERT 학습이 값어치를 하는지
- [ ] **Oracle router** — 상한선 대비 달성률. RouterBench §4.4 에 정의가 있다
- [ ] **always-Haiku / always-Sonnet / always-Opus** — 비용 하한선과 품질 손실
- [ ] **RouteLLM BERT 분류기** — 선행연구 대비 개선폭
- [ ] 결과를 **비용-정확도 파레토 곡선 한 장**으로 산출

**실제 Claude 호출이 필요 없다.** RouterBench 에 11개 모델의 정오답과 비용이 이미
들어 있으므로, 라우팅 정책 함수만 교체해 오프라인으로 전부 계산한다.
평가 스크립트를 처음 짤 때 **정책을 교체 가능한 형태로** 설계해야 한 번에 들어간다.

> ⚠️ RouteLLM 공개 체크포인트를 그대로 쓰면 안 된다. 학습 도메인이 달라
> (Chatbot Arena 선호 데이터 vs RouterBench solve rate) 불리하게 나오는데,
> 그것은 방법의 차이가 아니라 데이터의 차이다. 같은 RouterBench 라벨로 재학습해 비교한다.

> 비교 대상이 **아닌** 것: 프롬프트 압축(LLMLingua), 컨텍스트 편집, Batch 할인,
> 프롬프트 캐싱. 이들은 우리 라우터와 병행 가능한 직교 기법이므로
> Related Work 에서 그렇게 정리한다. 상용 게이트웨이(OpenRouter 등)는 라우팅 정책이
> 비공개라 같은 테스트셋의 정오답을 얻을 수 없어 정량 비교가 불가능하다.

담당: RouterBench 데이터와 평가 스크립트를 가진 팀원. 게이트웨이는 이 실험에 관여하지 않는다.

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

## 5. 시연 시나리오

### 5-0. 중간 시연 (지금 가능한 범위)

목표는 "어디까지 했고, 얼마나 진척됐고, 성능이 어느 정도인가" 세 가지다.
최종 발표(5-1)의 8단계 중 **1~5번만 구현되어 있다.** 6~8번은 없다고 먼저 말하고 시작한다.

#### 사전 준비 (시연 당일 아침)

1. **운영로그 초기화** — 지금 28행 중 11행이 stub 라우터가 만든 것이다.
   그대로 두면 "medium 62.5%"가 라우터 판단인 것처럼 보인다.
   ```sql
   TRUNCATE request_log, escalation_log, cache_entry RESTART IDENTITY;
   ```
2. **캐시 워밍** — 4단계에서 쓸 질문(`de1`)을 미리 한 번 던져 캐시에 넣어둔다.
3. **접속 확인** — Cloud Run URL 이 뜨는지, 첫 요청이 cold start 로 느리지 않은지.

#### 검증된 시연 질문 (2026-08-16 실측)

35문항을 `/router/classify` 에 직접 던져 확인했다. 즉흥 질문은 쓰지 않는다.

**언어는 영어로 한다.** 같은 질문의 한국어본을 함께 돌린 결과다.

| | easy 적중 | medium 적중 | hard 적중 | 전체 |
|---|---|---|---|---|
| 영어 | 6/7 | 0/6 | 3/6 | 47% |
| 한국어 | 2/7 | **6/6** | **0/6** | 42% |

한국어는 medium 은 잘 맞히지만 **hard 를 6/6 전부 easy 로 보낸다.** 시연의 하이라이트인
"어려운 질문 → Opus"가 한국어로는 절대 안 나온다. 게다가 confidence 가 0.37~0.60 에
몰려 있어 3-class 무작위(0.33)와 크게 다르지 않다. bge-large-**en** 과 RouterBench 가
영어 전용이라 예상된 결과다.

**1단계 (easy → Haiku) — 신뢰도 높음**

| id | confidence | 질문 |
|---|---|---|
| `de4` | **0.97** | Who wrote the play Romeo and Juliet? |
| `de3` | 0.93 | What does HTTP stand for? |
| `de5` | 0.93 | What is the chemical symbol for gold? |

**2단계 (hard → Opus) — 맞지만 confidence 가 낮다**

| id | confidence | 질문 |
|---|---|---|
| `dh1` | 0.64 | Design a rate limiter for a distributed API gateway … (20 servers, no SPOF) |
| `dh4` | 0.63 | Derive the amortized time complexity of union-find … |
| `dh3` | 0.56 | A database serves 40% write and 60% read traffic … |
| `h04` | 0.52 | Given a 10TB dataset that no longer fits in memory … |

**4단계 (캐시 히트) — 안전 확인됨**

```
de1   What is the capital of Australia?
de1p  Which city is the capital of Australia?     유사도 0.9730
```
이 질문셋 안에서 다른 어떤 쌍도 0.75 를 넘지 않는다. 오탐 위험 없음.

**medium 은 라이브로 보여줄 수 없다.** 영어 medium 후보 18개 중 1개만 medium 으로
분류됐고 그마저 confidence 0.37 이었다. 대신 **3단계 에스컬레이션이 Sonnet 을 띄운다** —
Haiku → Sonnet 경로로 세 모델이 모두 화면에 나온다. medium 붕괴는 숨기지 말고
아래 "먼저 인정할 것" 1번에서 측정치로 제시한다.

재현:
```
python experiments/probe_router.py --questions experiments/questions-demo-en.json
python experiments/probe_router.py --questions experiments/questions-demo-ko.json
python experiments/probe_router.py --questions experiments/questions-demo-en2.json
```

#### 진행 순서

| # | 화면 | 할 것 | 할 말 |
|---|---|---|---|
| 0 | — | 구조 한 장 | "게이트웨이가 캐시를 보고, 미스면 라우터가 티어를 정하고, 그 티어의 Claude 를 부릅니다" |
| 1 | 채팅 | 쉬운 질문 | 답변 아래 **Haiku 배지 + 토큰 + 지연** 을 짚는다 |
| 2 | 채팅 | 어려운 질문 | 같은 자리에 **Opus 배지**. 모델이 바뀐 게 눈에 보이는 지점 |
| 3 | 채팅 | 2번 답변에 불만족 버튼 | 에스컬레이션 경로 `Haiku → Sonnet` 표시. "이 기록이 재학습 정정 라벨이 됩니다" |
| 4 | 채팅 | 1번과 같은 질문 재전송 | **⚡ Cache Hit / LLM 호출 없음 / 지연 급감**. 여기가 가장 잘 보이는 장면 |
| 5 | 지표 | 헤더 "지표 보기" | 누적 히트율·티어 분포·비용·지연. **"이건 방금 시연한 몇 건이라 통계가 아닙니다"를 먼저 말한다** |

#### 진척도를 말하는 방식

숫자로 말하는 편이 낫다.

- 게이트웨이·프론트·DB·배포 파이프라인: **동작함** (Cloud Run + GCE, 자동 롤백 포함)
- 라우터 연동: **동작함** (ModernBERT 실서버 호출)
- 캐시: **동작함** (pgvector HNSW, 임계값 튜닝 남음)
- 정량 평가·baseline 비교: **미착수** — 최종 발표 과제

#### 먼저 인정하고 갈 것 (질문받기 전에)

방어하려 하지 말고 측정한 결과를 그대로 내는 편이 유리하다.

1. **라우터가 easy 로 붕괴한다.** 영어 35문항에서 easy 29 / medium 2 / hard 4 로
   분류됐다. medium 후보 18개 중 17개가 easy 로 갔고, `Explain the CAP theorem …`
   같은 질문에 confidence 0.88 로 easy 를 확신했다. 원인은 학습 데이터
   (RouterBench)가 객관식 상식 문제에 치우친 것으로 본다 —
   **기획서 3-3 이 예측한 클래스 불균형 위험이 실제로 나타났다.**
   완화안: 클래스 가중 손실, 개방형 질문 데이터 보강, 결정 임계값 재조정.
2. **캐시 임계값이 아직 안전하지 않다.** 92문항 실측에서 paraphrase 최소 0.8621 <
   hard_negative 최대 0.9371 로 두 분포가 겹친다. 현재 0.80 은 오탐을 낸다.
   → cross-encoder 리랭커 도입을 팀원에게 요청해 둔 상태 (`experiments/RERANKER_REQUEST.md`)
3. **지연 숫자에 배포 아티팩트가 섞여 있다.** 라우터가 us-west1, 게이트웨이가
   asia-northeast3 라 왕복 120ms 가 매 요청에 붙는다. 같은 리전으로 옮기면 사라진다.

#### 예상 질문

| 질문 | 답 |
|---|---|
| "절감률이 왜 28%밖에 안 되나" | 캐시 히트는 토큰이 0 이라 단가 비교에서 제외했다. 캐시 효과는 생략된 호출 수로 따로 본다. 합치면 더 크지만 그건 두 개를 섞은 숫자다 |
| "그 비용 비교가 공정한가" | 아니다. 같은 토큰을 Opus 단가로 환산한 값이라 Opus 가 더 길게 답할 가능성은 반영이 안 됐다. 화면에도 그렇게 써 뒀다. 정확한 비교는 고정 실험에서 Opus 를 실제로 돌려 낸다 |
| "품질은 어떻게 보장하나" | 지금은 못 한다. under-routing 비율과 Oracle 대비 달성률이 최종 발표 과제다 |
| "기존 방법 대비 나은 점은" | 그 비교(파레토 곡선)가 남은 핵심 작업이다 |

### 5-1. 최종 발표 시나리오

| # | 화면 | 보여줄 것 | 예상 질문에 대한 방어 |
|---|---|---|---|
| 1 | 채팅 | 쉬운 질문 → 낮은 티어 선택 | |
| 2 | 채팅 | 어려운 질문 → 상위 모델 선택 | |
| 3 | 채팅 | 불만족 버튼 → 에스컬레이션 경로와 개선된 답변 | 재학습 정정 라벨이 여기서 나온다 |
| 4 | 채팅 | 같은/유사 질문 재전송 → cache hit, 토큰 0, latency 감소 | |
| 5 | `/analytics` | 누적 cache hit rate, 모델 분포, 비용·토큰 절감 | "실제로 쓸 때도 그러냐" |
| 6 | `/experiments/:runId` | Always Opus 대비 라우팅·캐시 효과 (고정 결과) | "재현 가능하냐" |
| 7 | `/experiments/:runId` | confusion matrix, under-routing 비율 | **"비용만 줄인 거 아니냐"** |
| 8 | 파레토 곡선 | Random·Rule-based·Oracle 대비 위치 | **"기존 기법 대비 뭐가 낫냐"** |

7·8번이 방어의 핵심이다.

**7번** — 라우터가 전부 Haiku 로 보내면 비용은 최저지만 품질이 무너진다.
under-routing 비율을 같이 보여야 절감률만 높인 결과가 아님이 증명된다.

**8번** — 교수님이 지적하신 부분이다. Oracle 을 상한, Random·always-Haiku 를 하한으로
두고 우리 점이 Rule-based 보다 **왼쪽 위**에 있어야 한다.
"Always-Opus 대비 3배 절감"보다 **"Oracle 성능의 N% 달성"** 이 설득력이 크다.

```
정확도
  ▲          ┌ Oracle (상한선)
  │        ● │
  │      ●───┘ Ours
  │    ●  Rule-based
  │  ●  Random          ● always-Opus
  │ ● always-Haiku
  └──────────────────────────▶ 비용
```

핵심 설명은 다음과 같다.

> 라우팅은 요청마다 필요한 수준의 모델을 선택해 모델 단가를 낮추고, 시맨틱 캐시는
> 충분히 유사한 기존 답변을 재사용해 LLM 호출 자체와 토큰 비용을 제거한다.
> 두 효과는 독립적이며, Oracle 대비 달성률로 라우터 자체의 품질을 별도로 검증한다.

## 6. 구현 설계 — 분석·실험 화면

발표 시나리오 5~8번을 만들기 위한 구체적인 설계다. 순서대로 진행한다.

### 6-1. `/analytics` — 누적 집계 (시나리오 5)

`request_log` 에 이미 데이터가 쌓이고 있으므로 집계 쿼리만 붙이면 된다. 가장 값이 싸다.

**API**

```
GET /api/analytics/summary?period=today|7d|all
```

```json
{
  "period": "7d",
  "totalRequests": 128,
  "successRate": 0.98,
  "cacheHitRate": 0.31,
  "llmCallsSkipped": 40,
  "byTier":  [{ "tier": "easy", "count": 52, "inputTokens": 6100, "outputTokens": 21000 }],
  "cost":    { "actual": 0.42, "alwaysOpus": 1.31, "savedRatio": 0.68 },
  "latency": { "avgMs": 1820, "p95Ms": 4400, "cacheHitAvgMs": 61 },
  "errors":  [{ "status": "llm_error", "count": 2 }]
}
```

**작업**

- [ ] `AnalyticsService` — `request_log` 집계 SQL (period 필터)
- [ ] Always-Opus 기준 비용 계산 — 같은 토큰 수를 Opus 단가로 환산
- [ ] 모델 단가를 `application.yml` 로 분리 (`llm.pricing.*`) — 가격 개정 시 코드 수정 없이 대응
- [ ] `AnalyticsController` — `GET /api/analytics/summary`
- [ ] 프론트 `/analytics` 페이지 + 상단 이동 버튼
- [ ] `origin='escalate'` 행을 비용에 포함하되 요청 수에서는 분리 집계
- [ ] 보안 테스트 — 원문 질문·API key 미반환, 집계값만

> ⚠️ Always-Opus 환산은 **실제 사용한 토큰 수**로 계산한다. 같은 질문을 Opus 에 보냈을 때
> 출력 토큰이 같다는 보장은 없으므로, 이 값은 "동일 토큰 기준 단가 차이"임을 화면에 명시한다.
> 정확한 비교는 6-2 의 고정 실험에서 실제로 Opus 를 돌려 얻는다.

### 6-2. `/experiments/:runId` — 고정 실험 (시나리오 6·7)

같은 질문셋을 여러 구성으로 돌려 재현 가능한 결과를 남긴다.

**스키마**

```sql
CREATE TABLE experiment_run (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL UNIQUE,
    config          VARCHAR(32) NOT NULL,   -- always_opus | router_only | router_cache
    dataset_version VARCHAR(64) NOT NULL,
    router_model_version VARCHAR(128),
    embed_model_version  VARCHAR(64),
    pricing_version VARCHAR(32) NOT NULL,
    thresholds      JSONB,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE TABLE experiment_result (
    id            BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES experiment_run(run_id),
    question_id   VARCHAR(64) NOT NULL,
    expected_tier VARCHAR(10) NOT NULL,   -- 정답 라벨
    actual_tier   VARCHAR(10),            -- 라우터 판정
    cache_hit     BOOLEAN NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms    INTEGER NOT NULL
);
```

**작업**

- [ ] 질문셋 파일 (`experiments/questions-v1.json`) — id, question, expectedTier
- [ ] `ExperimentRunner` — 구성별로 질문셋을 순회하며 실행
- [ ] `config=always_opus` 는 라우터를 건너뛰고 hard 로 고정
- [ ] `config=router_only` 는 캐시를 우회
- [ ] 실행 결과를 `experiment_result` 에 적재하고 `finished_at` 기록
- [ ] `GET /api/experiments/:runId` — 집계 + confusion matrix + under-routing 비율
- [ ] 프론트 `/experiments/:runId` 화면
- [ ] 실행은 수동 트리거로 제한 (실수로 반복 호출되면 크레딧이 녹는다)

> 실행 순서 주의: `router_cache` 를 먼저 돌리면 캐시가 채워져 뒤 구성이 오염된다.
> 구성마다 `cache_entry` 를 비우거나 실험 전용 스키마를 쓴다.

### 6-3. baseline 파레토 곡선 (시나리오 8)

**게이트웨이 작업이 아니다.** RouterBench 오프라인 계산이라 팀원 담당이며,
§4 P1(B) 에 항목이 있다. 게이트웨이는 결과 이미지나 CSV 를 받아
`/experiments` 화면에 함께 표시하는 정도만 맡는다.

- [ ] 팀원과 결과 전달 형식 합의 (CSV 컬럼: policy, cost, accuracy)
- [ ] `/experiments` 화면에 파레토 곡선 표시

### 우선순위

| 순서 | 항목 | 이유 |
|---|---|---|
| 1 | 6-1 `/analytics` | 데이터가 이미 있어 가장 싸고, 데모에서 바로 보인다 |
| 2 | 6-2 고정 실험 | 재현 가능한 숫자. 발표에서 빠질 수 없다 |
| 3 | 6-3 파레토 곡선 | 교수님 지적에 대한 답이지만 팀원 협업이 필요하다 |

## 7. 작업 규칙

- 기능 변경은 실패 테스트 작성 → 최소 구현 → 전체 관련 테스트 순으로 진행한다.
- DB 초기 스키마와 운영 마이그레이션은 항상 함께 수정한다.
- 스키마 마이그레이션은 백엔드 컨테이너보다 먼저 적용한다.
- 비용 실험에는 model ID, 가격표 버전, router/embed model version을 함께 저장한다.
- 공개 분석 API는 원문 질문과 민감정보를 반환하지 않는다.
- `main` 직접 push 대신 기능 브랜치와 PR을 기본으로 사용한다.
