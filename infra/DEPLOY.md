# 배포 절차 (프론트엔드 / 게이트웨이)

Notion `07 Infra & Runbook` 의 Deploy Gateway / Deploy Frontend 항목에 대응한다.

```
브라우저
   │
   ▼
Cloud Run (nginx + React)          ← 공개. 인증 없음
   │  /api/*  +  X-Gateway-Key
   ▼
GCE VM (Spring Gateway + PostgreSQL/pgvector)   ← 키 없으면 거부
   │
   ▼
Anthropic Claude API
```

프론트 nginx 가 `/api` 를 프록시하므로 **브라우저는 Cloud Run 도메인만 본다.**
그래서 CORS 설정이 필요 없고 백엔드 IP 도 노출되지 않는다.

---

## 0. 사전 준비 (반드시 먼저)

- [ ] **예산 알림 설정** — 크레딧의 50 / 80 / 100 % 지점. GCE 는 켜둔 시간만큼 계속 과금된다.
- [ ] gcloud CLI 설치 후 인증

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com compute.googleapis.com
```

- [ ] 공유 시크릿 생성 (한 번만, 안전한 곳에 보관)

```bash
openssl rand -hex 32
```

> ⚠️ **비용 주의**: e2-standard-2 를 24시간 켜두면 약 \$49/월이다.
> 게이트웨이와 라우터 두 대면 약 \$98/월이고, \$300 크레딧으로 GKE 학습까지 해야 한다.
> **개발 중에는 쓸 때만 켜고 끄는 것**이 크레딧 관리의 핵심이다.

---

## 1. Artifact Registry

```bash
gcloud artifacts repositories create llm-router \
  --repository-format=docker --location=asia-northeast3

gcloud auth configure-docker asia-northeast3-docker.pkg.dev
```

---

## 2. 백엔드 (GCE) — 프론트보다 먼저

프론트가 백엔드 주소를 알아야 하므로 순서가 고정된다.

### 2-1. 이미지 빌드/푸시

```bash
cd backend
REPO=asia-northeast3-docker.pkg.dev/<PROJECT_ID>/llm-router
docker build -t $REPO/gateway:v1 .
docker push $REPO/gateway:v1
```

### 2-2. VM 생성

```bash
gcloud compute instances create llm-router-workspace \
  --zone=asia-northeast3-a \
  --machine-type=e2-standard-2 \
  --image-family=cos-stable --image-project=cos-cloud \
  --boot-disk-size=30GB \
  --tags=llm-gateway
```

### 2-3. 방화벽

8080 을 공개하되, 실제 접근 제어는 `X-Gateway-Key` 로 한다.
Cloud Run 은 고정 egress IP 가 없어 IP 기반 제한이 어렵기 때문이다.

```bash
gcloud compute firewall-rules create allow-gateway-8080 \
  --allow=tcp:8080 --target-tags=llm-gateway
```

### 2-4. VM 안에서 기동

```bash
gcloud compute ssh llm-router-workspace --zone=asia-northeast3-a
```

```bash
# PostgreSQL + pgvector (게이트웨이와 같은 VM 에 공동 배치)
docker run -d --name llmrouter-postgres --restart unless-stopped \
  -e POSTGRES_DB=llmrouter -e POSTGRES_USER=llmrouter \
  -e POSTGRES_PASSWORD=<DB_PASSWORD> \
  -p 5432:5432 -v pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

# 스키마 적용 (infra/postgres/init/01-schema.sql 을 VM 으로 복사한 뒤)
#
# 신규 DB 는 이 파일 하나면 최신 상태가 된다.
# infra/postgres/migrations/ 는 이미 돌고 있는 DB 를 따라잡게 하는 용도라 여기서는 필요 없다.
docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter < 01-schema.sql

# 게이트웨이
docker run -d --name llm-router-backend --restart unless-stopped \
  --network host \
  -e SPRING_PROFILES_ACTIVE=prod \
  -e ANTHROPIC_API_KEY=<KEY> \
  -e POSTGRES_PASSWORD=<DB_PASSWORD> \
  -e GATEWAY_KEY=<공유 시크릿> \
  $REPO/gateway:v1
```

> `SPRING_PROFILES_ACTIVE=prod` 를 반드시 넣는다.
> 이 프로파일에서 `GATEWAY_KEY` 가 비어 있으면 **기동 자체가 실패한다.**
> 키 미설정은 "검증 통과"로 동작하므로(fail-open), 이 장치가 없으면
> 환경변수를 빠뜨렸을 때 경고 없이 8080 이 무방비로 열린다.
>
> 컨테이너가 바로 죽는다면 `docker logs llm-router-backend` 로 확인할 것.

### 2-5. 확인

```bash
curl -i http://<VM_EXTERNAL_IP>:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"test","conversationId":"t1"}'
# → 401 GATEWAY_KEY_INVALID 이어야 정상 (키 없이 호출했으므로)
```

---

## 2-6. 재배포 — 스키마 마이그레이션이 있을 때

> ⚠️ **마이그레이션을 먼저 적용하고 컨테이너를 교체한다. 순서가 바뀌면 안 된다.**
>
> 새 코드는 새 컬럼에 INSERT 하는데 DB 에 그 컬럼이 없으면 매 요청마다 적재가 실패한다.
> `RequestLogService` 는 적재 실패를 의도적으로 삼키므로 **응답은 정상으로 보이고
> 로그만 조용히 비어간다.** 비용 그래프의 원천 데이터가 사라지는 것을
> 나중에야 알아차리게 된다.

마이그레이션 파일은 `infra/postgres/migrations/` 에 번호순으로 쌓인다.
전부 `IF NOT EXISTS` 라 여러 번 돌려도 안전하니, 어디까지 적용했는지 모르겠으면
처음부터 다 돌리면 된다.

```bash
# 1. VM 으로 복사
gcloud compute scp infra/postgres/migrations/*.sql \
  llm-router-workspace:/tmp/ --zone=asia-northeast3-a

# 2. VM 안에서 적용 (컨테이너 교체 전!)
gcloud compute ssh llm-router-workspace --zone=asia-northeast3-a
for f in /tmp/0*.sql; do
  echo "--- $f"
  docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter -v ON_ERROR_STOP=1 < "$f"
done

# 3. 그 다음에 게이트웨이 컨테이너 교체
```

적용 여부는 컬럼 존재로 확인한다.

```bash
docker exec llmrouter-postgres psql -U llmrouter -d llmrouter -c '\d request_log' \
  | grep -E 'conversation_id|cache_entry_id|origin|parent_request_id'
```

> 배포 스크립트(`scripts/deploy-backend.sh`, PR #17)는 **마이그레이션을 돌리지 않는다.**
> 스키마 변경이 포함된 배포에서는 스크립트를 실행하기 전에 위 절차를 손으로 먼저 해야 한다.

---

## 3. 프론트엔드 (Cloud Run)

```bash
cd frontend
docker build -t $REPO/frontend:v1 .
docker push $REPO/frontend:v1

gcloud run deploy llm-router-frontend \
  --image=$REPO/frontend:v1 \
  --region=asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars="BACKEND_URL=http://<VM_EXTERNAL_IP>:8080,GATEWAY_KEY=<공유 시크릿>"
```

`PORT` 는 Cloud Run 이 자동 주입하므로 지정하지 않는다.

---

## 4. 점검

- [ ] Cloud Run URL 접속 → 채팅 UI 표시
- [ ] 질문 전송 → 답변 + 모델 배지 + 토큰 표시
- [ ] 같은 질문 재전송 → 캐시 히트 (토큰 0, 지연시간 급감)
- [ ] VM 8080 직접 호출 → 401
- [ ] 빠르게 연속 요청 → 429 (nginx rate limit, 20r/m)
- [ ] `request_log` 에 행이 쌓이는지 확인 — 특히 `conversation_id` / `cache_entry_id` / `origin`
      이 채워지는지. 비어 있으면 마이그레이션을 빠뜨린 것이다 (§2-6)
- [ ] 답변 아래 `↑ Opus로 다시 답변` 버튼 → 클릭 → 답변 교체 + `↑ Sonnet → Opus` 경로 표시
- [ ] `escalation_log` 에 행이 쌓이는지 확인 (재학습 정정 라벨)
- [ ] 에스컬레이션 후 `request_log` 에 `origin='escalate'` 행이 생기는지 —
      없으면 비용 집계가 에스컬레이션을 빠뜨린다
- [ ] `GATEWAY_KEY` 없이 기동 시도 → 컨테이너가 죽는지 확인 (fail-fast 동작 검증)

---

## 5. 중지 (크레딧 절약)

```bash
gcloud compute instances stop llm-router-workspace --zone=asia-northeast3-a
```

Cloud Run 은 scale-to-zero 라 접속이 없으면 과금되지 않는다. **GCE 만 끄면 된다.**

---

## 6. 문제 해결

### 에스컬레이션이 계속 409 를 낸다

`/api/escalate` 는 LLM 을 부르기 전에 `escalation_log` 에 자리를 예약한다
(`response IS NULL` 인 행). 프로세스가 LLM 호출 도중 죽으면 그 예약이 남는다.

`llm.escalation.reservation-timeout`(기본 15분)이 지나면 다음 요청이 알아서 인수하므로
보통은 기다리면 된다. Anthropic SDK 요청 타임아웃(기본 10분)보다 길게 잡아야
"느리지만 살아 있는" 호출을 가로채지 않는다. 급하면 직접 지운다.

```sql
-- 남아 있는 예약 확인
SELECT request_id, escalation_seq, final_tier, created_at
FROM escalation_log WHERE response IS NULL;

-- 지우기 (완료된 행은 response 가 있으므로 이 조건에 안 걸린다)
DELETE FROM escalation_log WHERE response IS NULL AND created_at < now() - interval '15 minutes';
```

### 답변은 나오는데 request_log 가 비어 있다

마이그레이션 누락이 가장 흔한 원인이다. §2-6 참조.
`RequestLogService` 가 적재 실패를 삼키므로 사용자 화면에는 아무 이상이 없다.
`docker logs llm-router-backend | grep 'request_log 적재 실패'` 로 확인한다.

---

## 7. main 머지 후 반복 배포

현재 운영 리소스는 다음과 같다.

- 프론트엔드: Cloud Run `llm-router-frontend` (`asia-northeast3`)
- 백엔드 VM: GCE `llm-router-workspace` (`asia-northeast3-a`)
- 백엔드 컨테이너: `llm-router-backend`
- 고정 외부 IP: `llm-router-workspace-ip`

Cloud Run 은 VPC 직접 연결로 VM 내부 IP를 사용한다. `BACKEND_URL`, Secret Manager의
`GATEWAY_KEY`, VPC 설정은 기존 Cloud Run 서비스 설정에 유지되므로 반복 배포 때 값을
명령줄에 다시 쓰지 않는다.

배포 전 반드시 최신 `main`과 깨끗한 작업 트리를 준비한다.

```bash
git switch main
git pull --ff-only origin main
```

프론트만 바뀌었으면 다음 명령으로 빌드 검사 후 Cloud Run에 배포한다.

```bash
./scripts/deploy-frontend.sh
```

백엔드만 바뀌었으면 다음 명령으로 테스트 후 GCE 컨테이너를 교체한다.

```bash
./scripts/deploy-backend.sh
```

> 스키마 변경이 포함된 백엔드 배포는 스크립트를 실행하기 전에 반드시 §2-6의
> 마이그레이션을 먼저 적용한다. 현재 배포 스크립트는 DB 마이그레이션을 실행하지 않는다.

백엔드 스크립트는 현재 컨테이너의 환경변수를 VM 내부 임시 파일로 승계하며 값을
터미널에 출력하지 않는다. 새 컨테이너가 60초 안에 `401 GATEWAY_KEY_INVALID` 상태로
응답하지 않으면 직전 컨테이너로 자동 롤백한다. 성공 시 직전 컨테이너는
`llm-router-backend-previous` 이름으로 정지 상태를 유지한다.

프론트와 백엔드가 함께 바뀌었으면 백엔드를 먼저 배포·검증한 뒤 프론트를 배포한다.
다른 프로젝트나 리소스를 사용할 때는 환경변수로 기본값을 덮어쓸 수 있다.

```bash
PROJECT_ID=<PROJECT_ID> REGION=<REGION> ./scripts/deploy-frontend.sh
PROJECT_ID=<PROJECT_ID> ZONE=<ZONE> BACKEND_INSTANCE=<VM_NAME> ./scripts/deploy-backend.sh
```

---

## 미해결

- [ ] JWT 인증 — 현재 `SecurityConfig` 는 permitAll 이고 공유 시크릿으로만 막고 있다
- [ ] HTTPS — 백엔드가 평문 HTTP 다. 공유 시크릿이 네트워크에 노출된다.
      Cloud Run→GCE 구간이므로 데모 수준에서는 감수하나, 개선 시 로드밸런서 또는 Cloud Run 사이드카 검토
- [ ] FastAPI 라우터 서빙 배포 (팀원 담당, M4)
