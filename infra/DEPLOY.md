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
gcloud compute instances create llm-gateway \
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
gcloud compute ssh llm-gateway --zone=asia-northeast3-a
```

```bash
# PostgreSQL + pgvector (게이트웨이와 같은 VM 에 공동 배치)
docker run -d --name llmrouter-postgres --restart unless-stopped \
  -e POSTGRES_DB=llmrouter -e POSTGRES_USER=llmrouter \
  -e POSTGRES_PASSWORD=<DB_PASSWORD> \
  -p 5432:5432 -v pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

# 스키마 적용 (infra/postgres/init/01-schema.sql 을 VM 으로 복사한 뒤)
docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter < 01-schema.sql

# 게이트웨이
docker run -d --name llm-gateway --restart unless-stopped \
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
> 컨테이너가 바로 죽는다면 `docker logs llm-gateway` 로 확인할 것.

### 2-5. 확인

```bash
curl -i http://<VM_EXTERNAL_IP>:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"test","conversationId":"t1"}'
# → 401 GATEWAY_KEY_INVALID 이어야 정상 (키 없이 호출했으므로)
```

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
- [ ] `request_log` 에 행이 쌓이는지 확인
- [ ] `GATEWAY_KEY` 없이 기동 시도 → 컨테이너가 죽는지 확인 (fail-fast 동작 검증)

---

## 5. 중지 (크레딧 절약)

```bash
gcloud compute instances stop llm-gateway --zone=asia-northeast3-a
```

Cloud Run 은 scale-to-zero 라 접속이 없으면 과금되지 않는다. **GCE 만 끄면 된다.**

---

## 미해결

- [ ] JWT 인증 — 현재 `SecurityConfig` 는 permitAll 이고 공유 시크릿으로만 막고 있다
- [ ] HTTPS — 백엔드가 평문 HTTP 다. 공유 시크릿이 네트워크에 노출된다.
      Cloud Run→GCE 구간이므로 데모 수준에서는 감수하나, 개선 시 로드밸런서 또는 Cloud Run 사이드카 검토
- [ ] FastAPI 라우터 서빙 배포 (팀원 담당, M4)
