# ── 1단계: 빌드 ──────────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── 2단계: 서빙 ──────────────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY --from=build /build/dist /usr/share/nginx/html

# templates/ 에 넣으면 컨테이너 시작 시 envsubst 로 환경변수가 치환된다.
# Cloud Run 은 리스닝 포트를 $PORT 로 주입하므로 이 방식이 필요하다.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# 기본값. 배포 시 Cloud Run 환경변수로 덮어쓴다.
ENV PORT=8080
ENV BACKEND_URL=http://localhost:8080
ENV GATEWAY_KEY=""
ENV NGINX_PROXY_SEND_TIMEOUT=600s
ENV NGINX_PROXY_READ_TIMEOUT=600s

EXPOSE 8080
