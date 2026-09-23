#!/usr/bin/env bash
set -euo pipefail

archive_path="${1:-}"
commit_tag="${2:-}"
container_name="llm-router-backend"
previous_name="llm-router-backend-previous"
image_name="llm-router-backend:${commit_tag}"

if [[ ! "$archive_path" =~ ^/tmp/llm-router-backend-[A-Za-z0-9._-]+\.tar\.gz$ ]]; then
  echo "허용되지 않은 아카이브 경로입니다: $archive_path" >&2
  exit 1
fi

if [[ ! "$commit_tag" =~ ^[0-9a-f]{12}$ ]]; then
  echo "올바르지 않은 커밋 태그입니다: $commit_tag" >&2
  exit 1
fi

if [[ ! -f "$archive_path" ]]; then
  echo "배포 아카이브가 없습니다: $archive_path" >&2
  exit 1
fi

work_dir="$(mktemp -d /tmp/llm-router-backend-deploy-XXXXXX)"
env_file="$(mktemp /tmp/llm-router-backend-env-XXXXXX)"
router_env_file="${env_file}.router"

cleanup() {
  rm -rf "$work_dir"
  rm -f "$env_file" "$router_env_file" "$archive_path"
}
trap cleanup EXIT

chmod 600 "$env_file"
tar -xzf "$archive_path" -C "$work_dir"

echo "새 Docker 이미지 빌드: $image_name"
docker build -t "$image_name" "$work_dir/backend"

if ! docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "기존 $container_name 컨테이너가 없어 환경변수를 안전하게 승계할 수 없습니다." >&2
  exit 1
fi

# 환경변수 파일은 VM의 임시 파일에만 저장하며 출력하거나 로컬로 전송하지 않는다.
docker inspect "$container_name" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' > "$env_file"

# FastAPI 토큰은 저장소·배포 인자에 싣지 않고 VM에서 Secret Manager로 직접 읽는다.
# 기존 ROUTER_* 값은 제거하고 매 배포마다 최신 토큰과 내부 주소로 교체한다.
# 이 단계는 기존 컨테이너를 멈추기 전이라 권한/네트워크 실패 시 서비스에 영향이 없다.
grep -v -E '^(ROUTER_MODE|ROUTER_BASE_URL|ROUTER_API_TOKEN)=' "$env_file" > "$router_env_file"
router_api_token="$(gcloud secrets versions access latest \
  --secret="${ROUTER_TOKEN_SECRET:-llm-router-internal-api-token}")"
if [[ -z "$router_api_token" ]]; then
  echo "FastAPI 라우터 토큰을 Secret Manager에서 읽지 못했습니다." >&2
  exit 1
fi
printf 'ROUTER_MODE=fastapi\nROUTER_BASE_URL=%s\nROUTER_API_TOKEN=%s\n' \
  "${ROUTER_BASE_URL:-http://10.138.0.2:8000}" "$router_api_token" >> "$router_env_file"
mv "$router_env_file" "$env_file"
chmod 600 "$env_file"
unset router_api_token

if docker container inspect "$previous_name" >/dev/null 2>&1; then
  docker rm "$previous_name"
fi

docker stop "$container_name"
docker rename "$container_name" "$previous_name"

rollback() {
  echo "새 컨테이너 기동 실패. 이전 버전으로 롤백합니다." >&2
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  docker rename "$previous_name" "$container_name"
  docker start "$container_name"
}

if ! docker run -d \
  --name "$container_name" \
  --restart unless-stopped \
  --network host \
  --env-file "$env_file" \
  "$image_name"; then
  rollback
  exit 1
fi

healthy=false
for _ in $(seq 1 30); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" == "true" ]]; then
    status="$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/chat || true)"
    if [[ "$status" == "401" ]]; then
      healthy=true
      break
    fi
  fi
  sleep 2
done

if [[ "$healthy" != "true" ]]; then
  docker logs --tail 100 "$container_name" >&2 || true
  rollback
  exit 1
fi

echo "백엔드 배포 완료: $image_name"
echo "이전 컨테이너는 롤백용으로 $previous_name 이름에 정지 상태로 유지됩니다."
