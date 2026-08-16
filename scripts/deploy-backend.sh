#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-asia-northeast3-a}"
INSTANCE="${BACKEND_INSTANCE:-llm-router-workspace}"
COMMIT_TAG="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID가 설정되지 않았고 gcloud 기본 프로젝트도 없습니다." >&2
  exit 1
fi

current_branch="$(git -C "$REPO_ROOT" branch --show-current)"
if [[ "$current_branch" != "main" ]]; then
  echo "main 브랜치에서만 배포할 수 있습니다. 현재: $current_branch" >&2
  exit 1
fi

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "커밋되지 않은 변경이 있습니다. 작업 트리를 정리한 뒤 다시 실행하세요." >&2
  exit 1
fi

archive="$(mktemp "/tmp/llm-router-backend-${COMMIT_TAG}-XXXXXX.tar.gz")"
remote_archive="/tmp/$(basename "$archive")"
remote_script="/tmp/remote-deploy-backend.sh"

cleanup() {
  rm -f "$archive"
}
trap cleanup EXIT

echo "[1/4] 백엔드 테스트"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env GRADLE_USER_HOME=/tmp/gradle-home \
  --volume "$REPO_ROOT/backend:/workspace" \
  --workdir /workspace \
  eclipse-temurin:24-jdk \
  bash ./gradlew test --no-daemon

echo "[2/4] 배포 소스 패키징"
tar -C "$REPO_ROOT" \
  --exclude="backend/.gradle" \
  --exclude="backend/build" \
  -czf "$archive" backend

echo "[3/4] GCE로 소스와 배포 도우미 전송"
gcloud compute scp \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  "$archive" "$SCRIPT_DIR/remote-deploy-backend.sh" \
  "$INSTANCE:/tmp/"

echo "[4/4] GCE 백엔드 교체: llm-router-backend:$COMMIT_TAG"
gcloud compute ssh "$INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --command="bash '$remote_script' '$remote_archive' '$COMMIT_TAG'"
