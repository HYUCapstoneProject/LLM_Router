#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast3}"
SERVICE="${FRONTEND_SERVICE:-llm-router-frontend}"

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

echo "[1/3] 프론트엔드 의존성 설치 및 빌드 검사"
npm --prefix "$REPO_ROOT/frontend" ci
npm --prefix "$REPO_ROOT/frontend" run build

echo "[2/3] Cloud Run 배포: $SERVICE ($PROJECT_ID/$REGION)"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source "$REPO_ROOT/frontend" \
  --allow-unauthenticated \
  --quiet

echo "[3/3] 배포 URL"
gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)'
