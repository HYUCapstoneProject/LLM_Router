#!/usr/bin/env bash
#
# 시연 당일 아침에 운영 데이터를 초기화하고 캐시를 예열한다.
#
# 왜 초기화가 필요한가: 개발 중 쌓인 행에는 stub 라우터가 만든 티어 분포와
# 같은 질문을 반복해서 만든 캐시 히트가 섞여 있다. 그대로 두고 지표 화면을 보여주면
# 라우터 성능도 캐시 히트율도 실제보다 좋게 보인다.
#
# GCE 워크스페이스 VM 안에서 실행한다.
#     gcloud compute scp scripts/reset-demo-data.sh llm-router-workspace:/tmp/ --zone=asia-northeast3-a
#     gcloud compute ssh llm-router-workspace --zone=asia-northeast3-a --command="bash /tmp/reset-demo-data.sh"
#
# 캐시 예열까지 하려면 --warm <프론트URL> 을 준다. Claude 를 1회 호출한다.
set -euo pipefail

CONTAINER=llmrouter-postgres
WARM_URL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --warm) WARM_URL="${2:?--warm 뒤에 프론트 URL 이 필요하다}"; shift 2 ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
  esac
done

psql() { docker exec -i "$CONTAINER" psql -U llmrouter -d llmrouter "$@"; }

echo "=== 지울 데이터 ==="
psql -c "
  SELECT 'request_log'    AS table, count(*) FROM request_log
  UNION ALL SELECT 'escalation_log', count(*) FROM escalation_log
  UNION ALL SELECT 'cache_entry',    count(*) FROM cache_entry;"

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="/tmp/llmrouter-predemo-${STAMP}.sql"
echo
echo "=== 백업 -> ${BACKUP} ==="
docker exec "$CONTAINER" pg_dump -U llmrouter -d llmrouter --data-only \
  -t request_log -t escalation_log -t cache_entry > "$BACKUP"
ls -la "$BACKUP"

echo
echo "=== 초기화 ==="
psql -v ON_ERROR_STOP=1 -c \
  "TRUNCATE request_log, escalation_log, cache_entry RESTART IDENTITY;"

echo
echo "=== 초기화 후 ==="
psql -c "
  SELECT 'request_log'    AS table, count(*) FROM request_log
  UNION ALL SELECT 'escalation_log', count(*) FROM escalation_log
  UNION ALL SELECT 'cache_entry',    count(*) FROM cache_entry;"

if [[ -n "$WARM_URL" ]]; then
  # 시연 4단계(같은 질문 재전송 -> exact 히트)를 위해 캐시에 한 건을 심어둔다.
  # 이 질문은 라우터 실측에서 easy 로 안정적으로 분류된 것이다(PLAN §5-0).
  WARM_Q="What is the capital of Australia?"
  echo
  echo "=== 캐시 예열: \"${WARM_Q}\" (Claude 1회 호출) ==="
  curl -sS -X POST "${WARM_URL%/}/api/chat" \
    -H 'Content-Type: application/json' \
    -d "{\"message\":\"${WARM_Q}\",\"conversationId\":\"demo-warmup\"}" \
    -o /tmp/warm.json -w '  HTTP %{http_code}  %{time_total}s\n'
  echo "  cacheHit=$(grep -o '\"cacheHit\":[a-z]*' /tmp/warm.json | cut -d: -f2)"
  rm -f /tmp/warm.json
  echo
  echo "  이제 같은 질문을 다시 보내면 exact 히트로 약 5ms 에 응답한다."
fi

echo
echo "복구가 필요하면:"
echo "  docker exec -i ${CONTAINER} psql -U llmrouter -d llmrouter < ${BACKUP}"
