-- 09: R0/R1의 라우팅 정책과 실제 후보군을 요청 시점 그대로 보존한다.
-- v2 사용 전에 적용한다. 기존 행은 소급 추정하지 않고 NULL로 둔다.
-- docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter < 09-routing-decision.sql
BEGIN;
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS routing_decision JSONB;
COMMIT;
