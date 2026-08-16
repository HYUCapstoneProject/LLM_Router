-- 04: FastAPI 라우터 confidence 기록
-- 새 DB는 init/01-schema.sql에 이미 반영되어 있다.

BEGIN;

ALTER TABLE request_log
    ADD COLUMN IF NOT EXISTS router_confidence REAL;

COMMIT;
