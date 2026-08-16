-- 03: 요청 성공·실패 상태 기록
-- 기존 운영 DB에 request_log.status/error_type을 추가한다.
-- 새 DB는 init/01-schema.sql에 이미 반영되어 있다.

BEGIN;

ALTER TABLE request_log
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'success';

ALTER TABLE request_log
    ADD COLUMN IF NOT EXISTS error_type VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_request_log_status ON request_log (status);

COMMIT;
