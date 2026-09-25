-- 10: Atomic feedback claims, immutable requests and activation-aware polling.
-- Apply before deploying the common Spring code. No request payload is invented for old batches.
BEGIN;
ALTER TABLE feedback_batch ADD COLUMN IF NOT EXISTS request_payload JSONB;
ALTER TABLE feedback_batch ADD COLUMN IF NOT EXISTS polling_complete BOOLEAN NOT NULL DEFAULT false;

-- A pre-migration unaccepted batch might already have reached FastAPI. Preserve event
-- ownership and require reconciliation rather than submitting a newly reconstructed body.
UPDATE feedback_batch
   SET pipeline_status = 'unknown', failure_code = 'LEGACY_PAYLOAD_UNAVAILABLE',
       retryable = false, polling_complete = true
 WHERE pipeline_run_id IS NULL AND pipeline_status = 'submitted' AND request_payload IS NULL;

-- Existing accepted jobs are re-polled once, including prior stale/succeeded snapshots.
DROP INDEX IF EXISTS idx_feedback_batch_pending;
CREATE INDEX idx_feedback_batch_pending ON feedback_batch (last_polled_at NULLS FIRST)
    WHERE polling_complete = false;
COMMIT;
