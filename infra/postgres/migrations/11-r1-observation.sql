-- 11: Internal, explicitly assessed R1 outcomes. This table is never populated by chain completion.
BEGIN;
CREATE TABLE IF NOT EXISTS r1_observation (
    event_id VARCHAR(200) PRIMARY KEY,
    assessment_id VARCHAR(200) NOT NULL UNIQUE,
    request_id UUID NOT NULL,
    request_log_id BIGINT NOT NULL,
    root_request_id UUID NOT NULL,
    candidate_model_id VARCHAR(200) NOT NULL,
    policy_manifest_sha256 CHAR(64) NOT NULL,
    target_contract_version VARCHAR(200) NOT NULL,
    original_question_sha256 CHAR(64) NOT NULL,
    observation_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (request_id, candidate_model_id),
    CHECK ((observation_payload->>'schema_version' = 'r1-observation-v1') IS TRUE),
    CHECK ((jsonb_typeof(observation_payload->'success') = 'boolean') IS TRUE),
    CHECK ((observation_payload->'pii_redacted' = 'true'::jsonb) IS TRUE)
);
COMMIT;
