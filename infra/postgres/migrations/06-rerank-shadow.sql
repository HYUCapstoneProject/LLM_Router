-- 06: 리랭커 shadow mode 기록
-- 새 DB는 init/01-schema.sql에 이미 반영되어 있다.
--
-- shadow mode 는 점수만 남기고 캐시 판정은 바꾸지 않는다.
-- 58쌍 스크리닝에서 사전 gate(false hit 0 + recall 90%)를 통과한 모델이 없었으므로,
-- 실제 트래픽에서 점수 분포를 먼저 모은 뒤에 임계값을 고정한다.

BEGIN;

ALTER TABLE request_log
    -- 캐시 히트 후보에 대한 cross-encoder 점수. 미호출이면 NULL.
    ADD COLUMN IF NOT EXISTS rerank_score REAL,
    -- 점수는 이 버전에서만 의미가 있다. 값이 바뀌면 이전 점수와 비교하면 안 되고
    -- 임계값도 다시 측정해야 한다.
    ADD COLUMN IF NOT EXISTS rerank_model_version VARCHAR(64),
    -- 리랭커를 실제로 적용했다면 이 히트를 막았을지. shadow 평가의 핵심 열이다.
    ADD COLUMN IF NOT EXISTS rerank_would_reject BOOLEAN;

COMMIT;
