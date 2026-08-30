-- 07: 리랭커 재정렬 관측
-- 새 DB는 init/01-schema.sql에 이미 반영되어 있다.
--
-- 06 에서는 임계값을 통과한 최선 후보 1건만 리랭커에 넘겼다. 그래서 거부만
-- 관측할 수 있었고, 리랭커가 후보 사이 순위를 다시 매기는지는 잴 수 없었다.
-- top-k 를 넘기면서 그 값을 남긴다.

BEGIN;

ALTER TABLE request_log
    -- 리랭커가 점수를 매긴 후보 수. would_reorder 해석에 필요하다 —
    -- 후보가 1개면 재정렬은 정의상 일어날 수 없다.
    ADD COLUMN IF NOT EXISTS rerank_candidate_count INTEGER,
    -- 리랭커 1등이 코사인 1등과 달랐는지. 재정렬의 값어치를 재는 열이다.
    ADD COLUMN IF NOT EXISTS rerank_would_reorder BOOLEAN;

COMMIT;
