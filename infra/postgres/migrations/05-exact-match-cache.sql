-- 05: exact match 캐시 계층
-- 새 DB는 init/01-schema.sql에 이미 반영되어 있다.
--
-- 완전히 동일한 질문이 다시 오면 임베딩 호출 자체를 건너뛴다.
-- FastAPI가 us-west1, 게이트웨이가 asia-northeast3라 embed 왕복에만 약 120ms가 든다.
--
-- 정규화(공백 정리)와 해시 계산은 Java(QuestionHash)에서만 한다.
-- SQL에서도 정규화하면 두 구현이 어긋났을 때 인덱스 조회가 조용히 빗나간다.

BEGIN;

-- NULL 허용: 이 마이그레이션 이전에 쌓인 행은 해시가 없다.
-- 그 행들은 exact match 대상이 아니고 기존 시맨틱 경로로만 히트한다.
ALTER TABLE cache_entry
    ADD COLUMN IF NOT EXISTS question_hash CHAR(64);

CREATE INDEX IF NOT EXISTS idx_cache_entry_question_hash
    ON cache_entry (question_hash);

-- exact / semantic 히트를 나눠 기록한다.
-- 두 값을 합친 히트율은 "같은 질문을 다시 쳤다"까지 포함하므로
-- 시맨틱 캐시의 성능으로 제시하면 과대평가가 된다.
ALTER TABLE request_log
    ADD COLUMN IF NOT EXISTS cache_match_type VARCHAR(10);   -- exact | semantic | NULL(미스)

COMMIT;
