-- 02: 에스컬레이션 기록
--
-- 이 파일은 "이미 돌고 있는" DB 에 적용한다.
-- init/01-schema.sql 은 컨테이너 데이터 볼륨이 비어 있을 때만 1회 실행되므로,
-- 운영 DB 나 기존 로컬 볼륨에는 이 마이그레이션을 직접 넣어야 한다.
--
--   docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter < 02-escalation.sql
--
-- 전부 IF NOT EXISTS 라 여러 번 돌려도 안전하다.
-- init/01-schema.sql 에도 같은 내용이 반영되어 있어야 한다 (새 볼륨용). 둘은 항상 같이 고친다.
--
-- 피드백 배치 내보내기(feedback_batch, exported_batch_id)는 별도 기능이라
-- 03-feedback-batch.sql 로 분리했다. 이 파일만 적용해도 에스컬레이션은 완전히 동작한다.
--
-- 배포 순서 주의: 새 백엔드 컨테이너를 올리기 "전에" 적용해야 한다.
-- conversation_id 컬럼이 없으면 모든 /api/chat 의 request_log INSERT 가 실패한다.

BEGIN;

-- ─────────────────────────────────────────────────────────────
-- 1. request_log — 에스컬레이션 비용 추적과 캐시 행 연결
-- ─────────────────────────────────────────────────────────────

-- ChatRequest.conversationId 는 지금까지 검증만 하고 버려졌다.
-- 피드백 이벤트 스키마가 이 값을 실어 보내므로 저장한다.
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64);

-- 이 요청이 "히트했거나 새로 만든" 캐시 행.
--
-- /api/escalate 가 불만족한 답이 어느 캐시 행에서 나왔는지 알아야
--   (1) escalation_log.cache_entry_id 를 채워 라우터 과소 라우팅과
--       캐시 false-hit 을 구분하고 (Decision Log 2026-08-03),
--   (2) 그 행을 정확히 지목해 갱신할 수 있다 (기획서 4-3).
-- 이 링크가 없으면 유사한 행을 새로 만들어 캐시가 오염된다.
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS cache_entry_id BIGINT
    REFERENCES cache_entry(id) ON DELETE SET NULL;

-- 에스컬레이션은 실제로 과금되는 LLM 호출이다.
--
-- RequestIdFilter 가 POST /api/escalate 에 "새" UUID 를 발급하므로,
-- 이 두 컬럼이 없으면 비용 집계 `SELECT sum(input_tokens) FROM request_log` 가
-- 에스컬레이션 호출을 통째로 빠뜨린다 — 그것도 라우터에 유리한 방향으로.
-- 최종 보고서의 "Router vs always-Opus" 숫자가 조용히 부풀려진다.
--
-- 따라서 에스컬레이션 호출도 정상 request_log 행으로 남기되,
-- origin 으로 구분해서 라우터 정확도 집계만 `WHERE origin = 'chat'` 을 붙인다.
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS parent_request_id UUID;
ALTER TABLE request_log ADD COLUMN IF NOT EXISTS origin VARCHAR(16) NOT NULL DEFAULT 'chat';

CREATE INDEX IF NOT EXISTS idx_request_log_parent ON request_log (parent_request_id);
CREATE INDEX IF NOT EXISTS idx_request_log_origin ON request_log (origin);

-- ─────────────────────────────────────────────────────────────
-- 2. escalation_log — 재학습 정정 라벨
-- ─────────────────────────────────────────────────────────────
--
-- 모델링: 에스컬레이션 "단계당 1행". request_id 는 계속 비유니크다.
--
--   seq 1: initial=easy, final=medium, path={haiku,sonnet}
--   seq 2: initial=easy, final=hard,   path={haiku,sonnet,opus}
--
-- initial_tier 는 항상 "라우터의 최초 판정"으로 고정한다(기존 컬럼 주석과 동일).
-- 직전 단계의 티어가 아니다.
-- escalation_path 는 누적 저장해서 한 행만 봐도 경로 전체를 알 수 있게 한다.

ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64);

ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS escalation_seq INTEGER NOT NULL DEFAULT 1;

-- 체인의 종결 표시.
--
-- 새 행은 true("사용자가 여기서 멈췄다")로 들어가고,
-- seq N+1 을 쓸 때 같은 트랜잭션에서 seq N 을 false 로 뒤집는다.
--
-- 이게 없으면 haiku→sonnet→opus 체인이 같은 질문에 대해
-- user_satisfied=true 인 이벤트를 서로 다른 final_label 로 3건 내보내고,
-- ML 의 충돌 탐지(같은 정규화 질문의 final_label 이 2종 이상 → 격리)가
-- 그 질문군을 통째로 걸러버린다.
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS user_satisfied BOOLEAN NOT NULL DEFAULT true;

-- 최초 판정을 내린 라우터 버전.
--
-- request_log 에서 조인하지 않고 비정규화 저장한다.
-- RequestLogService 가 적재 실패를 의도적으로 삼키므로 조인 대상 행이 없을 수 있는데,
-- ML 은 이 값이 비면 missing_router_model_version 으로 행을 거부한다(필수 컬럼).
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS router_model_version VARCHAR(128);

ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS final_model_id VARCHAR(64);

-- 상위 모델이 준 답. NULL 이면 "예약됨, 아직 응답 없음"이다.
-- LLM 호출 전에 행을 먼저 꽂아 유니크 인덱스로 중복 호출을 막고(이중 과금 방지),
-- 완료 시 채운다. 같은 요청이 다시 오면 저장된 답을 그대로 돌려준다(멱등).
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS response TEXT;
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS input_tokens  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS latency_ms    INTEGER;

-- 멱등성 2중 방어.
--   (request_id, escalation_seq): MAX(seq)+1 계산의 경쟁 조건
--   (request_id, final_tier)    : 프론트 더블클릭.
--                                 체인이 단조 증가라 각 티어는 최대 1회만 도달 가능하므로 성립한다.
CREATE UNIQUE INDEX IF NOT EXISTS uq_escalation_log_step
    ON escalation_log (request_id, escalation_seq);
CREATE UNIQUE INDEX IF NOT EXISTS uq_escalation_log_tier
    ON escalation_log (request_id, final_tier);

COMMIT;
