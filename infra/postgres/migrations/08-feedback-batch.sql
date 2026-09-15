-- 08: 피드백 배치 내보내기와 재학습 상태 추적
--
-- 02-escalation.sql 주석이 "03-feedback-batch.sql 로 분리했다"고 예고했지만
-- 그 파일은 만들어진 적이 없다. 03 자리는 request-log-status 가 쓰고 있어
-- 번호를 08 로 잡는다.
--
--   docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter < 08-feedback-batch.sql
--
-- 전부 IF NOT EXISTS 라 여러 번 돌려도 안전하다.
-- init/01-schema.sql 에도 같은 내용이 반영되어야 한다 (새 볼륨용). 둘은 항상 같이 고친다.

BEGIN;

-- ─────────────────────────────────────────────────────────────
-- 1. escalation_log — 어느 배치로 나갔는가
-- ─────────────────────────────────────────────────────────────
--
-- 이 컬럼이 없으면 같은 이벤트가 매 배치에 다시 실린다.
-- FastAPI 는 batchId 가 다르면 다른 배치로 받으므로 중복을 막아주지 않는다.
-- 즉 중복 제출 방지는 전적으로 게이트웨이 책임이다.
--
-- NULL = 아직 안 나감. 값이 있으면 그 batchId 로 제출 완료.
ALTER TABLE escalation_log ADD COLUMN IF NOT EXISTS exported_batch_id VARCHAR(200);

-- 선별 쿼리가 "아직 안 나간 것"을 훑는다. 부분 인덱스로 충분하다 —
-- 이미 내보낸 행은 조회 대상이 아니고, 시간이 갈수록 그쪽이 대다수가 된다.
CREATE INDEX IF NOT EXISTS idx_escalation_log_unexported
    ON escalation_log (created_at)
    WHERE exported_batch_id IS NULL;

-- ─────────────────────────────────────────────────────────────
-- 2. feedback_batch — 제출 기록과 재학습 진행 상태
-- ─────────────────────────────────────────────────────────────
--
-- 04 API Contract 의 "Spring 저장 권장 필드" + "DB column 권장안" 을 합친 것이다.
--
-- 핵심 원칙: **202 Accepted 는 학습 완료가 아니다.** GCS artifact 생성과
-- Vertex PipelineJob 제출이 끝났다는 뜻일 뿐이라, 제출 시점 정보(pipeline_run_id 등)와
-- 폴링으로 갱신되는 진행 상태(pipeline_status 등)를 한 행에 두되 의미를 분리해 둔다.
CREATE TABLE IF NOT EXISTS feedback_batch (
    -- 계약상 immutable 멱등성 키. 1~200자, 영문·숫자·점·밑줄·하이픈.
    -- PK 로 두면 같은 batchId 재제출이 DB 단에서 한 번 더 막힌다.
    batch_id                VARCHAR(200) PRIMARY KEY,

    -- 제출한 이벤트 수. events 배열 자체는 escalation_log.exported_batch_id 로
    -- 되짚을 수 있으므로 payload 를 통째로 복제하지 않는다.
    event_count             INTEGER      NOT NULL,

    -- 재시도 시 같은 payload 를 보내야 한다(계약: 재시도 때문에 새 batch ID 를
    -- 만들거나 payload 를 나누지 않는다). 이벤트 집합의 지문을 남겨,
    -- 재구성한 payload 가 제출본과 같은지 확인할 수 있게 한다.
    events_fingerprint      CHAR(64),

    -- ── 202 응답에서 받은 값 (제출 시점에 확정) ──
    pipeline_run_id         TEXT,          -- Google resource name. URL 이 아니다.
    logical_model_version   VARCHAR(64),
    candidate_model_version VARCHAR(200),
    training_mode           VARCHAR(64),
    version_policy_mode     VARCHAR(64),
    status_url              TEXT,          -- 상대 경로. base URL 에 resolve 해서 GET.

    -- ── 폴링으로 갱신되는 값 ──
    --
    -- queued | running | cancelling | paused | succeeded | rejected
    -- | failed | cancelled | unknown
    -- 'submitted' 는 202 직후 첫 폴링 전의 로컬 상태다(계약에 없는 값).
    pipeline_status         VARCHAR(32)  NOT NULL DEFAULT 'submitted',

    -- initializing | feedbackPreparation | feedbackMerge | training
    -- | evaluation | packaging | gate | publishing | complete
    pipeline_stage          VARCHAR(32),

    result_outcome          VARCHAR(32),
    failure_code            VARCHAR(64),
    retryable               BOOLEAN,

    -- pending | active | loadFailed | superseded | notApplicable
    --
    -- **학습 성공과 serving 활성화 성공은 다른 사건이다.** status=succeeded 여도
    -- activation 이 loadFailed 면 이전 모델이 계속 응답하고 있다.
    activation_status       VARCHAR(32),

    submitted_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ,   -- terminal 상태 도달 시각
    last_polled_at          TIMESTAMPTZ,

    -- 수동 버튼으로 강제 제출했는지. 자동 임계값(1,000건) 도달과 구분한다.
    -- 실험 서비스라 대부분 수동이 될 텐데, 그걸 자동 트리거 실적으로 집계하면 안 된다.
    triggered_by            VARCHAR(16)  NOT NULL DEFAULT 'auto'
);

-- 폴링 대상 조회: terminal 이 아닌 행만 훑는다.
CREATE INDEX IF NOT EXISTS idx_feedback_batch_pending
    ON feedback_batch (last_polled_at NULLS FIRST)
    WHERE pipeline_status NOT IN ('succeeded', 'rejected', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS idx_feedback_batch_submitted_at
    ON feedback_batch (submitted_at DESC);

COMMIT;
