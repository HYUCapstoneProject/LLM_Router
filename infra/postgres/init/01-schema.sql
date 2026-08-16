-- LLM Router 스키마 (로컬 개발용 초기화)
-- 컨테이너의 데이터 볼륨이 비어 있을 때 1회만 실행된다.

CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────────────────────
-- 1. 시맨틱 캐시
-- ─────────────────────────────────────────────────────────────
CREATE TABLE cache_entry (
    id                  BIGSERIAL PRIMARY KEY,
    question            TEXT         NOT NULL,
    embedding           vector(1024) NOT NULL,
    response            TEXT         NOT NULL,

    -- 이 답을 처음 만들 때 쓴 티어/모델. 캐시 히트 시 비용은 0이지만
    -- 어느 모델의 답이 재사용되는지는 캐시 품질 분석에 필요하다.
    tier                VARCHAR(10)  NOT NULL,   -- easy | medium | hard
    model_id            VARCHAR(64)  NOT NULL,   -- claude-sonnet-4-6

    -- 임베딩 모델을 교체하면 기존 벡터와 비교가 불가능하므로 버전을 남긴다.
    embed_model_version VARCHAR(64)  NOT NULL,

    -- 이 항목을 재사용하기 위해 요구하는 최소 유사도.
    -- 저장 시점에 이 항목의 tier 로부터 결정된다(application.yml).
    --
    -- 왜 항목별로 두는가:
    --   들어온 질문의 tier 는 캐시 조회 시점에 아직 없다.
    --   classify 는 cache miss 일 때만 호출하기 때문이다.
    --   대신 "이 캐시 항목이 어떤 난이도의 답인가"를 기준으로 삼는다 —
    --   hard 질문의 답은 재사용에 더 높은 유사도를 요구한다.
    --
    -- 초기값은 전 티어 동일(= 고정 임계값)이며,
    -- EXP-003 에서 QQP 로 티어별 곡선을 측정한 뒤 값을 벌린다.
    threshold           REAL         NOT NULL DEFAULT 0.80,

    hit_count           INTEGER      NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 코사인 거리 기준 ANN 인덱스.
-- FastAPI 가 L2 정규화된 벡터를 반환한다는 전제이며(04 API Contract),
-- 그 경우 `유사도 = 1 - (a <=> b)` 가 그대로 성립한다.
CREATE INDEX idx_cache_entry_embedding
    ON cache_entry USING hnsw (embedding vector_cosine_ops);

-- 조회 쿼리 형태 (Gateway 가 이 모양으로 질의한다):
--
--   SELECT * FROM (
--       SELECT *, 1 - (embedding <=> $1) AS similarity
--       FROM cache_entry
--       ORDER BY embedding <=> $1     -- 여기서 HNSW 인덱스 사용
--       LIMIT 5
--   ) t
--   WHERE t.similarity >= t.threshold  -- 항목별 임계값은 바깥에서 적용
--   ORDER BY t.similarity DESC
--   LIMIT 1;
--
-- threshold 를 안쪽 WHERE 에 직접 걸면 인덱스를 타지 못하고 전수 스캔이 된다.

-- ─────────────────────────────────────────────────────────────
-- 2. 요청 로그  ← 비용 그래프의 원천 데이터
-- ─────────────────────────────────────────────────────────────
-- 최종 보고서 필수 그림 "Router vs always-Opus 비용/정확도"를 그리려면
-- 요청 단위 토큰이 집계 가능한 형태로 쌓여 있어야 한다.
-- 콘솔 로그를 나중에 파싱하는 방식은 재현성이 없어 M7 freeze 때 숫자를 다시 못 만든다.
CREATE TABLE request_log (
    id                   BIGSERIAL PRIMARY KEY,
    request_id           UUID         NOT NULL UNIQUE,
    question             TEXT         NOT NULL,
    conversation_id      VARCHAR(64),

    tier                 VARCHAR(10),
    model_id             VARCHAR(64),

    -- 재학습 전후 비교에 필수. 어느 모델 버전이 내린 판정인지 알아야 한다.
    router_model_version VARCHAR(128),
    router_confidence    REAL,
    embed_model_version  VARCHAR(64),

    cache_hit            BOOLEAN      NOT NULL,
    similarity           REAL,                     -- 캐시 조회 시 최고 유사도 (임계값 튜닝용)

    -- 이 요청이 히트했거나 새로 만든 캐시 행.
    -- /api/escalate 가 "불만족한 답이 어느 캐시 행에서 나왔는지" 를 알아야
    -- 그 행을 정확히 갱신하고(기획서 4-3), escalation_log 에 근거를 남길 수 있다.
    cache_entry_id       BIGINT       REFERENCES cache_entry(id) ON DELETE SET NULL,

    -- 캐시 히트면 0. LLM 호출이 없었으므로 비용도 0이다.
    input_tokens         INTEGER      NOT NULL DEFAULT 0,
    output_tokens        INTEGER      NOT NULL DEFAULT 0,

    latency_ms           INTEGER      NOT NULL,

    -- NULL 이면 정상 라우팅. 값이 있으면 FastAPI 장애로 폴백된 요청이므로
    -- 라우터 정확도 집계에서 제외한다 (04 API Contract / Failure Handling).
    fallback_reason      VARCHAR(32),

    -- 실패한 요청도 남긴다. 성공만 기록하면 "전체 중 몇 %가 실패했나",
    -- "지연시간 분포" 같은 지표를 낼 수 없다.
    -- success | llm_error | db_error | internal_error
    status               VARCHAR(20)  NOT NULL DEFAULT 'success',
    error_type           VARCHAR(64),   -- 예외 클래스명. 실패 원인 분류용

    -- 에스컬레이션은 실제로 과금되는 LLM 호출이다.
    -- RequestIdFilter 가 /api/escalate 에 "새" requestId 를 발급하므로,
    -- 이 구분이 없으면 비용 집계가 에스컬레이션을 통째로 빠뜨린다 —
    -- 그것도 라우터에 유리한 방향으로. 그래서 정상 행으로 남기고 origin 으로 나눈다.
    --   비용 집계     : 전체 행
    --   라우터 정확도 : WHERE origin = 'chat'
    parent_request_id    UUID,                     -- 에스컬레이션이면 원본 chat 의 request_id
    origin               VARCHAR(16)  NOT NULL DEFAULT 'chat',   -- chat | escalate

    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_request_log_created_at ON request_log (created_at);
CREATE INDEX idx_request_log_tier       ON request_log (tier);
CREATE INDEX idx_request_log_cache_hit  ON request_log (cache_hit);
CREATE INDEX idx_request_log_status     ON request_log (status);
CREATE INDEX idx_request_log_parent     ON request_log (parent_request_id);
CREATE INDEX idx_request_log_origin     ON request_log (origin);

-- ─────────────────────────────────────────────────────────────
-- 3. 에스컬레이션 로그  ← 재학습의 정정 라벨
-- ─────────────────────────────────────────────────────────────
-- 사용자가 만족한 티어가 그 질문의 실제 필요 수준이다 (기획서 4-1).
-- 이 기록이 N건 이상 쌓이면 Kubeflow 재학습을 트리거한다.
--
-- 모델링: 에스컬레이션 "단계당 1행". request_id 는 비유니크다.
--   seq 1: initial=easy, final=medium, path={haiku,sonnet}
--   seq 2: initial=easy, final=hard,   path={haiku,sonnet,opus}
-- initial_tier 는 항상 라우터의 최초 판정이고, escalation_path 는 누적 저장한다.
CREATE TABLE escalation_log (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID         NOT NULL,
    question        TEXT         NOT NULL,
    conversation_id VARCHAR(64),
    embedding       vector(1024),

    initial_tier    VARCHAR(10)  NOT NULL,   -- 라우터의 최초 판정
    escalation_path TEXT[]       NOT NULL,   -- {haiku,sonnet}
    final_tier      VARCHAR(10)  NOT NULL,   -- 사용자가 만족한 티어 = 정정 라벨
    final_model_id  VARCHAR(64),
    escalation_seq  INTEGER      NOT NULL DEFAULT 1,

    -- 상위 모델이 준 답.
    --
    -- NULL 이면 "예약됨, 아직 응답 없음"이다. LLM 을 부르기 전에 행을 먼저 꽂아
    -- 유니크 인덱스로 중복 호출을 막고(이중 과금 방지), 완료 시 이 컬럼을 채운다.
    -- 같은 요청이 다시 오면 여기 저장된 답을 그대로 돌려준다(멱등).
    --
    -- 보고서에서 "haiku 는 이렇게 답해 불만족, opus 는 이렇게 답해 만족"을
    -- 대조해 보여줄 때도 쓴다.
    response        TEXT,

    -- 체인의 종결 표시. seq N+1 을 쓸 때 같은 트랜잭션에서 seq N 을 false 로 뒤집는다.
    -- 이게 없으면 한 질문에 대해 final_label 이 다른 "만족" 이벤트가 여러 건 나가고,
    -- ML 의 충돌 탐지가 그 질문군을 통째로 격리한다.
    user_satisfied  BOOLEAN      NOT NULL DEFAULT true,

    -- 최초 판정을 내린 라우터 버전. request_log 조인 대신 비정규화한다 —
    -- RequestLogService 는 적재 실패를 삼키므로 조인 대상이 없을 수 있고,
    -- ML 은 이 값이 비면 행을 거부한다.
    router_model_version VARCHAR(128),

    input_tokens    INTEGER      NOT NULL DEFAULT 0,
    output_tokens   INTEGER      NOT NULL DEFAULT 0,
    latency_ms      INTEGER,

    -- 불만족한 답이 캐시에서 나온 것인가.
    -- true 면 라우터의 과소 라우팅이 아니라 캐시의 false hit 증거다.
    -- 두 신호를 섞으면 재학습 라벨이 오염되므로 반드시 구분한다.
    -- (적응형 임계값 Level 3 의 검증 신호이기도 하다)
    was_cache_hit   BOOLEAN      NOT NULL DEFAULT false,
    cache_entry_id  BIGINT       REFERENCES cache_entry(id) ON DELETE SET NULL,
    similarity      REAL,                    -- 그때 매칭된 유사도

    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_escalation_log_created_at ON escalation_log (created_at);
CREATE INDEX idx_escalation_log_request_id ON escalation_log (request_id);

-- 멱등성 2중 방어.
--   (request_id, escalation_seq): MAX(seq)+1 계산의 경쟁 조건
--   (request_id, final_tier)    : 프론트 더블클릭.
--                                 체인이 단조 증가라 각 티어는 최대 1회만 도달 가능하다.
CREATE UNIQUE INDEX uq_escalation_log_step ON escalation_log (request_id, escalation_seq);
CREATE UNIQUE INDEX uq_escalation_log_tier ON escalation_log (request_id, final_tier);
