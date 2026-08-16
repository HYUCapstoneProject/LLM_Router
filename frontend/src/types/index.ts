export type ModelType = 'haiku' | 'sonnet' | 'opus'

export type TierType = 'easy' | 'medium' | 'hard'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** 에스컬레이션 요청 시 백엔드에 넘길 식별자 */
  requestId?: string
  model?: ModelType
  tier?: TierType
  cacheHit?: boolean
  inputTokens?: number
  outputTokens?: number
  latency?: number
  /** 에스컬레이션 경로. 예: ['haiku','sonnet','opus'] */
  escalationPath?: ModelType[]
  /** 상위 모델 재요청이 진행 중 */
  escalating?: boolean
  /**
   * 재요청 실패 사유.
   *
   * error 와 달리 content 를 덮지 않는다. 업그레이드가 실패했다고
   * 멀쩡히 받아둔 답변을 잃으면 안 된다.
   */
  escalationError?: string
  isStreaming?: boolean
  /** true 면 content 가 답변이 아니라 오류 메시지다 */
  error?: boolean
  timestamp: number
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: number
  updatedAt: number
}

export interface ChatRequest {
  message: string
  conversationId: string
}

export interface ChatResponse {
  requestId: string
  response: string
  model: ModelType
  tier: TierType
  /** 실제 호출한 Claude 모델 ID (예: claude-haiku-4-5) */
  modelId: string
  cacheHit: boolean
  /** 캐시 히트면 0 (LLM 호출 없음) */
  inputTokens: number
  outputTokens: number
  latency: number
}

/** GET /api/analytics/summary — 집계값만 온다. 질문 원문은 포함되지 않는다 */
export interface AnalyticsSummary {
  period: AnalyticsPeriod
  /** 비용 계산에 쓴 단가표 버전. 숫자를 재현하려면 이 값이 필요하다 */
  pricingVersion: string
  requests: {
    total: number
    chat: number
    escalation: number
    success: number
    successRate: number
  }
  cache: {
    hits: number
    hitRate: number
    /** 캐시 히트로 아예 건너뛴 LLM 호출 수 */
    llmCallsSkipped: number
  }
  byTier: Array<{
    tier: TierType
    modelId: string
    count: number
    share: number
    inputTokens: number
    outputTokens: number
    cost: number
  }>
  cost: {
    actual: number
    alwaysOpus: number
    savedRatio: number
    /** 이 비교가 무엇인지. 화면에 그대로 노출한다 */
    note: string
  }
  latency: {
    avgMs: number
    cacheHitAvgMs: number
    cacheMissAvgMs: number
  }
  errors: Array<{ status: string; count: number }>
}

export type AnalyticsPeriod = 'today' | '7d' | 'all'

export interface EscalateRequest {
  /** /api/chat 응답의 requestId. 체인의 기준점이다 */
  requestId: string
  message: string
  currentModel: ModelType
  targetModel: ModelType
}

export interface EscalateResponse {
  /**
   * 원본 chat 의 requestId 를 그대로 돌려준다.
   * 같은 말풍선에서 sonnet → opus 로 한 번 더 올릴 수 있어야 하기 때문이다.
   */
  requestId: string
  response: string
  /** 라우터의 최초 판정. 직전 단계가 아니라 체인의 출발점이다 */
  initialModel: ModelType
  finalModel: ModelType
  escalationPath: ModelType[]
  latency: number
  tier: TierType
  modelId: string
  inputTokens: number
  outputTokens: number
}
