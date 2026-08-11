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
