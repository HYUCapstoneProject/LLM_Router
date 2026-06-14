export type ModelType = 'haiku' | 'sonnet' | 'opus'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  model?: ModelType
  cacheHit?: boolean
  latency?: number
  isStreaming?: boolean
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
  response: string
  model: ModelType
  cacheHit: boolean
  latency: number
}
