import type { ChatRequest, ChatResponse, ModelType } from '../types'

const USE_MOCK = false // 백엔드 연동 시 false로 변경

// ── Mock 응답 데이터 ──────────────────────────────────────────────
const MOCK_RESPONSES: Array<{ text: string; model: ModelType; cacheHit: boolean }> = [
  {
    text: '안녕하세요! 저는 LLM 라우팅 게이트웨이를 통해 응답하는 AI입니다. 간단한 질문은 Claude Haiku가 빠르게 처리합니다.',
    model: 'haiku',
    cacheHit: false,
  },
  {
    text: '이 질문은 중간 수준의 복잡도로 판단되어 Claude Sonnet이 처리하고 있습니다. 시맨틱 캐싱을 통해 유사한 질문은 더 빠르게 응답할 수 있습니다.',
    model: 'sonnet',
    cacheHit: false,
  },
  {
    text: '복잡한 추론이 필요한 질문으로 분류되어 Claude Opus가 처리합니다. BERT 분류기가 질문의 복잡도를 Heavy로 판단했습니다.',
    model: 'opus',
    cacheHit: false,
  },
  {
    text: '이전에 유사한 질문이 있었습니다. 시맨틱 캐시에서 응답을 가져왔습니다!',
    model: 'haiku',
    cacheHit: true,
  },
]

let mockIndex = 0

async function mockStreamResponse(
  text: string,
  onChunk: (chunk: string) => void,
): Promise<void> {
  const words = text.split(' ')
  for (const word of words) {
    await new Promise((r) => setTimeout(r, 40 + Math.random() * 60))
    onChunk(word + ' ')
  }
}

// ── 실제 API 호출 ────────────────────────────────────────────────
async function realSendMessage(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<ChatResponse>
}

// ── 공개 인터페이스 ──────────────────────────────────────────────
export async function sendMessage(
  req: ChatRequest,
  onChunk: (chunk: string) => void,
): Promise<Omit<ChatResponse, 'response'>> {
  if (USE_MOCK) {
    const mock = MOCK_RESPONSES[mockIndex % MOCK_RESPONSES.length]
    mockIndex++
    const start = Date.now()
    await mockStreamResponse(mock.text, onChunk)
    return {
      model: mock.model,
      cacheHit: mock.cacheHit,
      latency: Date.now() - start,
    }
  }

  // 실제 백엔드 연동 시: 스트리밍 미지원이면 전체 응답을 한 번에 emit
  const data = await realSendMessage(req)
  onChunk(data.response)
  return { model: data.model, cacheHit: data.cacheHit, latency: data.latency }
}
