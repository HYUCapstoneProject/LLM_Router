import type {
  ChatRequest,
  ChatResponse,
  EscalateRequest,
  EscalateResponse,
} from '../types'

/** 백엔드 ErrorResponse(04 API Contract)를 담아 UI 까지 전달한다 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    // 백엔드가 항상 ErrorResponse 를 주지만, 프록시 오류 등으로
    // JSON 이 아닐 수 있으므로 파싱 실패도 처리한다
    const err = await res.json().catch(() => null)
    throw new ApiError(
      err?.error ?? `서버 오류가 발생했습니다. (HTTP ${res.status})`,
      err?.code ?? 'UNKNOWN',
      err?.requestId,
    )
  }

  return res.json() as Promise<T>
}

/**
 * onChunk 를 받는 스트리밍 모양을 유지하는 이유:
 * /api/chat 은 언젠가 SSE 로 바뀔 예정이라 호출부를 그때 고치지 않아도 되게 한다.
 * 지금은 응답 전체를 한 번에 흘려보낸다.
 */
export async function sendMessage(
  req: ChatRequest,
  onChunk: (chunk: string) => void,
): Promise<Omit<ChatResponse, 'response'>> {
  const data = await postJson<ChatResponse>('/api/chat', req)
  onChunk(data.response)
  return {
    requestId: data.requestId,
    model: data.model,
    tier: data.tier,
    modelId: data.modelId,
    cacheHit: data.cacheHit,
    inputTokens: data.inputTokens,
    outputTokens: data.outputTokens,
    latency: data.latency,
  }
}

/**
 * 상위 모델로 재요청한다 (기획서 4-1).
 *
 * sendMessage 와 달리 스트리밍 흉내를 내지 않는다. /api/escalate 는 SSE 계획이 없다.
 *
 * 서버가 (requestId, targetModel) 로 멱등 처리하므로 같은 요청을 다시 보내도
 * LLM 이 다시 호출되지 않고 저장된 답이 그대로 돌아온다.
 */
export async function escalate(req: EscalateRequest): Promise<EscalateResponse> {
  return postJson<EscalateResponse>('/api/escalate', req)
}
