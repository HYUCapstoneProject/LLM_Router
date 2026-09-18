import type {
  AnalyticsPeriod,
  AnalyticsSummary,
  ChatRequest,
  ChatResponse,
  EscalateRequest,
  EscalateResponse,
  RetrainingOverview,
  RetrainingSubmission,
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

async function parse<T>(res: Response): Promise<T> {
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

async function postJson<T>(url: string, body: unknown): Promise<T> {
  return parse<T>(
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
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

/**
 * 운영 로그 집계를 가져온다 (시연용).
 *
 * 응답에 질문 원문은 없다. 서버가 집계값만 내려주므로 화면에 그대로 띄워도 된다.
 */
export async function fetchAnalytics(period: AnalyticsPeriod): Promise<AnalyticsSummary> {
  return parse<AnalyticsSummary>(await fetch(`/api/analytics/summary?period=${period}`))
}

/** 재학습 대기 건수와 최근 배치 이력을 가져온다 */
export async function fetchRetrainingStatus(): Promise<RetrainingOverview> {
  return parse<RetrainingOverview>(await fetch('/api/retraining/status'))
}

/**
 * 대기 중인 에스컬레이션을 지금 재학습으로 내보낸다 (강제 실행).
 *
 * ⚠️ 성공하면 팀원 쪽 모델 버전이 한 칸 올라간다. 계약상 승격까지 성공한 고유 batch
 * 3회가 한 major cycle 이라, 시험 삼아 반복하면 버전이 소모되고 되돌릴 수 없다.
 * 호출부에서 반드시 확인을 받는다.
 *
 * @returns 보낼 이벤트가 없으면 null (서버가 204 를 준다)
 */
export async function submitRetraining(): Promise<RetrainingSubmission | null> {
  const res = await fetch('/api/retraining/submit', { method: 'POST' })
  // 204 는 본문이 없다. parse() 를 그대로 태우면 JSON 파싱에서 터진다.
  if (res.status === 204) return null
  return parse<RetrainingSubmission>(res)
}
