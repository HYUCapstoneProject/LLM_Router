import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, sendMessage } from './chatApi'

describe('sendMessage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the request, emits the response once, and returns metadata', async () => {
    const response = {
      requestId: 'req-1', response: 'answer', model: 'sonnet', tier: 'medium', modelId: 'm-1',
      cacheHit: false, inputTokens: 3, outputTokens: 4, latency: 50,
    }
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(response) })
    vi.stubGlobal('fetch', fetchMock)
    const onChunk = vi.fn()

    const meta = await sendMessage({ message: ' q ', conversationId: 'c-1' }, onChunk)

    expect(fetchMock).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ message: ' q ', conversationId: 'c-1' }),
    }))
    expect(onChunk).toHaveBeenCalledOnce()
    expect(onChunk).toHaveBeenCalledWith('answer')
    expect(meta).not.toHaveProperty('response')
    expect(meta.requestId).toBe('req-1')
  })

  it('preserves structured backend error details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 429,
      json: () => Promise.resolve({ error: 'slow down', code: 'RATE_LIMIT', requestId: 'req-e' }),
    }))
    await expect(sendMessage({ message: 'q', conversationId: 'c' }, vi.fn()))
      .rejects.toEqual(expect.objectContaining({ name: 'ApiError', message: 'slow down', code: 'RATE_LIMIT', requestId: 'req-e' }))
  })

  it('falls back when an error body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 502, json: () => Promise.reject(new Error('html')),
    }))
    const promise = sendMessage({ message: 'q', conversationId: 'c' }, vi.fn())
    await expect(promise).rejects.toBeInstanceOf(ApiError)
    await expect(promise).rejects.toEqual(expect.objectContaining({ code: 'UNKNOWN' }))
  })
})
