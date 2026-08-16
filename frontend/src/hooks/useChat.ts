import { useState, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Conversation, Message } from '../types'
import { ApiError, escalate as requestEscalation, sendMessage } from '../api/chatApi'
import { nextModel } from '../lib/models'

function makeConversation(firstMessage?: string): Conversation {
  const now = Date.now()
  return {
    id: uuidv4(),
    title: firstMessage ? firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '…' : '') : '새 대화',
    messages: [],
    createdAt: now,
    updatedAt: now,
  }
}

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([makeConversation()])
  const [activeId, setActiveId] = useState<string>(() => conversations[0].id)
  const [isLoading, setIsLoading] = useState(false)

  const activeConversation = conversations.find((c) => c.id === activeId)!

  const updateConversation = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)))
  }, [])

  const startNewConversation = useCallback(() => {
    const conv = makeConversation()
    setConversations((prev) => [conv, ...prev])
    setActiveId(conv.id)
  }, [])

  const selectConversation = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const submit = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return

      const userMsg: Message = {
        id: uuidv4(),
        role: 'user',
        content: text.trim(),
        timestamp: Date.now(),
      }

      const assistantMsgId = uuidv4()
      const assistantMsg: Message = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: Date.now(),
      }

      // 첫 메시지이면 타이틀 업데이트
      updateConversation(activeId, (c) => ({
        ...c,
        title: c.messages.length === 0 ? userMsg.content.slice(0, 30) + (userMsg.content.length > 30 ? '…' : '') : c.title,
        messages: [...c.messages, userMsg, assistantMsg],
        updatedAt: Date.now(),
      }))

      setIsLoading(true)

      try {
        const meta = await sendMessage(
          { message: text.trim(), conversationId: activeId },
          (chunk) => {
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== activeId) return c
                return {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId ? { ...m, content: m.content + chunk } : m,
                  ),
                }
              }),
            )
          },
        )

        updateConversation(activeId, (c) => ({
          ...c,
          messages: c.messages.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  isStreaming: false,
                  requestId: meta.requestId,
                  model: meta.model,
                  tier: meta.tier,
                  cacheHit: meta.cacheHit,
                  inputTokens: meta.inputTokens,
                  outputTokens: meta.outputTokens,
                  latency: meta.latency,
                }
              : m,
          ),
          updatedAt: Date.now(),
        }))
      } catch (err) {
        // 백엔드가 내려준 메시지를 그대로 보여준다.
        // "크레딧 소진"과 "네트워크 오류"를 사용자가 구분할 수 있어야 한다.
        const detail =
          err instanceof ApiError
            ? err.message
            : '서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해주세요.'
        const requestId = err instanceof ApiError ? err.requestId : undefined

        updateConversation(activeId, (c) => ({
          ...c,
          messages: c.messages.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: detail, isStreaming: false, error: true, requestId }
              : m,
          ),
        }))
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, isLoading, updateConversation],
  )

  /**
   * 답변이 만족스럽지 않을 때 한 단계 위 모델로 재요청한다 (기획서 4-1).
   *
   * submit 과 달리 <b>실패해도 기존 답변을 지우지 않는다.</b>
   * submit 은 답이 없는 상태에서 실패하니 content 를 에러로 덮는 게 맞지만,
   * 여기서는 이미 받아둔 답이 있다. 업그레이드가 실패했다고 그걸 잃으면 안 된다.
   */
  const escalate = useCallback(
    async (messageId: string) => {
      if (isLoading) return

      const conversation = conversations.find((c) => c.id === activeId)
      if (!conversation) return

      const index = conversation.messages.findIndex((m) => m.id === messageId)
      if (index < 0) return

      const target = conversation.messages[index]
      if (target.role !== 'assistant' || target.isStreaming || target.error) return
      if (target.escalating || !target.requestId || !target.model) return

      const upgrade = nextModel(target.model)
      if (!upgrade) return

      // 계약이 원본 질문을 요구한다. 이 답변 바로 앞의 사용자 메시지가 그것이다.
      // (서버는 실제로 저장된 질문을 쓰지만 필수 필드라 채워 보낸다)
      const question = conversation.messages
        .slice(0, index)
        .reverse()
        .find((m) => m.role === 'user')
      if (!question) return

      const patch = (updater: (m: Message) => Message) =>
        updateConversation(activeId, (c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === messageId ? updater(m) : m)),
        }))

      patch((m) => ({ ...m, escalating: true, escalationError: undefined }))
      setIsLoading(true)

      try {
        const res = await requestEscalation({
          requestId: target.requestId,
          message: question.content,
          currentModel: target.model,
          targetModel: upgrade,
        })

        patch((m) => ({
          ...m,
          content: res.response,
          model: res.finalModel,
          tier: res.tier,
          // 에스컬레이션은 항상 실제 LLM 호출이다
          cacheHit: false,
          inputTokens: res.inputTokens,
          outputTokens: res.outputTokens,
          latency: res.latency,
          escalationPath: res.escalationPath,
          escalating: false,
        }))
        updateConversation(activeId, (c) => ({ ...c, updatedAt: Date.now() }))
      } catch (err) {
        const detail =
          err instanceof ApiError
            ? err.message
            : '서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.'
        patch((m) => ({ ...m, escalating: false, escalationError: detail }))
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, conversations, isLoading, updateConversation],
  )

  return {
    conversations,
    activeConversation,
    isLoading,
    startNewConversation,
    selectConversation,
    submit,
    escalate,
  }
}
