import { useState, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Conversation, Message } from '../types'
import { sendMessage } from '../api/chatApi'

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
              ? { ...m, isStreaming: false, model: meta.model, cacheHit: meta.cacheHit, latency: meta.latency }
              : m,
          ),
          updatedAt: Date.now(),
        }))
      } catch (err) {
        updateConversation(activeId, (c) => ({
          ...c,
          messages: c.messages.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: '오류가 발생했습니다. 다시 시도해주세요.', isStreaming: false }
              : m,
          ),
        }))
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, isLoading, updateConversation],
  )

  return {
    conversations,
    activeConversation,
    isLoading,
    startNewConversation,
    selectConversation,
    submit,
  }
}
