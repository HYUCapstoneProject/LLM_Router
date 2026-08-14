import { useEffect, useRef } from 'react'
import type { Conversation } from '../types'
import { MessageBubble } from './MessageBubble'
import { InputBox } from './InputBox'

interface Props {
  conversation: Conversation
  isLoading: boolean
  onSubmit: (text: string) => void
  onEscalate: (messageId: string) => void
}

export function ChatWindow({ conversation, isLoading, onSubmit, onEscalate }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation.messages])

  const isEmpty = conversation.messages.length === 0

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6">
        {isEmpty ? (
          <WelcomeScreen />
        ) : (
          <div className="max-w-3xl mx-auto">
            {conversation.messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onEscalate={onEscalate}
                busy={isLoading}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>
      <InputBox onSubmit={onSubmit} disabled={isLoading} />
    </div>
  )
}

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 text-center px-4">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-gray-100">LLM 라우팅 게이트웨이</h1>
        <p className="text-gray-400 text-sm max-w-md">
          질문의 복잡도를 BERT 분류기가 자동으로 판단하여 최적의 모델로 라우팅합니다.
        </p>
      </div>
      <div className="grid grid-cols-3 gap-3 w-full max-w-sm">
        {[
          { label: 'Haiku', desc: '간단한 질문', color: 'border-emerald-700 text-emerald-400 bg-emerald-950' },
          { label: 'Sonnet', desc: '중간 복잡도', color: 'border-blue-700 text-blue-400 bg-blue-950' },
          { label: 'Opus', desc: '고난도 추론', color: 'border-purple-700 text-purple-400 bg-purple-950' },
        ].map((m) => (
          <div key={m.label} className={`rounded-xl border p-3 ${m.color}`}>
            <div className="font-medium text-sm">{m.label}</div>
            <div className="text-xs opacity-70 mt-0.5">{m.desc}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-600">시맨틱 캐싱(pgvector)으로 반복 질문은 ⚡ 즉시 응답</p>
    </div>
  )
}
