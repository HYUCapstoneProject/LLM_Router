import type { Message } from '../types'
import { nextModel } from '../lib/models'
import { EscalateButton } from './EscalateButton'
import { ModelBadge } from './ModelBadge'

interface Props {
  message: Message
  /** 없으면 재요청 버튼을 그리지 않는다 */
  onEscalate?: (messageId: string) => void
  /** 다른 요청이 진행 중 */
  busy?: boolean
}

export function MessageBubble({ message, onEscalate, busy }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[70%] bg-gray-700 text-gray-100 rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  // 올라갈 곳이 남아 있고(opus 면 없음), 정상 응답이며, 체인 기준점이 있을 때만 올릴 수 있다.
  const upgrade =
    !message.isStreaming && !message.error && message.requestId && message.model
      ? nextModel(message.model)
      : null

  return (
    <div className="flex justify-start mb-4">
      <div className="flex gap-3 max-w-[80%]">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gray-600 flex items-center justify-center text-xs font-bold text-gray-300 mt-0.5">
          AI
        </div>
        <div>
          <div
            className={`rounded-2xl rounded-tl-sm px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
              message.error
                ? 'bg-red-950 text-red-200 border border-red-800'
                : 'bg-gray-800 text-gray-100'
            }`}
          >
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 bg-gray-400 ml-0.5 animate-pulse rounded-sm" />
            )}
          </div>
          {message.error && message.requestId && (
            <div className="mt-1 text-xs text-gray-600 font-mono">
              requestId: {message.requestId}
            </div>
          )}
          {!message.isStreaming && message.model && (
            <ModelBadge
              model={message.model}
              cacheHit={message.cacheHit}
              inputTokens={message.inputTokens}
              outputTokens={message.outputTokens}
              latency={message.latency}
              escalationPath={message.escalationPath}
            />
          )}
          {upgrade && onEscalate && (
            <EscalateButton
              target={upgrade}
              escalating={message.escalating}
              disabled={busy}
              onClick={() => onEscalate(message.id)}
            />
          )}
          {/* 재요청 실패는 답변을 덮지 않고 따로 알린다 */}
          {message.escalationError && (
            <div className="mt-1 text-xs text-red-400">{message.escalationError}</div>
          )}
        </div>
      </div>
    </div>
  )
}
