import type { Message } from '../types'
import { ModelBadge } from './ModelBadge'

interface Props {
  message: Message
}

export function MessageBubble({ message }: Props) {
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
            />
          )}
        </div>
      </div>
    </div>
  )
}
