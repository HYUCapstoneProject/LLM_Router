import type { Conversation } from '../types'

interface Props {
  conversations: Conversation[]
  activeId: string
  onSelect: (id: string) => void
  onNew: () => void
}

function formatTime(ts: number): string {
  const d = new Date(ts)
  const now = new Date()
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000)
  if (diffDays === 0) return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
  if (diffDays === 1) return '어제'
  if (diffDays < 7) return `${diffDays}일 전`
  return d.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
}

export function Sidebar({ conversations, activeId, onSelect, onNew }: Props) {
  return (
    <aside className="w-64 flex-shrink-0 bg-gray-950 border-r border-gray-800 flex flex-col h-full">
      <div className="p-3 border-b border-gray-800">
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:bg-gray-800 transition-colors border border-gray-700"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
          새 대화
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin py-2">
        {conversations.length === 0 ? (
          <p className="text-xs text-gray-600 text-center py-6">대화 없음</p>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`w-full text-left px-3 py-2.5 mx-1 rounded-lg transition-colors group ${
                conv.id === activeId
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
              style={{ width: 'calc(100% - 8px)' }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm truncate flex-1">{conv.title}</span>
                <span className="text-xs text-gray-600 flex-shrink-0">{formatTime(conv.updatedAt)}</span>
              </div>
              {conv.messages.length > 0 && (
                <p className="text-xs text-gray-600 truncate mt-0.5">
                  {conv.messages[conv.messages.length - 1]?.content.slice(0, 40)}
                </p>
              )}
            </button>
          ))
        )}
      </div>

      <div className="p-3 border-t border-gray-800">
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs text-gray-600">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Mock 모드 활성</span>
        </div>
      </div>
    </aside>
  )
}
