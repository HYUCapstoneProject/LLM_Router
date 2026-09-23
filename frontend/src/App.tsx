import { useState } from 'react'
import { useChat } from './hooks/useChat'
import { Sidebar } from './components/Sidebar'
import { ChatWindow } from './components/ChatWindow'
import { AnalyticsPanel } from './components/AnalyticsPanel'

export default function App() {
  const {
    conversations,
    activeConversation,
    isLoading,
    startNewConversation,
    selectConversation,
    submit,
    escalate,
  } = useChat()

  // 라우터 라이브러리를 쓰지 않으므로 화면 전환은 상태로 한다.
  // 대화 상태는 useChat 이 들고 있어서 탭을 오가도 유지된다.
  const [view, setView] = useState<'chat' | 'analytics'>('chat')

  return (
    <div className="flex h-screen bg-gray-900">
      <Sidebar
        conversations={conversations}
        activeId={activeConversation.id}
        onSelect={(id) => {
          selectConversation(id)
          setView('chat')
        }}
        onNew={() => {
          startNewConversation()
          setView('chat')
        }}
      />
      <main className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 flex-shrink-0">
          <h2 className="text-sm font-medium text-gray-300 truncate">
            {view === 'chat' ? activeConversation.title : '운영 지표'}
          </h2>
          <div className="flex items-center gap-3 flex-shrink-0">
            {view === 'chat' && (
              <span className="text-xs text-gray-600">
                {activeConversation.messages.filter((m) => m.role === 'user').length} 개 메시지
              </span>
            )}
            <button
              onClick={() => setView(view === 'chat' ? 'analytics' : 'chat')}
              className="text-xs px-3 py-1 rounded-md border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              {view === 'chat' ? '지표 보기' : '채팅으로'}
            </button>
          </div>
        </header>
        {view === 'chat' ? (
          <ChatWindow
            conversation={activeConversation}
            isLoading={isLoading}
            onSubmit={submit}
            onEscalate={escalate}
          />
        ) : (
          <AnalyticsPanel />
        )}
      </main>
    </div>
  )
}
