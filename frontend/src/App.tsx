import { useChat } from './hooks/useChat'
import { Sidebar } from './components/Sidebar'
import { ChatWindow } from './components/ChatWindow'

export default function App() {
  const { conversations, activeConversation, isLoading, startNewConversation, selectConversation, submit } = useChat()

  return (
    <div className="flex h-screen bg-gray-900">
      <Sidebar
        conversations={conversations}
        activeId={activeConversation.id}
        onSelect={selectConversation}
        onNew={startNewConversation}
      />
      <main className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 flex-shrink-0">
          <h2 className="text-sm font-medium text-gray-300 truncate">{activeConversation.title}</h2>
          <span className="text-xs text-gray-600 flex-shrink-0">
            {activeConversation.messages.filter((m) => m.role === 'user').length} 개 메시지
          </span>
        </header>
        <ChatWindow
          conversation={activeConversation}
          isLoading={isLoading}
          onSubmit={submit}
        />
      </main>
    </div>
  )
}
