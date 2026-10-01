import { useDeferredValue, useEffect, useState, useRef, type KeyboardEvent } from 'react'
import { countTokens, loadVocab, MAX_INPUT_TOKENS, usagePercent } from '../lib/bgeTokenizer'

interface Props {
  onSubmit: (text: string) => void
  disabled?: boolean
}

export function InputBox({ onSubmit, disabled }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [vocab, setVocab] = useState<Set<string> | null>(null)
  // 큰 붙여넣기에서도 타이핑이 밀리지 않게 토큰 계산은 지연 값으로 한다.
  const deferredValue = useDeferredValue(value)

  // 사전(약 230KB)은 처음 입력할 때 한 번만 받는다. 첫 화면 로딩에는 끼지 않는다.
  useEffect(() => {
    if (vocab || !value) return
    let active = true
    loadVocab()
      .then((loaded) => active && setVocab(loaded))
      .catch(() => undefined) // 못 받으면 표시만 생략한다. 서버가 최종 판정한다.
    return () => {
      active = false
    }
  }, [vocab, value])

  const tokens = vocab && deferredValue ? countTokens(deferredValue, vocab) : null
  const tooLong = tokens !== null && tokens > MAX_INPUT_TOKENS

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSubmit = () => {
    if (!value.trim() || disabled || tooLong) return
    onSubmit(value)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  return (
    <div className="border-t border-gray-700 bg-gray-900 px-4 py-4">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-3 bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 focus-within:border-gray-500 transition-colors">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder="메시지를 입력하세요... (Enter 전송, Shift+Enter 줄바꿈)"
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 resize-none outline-none text-sm leading-relaxed disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={disabled || !value.trim() || tooLong}
            className="flex-shrink-0 w-8 h-8 rounded-lg bg-white text-gray-900 flex items-center justify-center hover:bg-gray-200 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          </button>
        </div>
        {tokens !== null && (
          <p
            data-testid="input-usage"
            className={`text-xs text-right mt-1 ${tooLong ? 'text-red-400' : 'text-gray-500'}`}
          >
            {tooLong
              ? `입력 가능 길이의 ${usagePercent(tokens)}% — 질문을 줄여주세요`
              : `${usagePercent(tokens)}%`}
          </p>
        )}
        <p className="text-xs text-gray-600 text-center mt-2">
          LLM Gateway — BERT 분류기 → Haiku / Sonnet / Opus 자동 라우팅
        </p>
      </div>
    </div>
  )
}
