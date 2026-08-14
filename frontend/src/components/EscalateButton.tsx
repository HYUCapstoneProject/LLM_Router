import type { ModelType } from '../types'
import { MODEL_CONFIG } from '../lib/models'

interface Props {
  /** 올라갈 모델. 버튼 색도 이 모델의 것을 쓴다 */
  target: ModelType
  escalating?: boolean
  disabled?: boolean
  onClick: () => void
}

/**
 * "이 답변으로 부족하다" 버튼 (기획서 4-1).
 *
 * 요청 중에는 비활성화한다. 서버가 멱등 처리를 하므로 중복 클릭이 요금으로
 * 이어지지는 않지만, 굳이 왕복을 만들 이유도 없다.
 */
export function EscalateButton({ target, escalating, disabled, onClick }: Props) {
  const cfg = MODEL_CONFIG[target]

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={escalating || disabled}
      className={`mt-2 inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border
        transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${cfg.action}`}
    >
      {escalating ? (
        <>
          <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
          {cfg.label}에게 다시 묻는 중…
        </>
      ) : (
        <>⤴ {cfg.label}로 다시 답변</>
      )}
    </button>
  )
}
