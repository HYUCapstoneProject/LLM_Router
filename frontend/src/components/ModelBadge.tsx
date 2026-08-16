import type { ModelType } from '../types'
import { MODEL_CONFIG } from '../lib/models'

interface Props {
  model: ModelType
  cacheHit?: boolean
  inputTokens?: number
  outputTokens?: number
  latency?: number
  /** 에스컬레이션 경로. 2단계 이상이면 어디서 어디로 올라왔는지 보여준다 */
  escalationPath?: ModelType[]
}

export function ModelBadge({
  model,
  cacheHit,
  inputTokens,
  outputTokens,
  latency,
  escalationPath,
}: Props) {
  const cfg = MODEL_CONFIG[model]
  const hasTokens = inputTokens !== undefined && outputTokens !== undefined
  const escalated = escalationPath !== undefined && escalationPath.length > 1

  return (
    <div className="flex items-center gap-1.5 mt-2 flex-wrap">
      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium ${cfg.badge}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />
        Claude {cfg.label}
      </span>
      {/* 에스컬레이션이 눈에 보이는 유일한 흔적이다. 데모의 핵심 장면 */}
      {escalated && (
        <span className="text-xs text-gray-500">
          ↑ {escalationPath.map((m) => MODEL_CONFIG[m].label).join(' → ')}
        </span>
      )}
      {cacheHit && (
        <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border bg-amber-900 text-amber-300 border-amber-700 font-medium">
          ⚡ Cache Hit
        </span>
      )}
      {/* 캐시 히트는 LLM 호출이 없으므로 "0 토큰"이 아니라 호출 없음으로 표시 */}
      {cacheHit ? (
        <span className="text-xs text-amber-500">LLM 호출 없음</span>
      ) : (
        hasTokens && (
          <span className="text-xs text-gray-500">
            {inputTokens.toLocaleString()} in · {outputTokens.toLocaleString()} out
          </span>
        )
      )}
      {latency !== undefined && (
        <span className="text-xs text-gray-500">{latency.toLocaleString()} ms</span>
      )}
    </div>
  )
}
