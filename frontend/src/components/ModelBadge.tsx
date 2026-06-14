import type { ModelType } from '../types'

interface Props {
  model: ModelType
  cacheHit?: boolean
  latency?: number
}

const MODEL_CONFIG: Record<ModelType, { label: string; color: string }> = {
  haiku: { label: 'Haiku', color: 'bg-emerald-900 text-emerald-300 border-emerald-700' },
  sonnet: { label: 'Sonnet', color: 'bg-blue-900 text-blue-300 border-blue-700' },
  opus: { label: 'Opus', color: 'bg-purple-900 text-purple-300 border-purple-700' },
}

export function ModelBadge({ model, cacheHit, latency }: Props) {
  const cfg = MODEL_CONFIG[model]

  return (
    <div className="flex items-center gap-1.5 mt-2 flex-wrap">
      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium ${cfg.color}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />
        Claude {cfg.label}
      </span>
      {cacheHit && (
        <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border bg-amber-900 text-amber-300 border-amber-700 font-medium">
          ⚡ Cache Hit
        </span>
      )}
      {latency !== undefined && (
        <span className="text-xs text-gray-500">{latency.toLocaleString()} ms</span>
      )}
    </div>
  )
}
