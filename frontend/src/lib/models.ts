import type { ModelType } from '../types'

/**
 * 모델별 표시 정보.
 *
 * badge  — 답변에 붙는 모델 배지
 * action — 그 모델로 "올리는" 버튼. 목표 모델의 색을 써서 어디로 가는지 눈에 보이게 한다.
 */
export const MODEL_CONFIG: Record<ModelType, { label: string; badge: string; action: string }> = {
  haiku: {
    label: 'Haiku',
    badge: 'bg-emerald-900 text-emerald-300 border-emerald-700',
    action: 'border-emerald-700 text-emerald-300 hover:bg-emerald-950',
  },
  sonnet: {
    label: 'Sonnet',
    badge: 'bg-blue-900 text-blue-300 border-blue-700',
    action: 'border-blue-700 text-blue-300 hover:bg-blue-950',
  },
  opus: {
    label: 'Opus',
    badge: 'bg-purple-900 text-purple-300 border-purple-700',
    action: 'border-purple-700 text-purple-300 hover:bg-purple-950',
  },
}

/**
 * 에스컬레이션 단계. opus 가 마지막이라 더 올라갈 곳이 없다.
 *
 * 04 API Contract 의 `currentModel: haiku|sonnet` / `targetModel: sonnet|opus` 열거형이
 * 이 맵에서 그대로 떨어진다. "이 답변을 올릴 수 있는가"의 단일 진실 원천이다.
 */
const NEXT_MODEL: Record<ModelType, ModelType | null> = {
  haiku: 'sonnet',
  sonnet: 'opus',
  opus: null,
}

export function nextModel(model: ModelType): ModelType | null {
  return NEXT_MODEL[model]
}
