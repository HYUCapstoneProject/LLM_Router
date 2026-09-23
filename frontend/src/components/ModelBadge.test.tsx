import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ModelBadge } from './ModelBadge'

describe('ModelBadge', () => {
  it.each([['haiku', 'Haiku'], ['sonnet', 'Sonnet'], ['opus', 'Opus']] as const)(
    'renders %s model label', (model, label) => {
      render(<ModelBadge model={model} />)
      expect(screen.getByText(`Claude ${label}`)).toBeInTheDocument()
    },
  )

  it('shows token and latency metadata for an LLM call', () => {
    render(<ModelBadge model="sonnet" inputTokens={1200} outputTokens={34} latency={1500} />)
    expect(screen.getByText(/1,200 in/)).toBeInTheDocument()
    expect(screen.getByText(/1,500 ms/)).toBeInTheDocument()
  })

  it('shows cache status and hides token counts on a cache hit', () => {
    render(<ModelBadge model="haiku" cacheHit inputTokens={0} outputTokens={0} />)
    expect(screen.getByText(/Cache Hit/)).toBeInTheDocument()
    expect(screen.queryByText(/0 in/)).not.toBeInTheDocument()
  })
})
