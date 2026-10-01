import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { InputBox } from './InputBox'

// 사전 하나로 단어 하나 = 1 토큰이 되게 한다. 실제 토큰화 일치는 bgeTokenizer.test 가 본다.
vi.mock('../lib/bgeTokenizer', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../lib/bgeTokenizer')>()
  return { ...actual, loadVocab: () => Promise.resolve(new Set(['word'])) }
})

describe('InputBox', () => {
  it('submits trimmed non-empty input with Enter and clears it', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<InputBox onSubmit={onSubmit} />)
    const textbox = screen.getByRole('textbox')
    await user.type(textbox, '  hello  {enter}')
    expect(onSubmit).toHaveBeenCalledWith('  hello  ')
    expect(textbox).toHaveValue('')
  })

  it('uses Shift+Enter as a newline without submitting', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<InputBox onSubmit={onSubmit} />)
    const textbox = screen.getByRole('textbox')
    await user.type(textbox, 'first{shift>}{enter}{/shift}second')
    expect(onSubmit).not.toHaveBeenCalled()
    expect(textbox).toHaveValue('first\nsecond')
  })

  it('blocks blank or disabled submissions', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    const { rerender } = render(<InputBox onSubmit={onSubmit} />)
    await user.type(screen.getByRole('textbox'), '   {enter}')
    rerender(<InputBox onSubmit={onSubmit} disabled />)
    expect(screen.getByRole('button')).toBeDisabled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('shows usage as a percentage of the input limit', async () => {
    render(<InputBox onSubmit={vi.fn()} />)
    // 254 단어 + [CLS] + [SEP] = 256 토큰 = 50%
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: Array(254).fill('word').join(' ') },
    })
    expect(await screen.findByTestId('input-usage')).toHaveTextContent('50%')
  })

  it('blocks sending once the question exceeds the limit', async () => {
    const onSubmit = vi.fn()
    render(<InputBox onSubmit={onSubmit} />)
    const textbox = screen.getByRole('textbox')
    // 511 단어 + 2 = 513 토큰
    fireEvent.change(textbox, { target: { value: Array(511).fill('word').join(' ') } })
    expect(await screen.findByTestId('input-usage')).toHaveTextContent('101%')
    expect(screen.getByRole('button')).toBeDisabled()
    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
