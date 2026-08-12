import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { InputBox } from './InputBox'

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
})
