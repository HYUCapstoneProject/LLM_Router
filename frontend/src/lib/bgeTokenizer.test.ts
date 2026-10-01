/// <reference types="vite/client" />
import { describe, expect, it } from 'vitest'
import { countTokens, parseVocab, usagePercent } from './bgeTokenizer'
import golden from './__fixtures__/bgeTokenCounts.json'
import vocabText from '../../public/tokenizer/bge-large-en-v1.5-vocab.txt?raw'

// 정답값은 서버와 같은 revision 의 HF 토크나이저(Python)로 만들었다.
// 이 테스트가 깨지면 화면 숫자와 서버 거절 기준이 어긋난다.
const vocab = parseVocab(vocabText)

describe('bgeTokenizer', () => {
  it('loads the full BERT vocabulary', () => {
    expect(vocab.size).toBe(30522)
  })

  it.each(golden)('matches the server token count: %#', ({ text, tokens }) => {
    expect(countTokens(text, vocab)).toBe(tokens)
  })

  it('rounds usage up so one token over the limit never reads as 100%', () => {
    expect(usagePercent(512)).toBe(100)
    expect(usagePercent(513)).toBe(101)
    expect(usagePercent(3)).toBe(1)
  })
})
