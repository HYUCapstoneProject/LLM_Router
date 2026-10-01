// FastAPI 캐시 임베딩(bge-large-en-v1.5)과 같은 규칙으로 토큰 수만 센다.
//
// 서비스 입력 상한은 이 토크나이저 기준 512 토큰이다(FastAPI embed_max_length).
// 넘으면 서버가 400 MESSAGE_TOO_LONG 으로 거절하므로, 보내기 전에 화면에서 알려준다.
// 최종 판정은 서버가 한다. 여기 숫자는 안내용이다.
//
// 임베딩 모델(가중치)은 필요 없다. 토크나이저는 정규화 규칙과 사전만으로 동작해서
// 같은 사전과 같은 규칙이면 서버(HF tokenizers BertNormalizer + BertPreTokenizer +
// WordPiece)와 같은 수가 나온다. 일치 여부는 Python 이 낸 정답값과 테스트로 비교한다.

export const MAX_INPUT_TOKENS = 512

// 서버 Docker 이미지와 같은 revision(d4aa6901d3a41ba39fb536a557fa166f842b0e09)의 vocab.txt
export const VOCAB_URL = '/tokenizer/bge-large-en-v1.5-vocab.txt'

const SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
const MAX_CHARS_PER_WORD = 100

const OTHER = /\p{C}/u
const WHITESPACE = /\p{White_Space}/u
const MARK_NONSPACING = /\p{Mn}/gu
const PUNCTUATION = /[!-/:-@[-`{-~]|\p{P}/u

function isChineseChar(code: number): boolean {
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x20000 && code <= 0x2a6df) ||
    (code >= 0x2a700 && code <= 0x2b73f) ||
    (code >= 0x2b740 && code <= 0x2b81f) ||
    (code >= 0x2b920 && code <= 0x2ceaf) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0x2f800 && code <= 0x2fa1f)
  )
}

function normalize(text: string): string {
  let cleaned = ''
  for (const ch of text) {
    if (ch === '\t' || ch === '\n' || ch === '\r') {
      cleaned += ' '
      continue
    }
    if (ch === '\0' || ch === '�' || OTHER.test(ch)) continue
    if (WHITESPACE.test(ch)) {
      cleaned += ' '
    } else if (isChineseChar(ch.codePointAt(0)!)) {
      cleaned += ` ${ch} `
    } else {
      cleaned += ch
    }
  }
  // strip_accents 가 null 이면 lowercase 를 따른다: NFD 후 결합 부호 제거, 그다음 소문자.
  // 소문자는 글자 단위로 바꾼다. 문자열 toLowerCase 는 어말 Σ 를 ς 로 바꾸지만 서버는 σ 다.
  const stripped = cleaned.normalize('NFD').replace(MARK_NONSPACING, '')
  return Array.from(stripped, (ch) => ch.toLowerCase()).join('')
}

function preTokenize(text: string): string[] {
  const words: string[] = []
  let current = ''
  for (const ch of text) {
    if (WHITESPACE.test(ch)) {
      if (current) words.push(current)
      current = ''
    } else if (PUNCTUATION.test(ch)) {
      if (current) words.push(current)
      words.push(ch)
      current = ''
    } else {
      current += ch
    }
  }
  if (current) words.push(current)
  return words
}

function wordPieceCount(word: string, vocab: Set<string>): number {
  const chars = Array.from(word)
  if (chars.length > MAX_CHARS_PER_WORD) return 1
  let count = 0
  let start = 0
  while (start < chars.length) {
    let end = chars.length
    let found = false
    while (start < end) {
      const piece = (start > 0 ? '##' : '') + chars.slice(start, end).join('')
      if (vocab.has(piece)) {
        found = true
        break
      }
      end -= 1
    }
    // 한 조각이라도 사전에 없으면 단어 전체가 [UNK] 하나가 된다.
    if (!found) return 1
    count += 1
    start = end
  }
  return count
}

function countSegment(segment: string, vocab: Set<string>): number {
  let count = 0
  for (const word of preTokenize(normalize(segment))) {
    count += wordPieceCount(word, vocab)
  }
  return count
}

/** [CLS]·[SEP] 를 포함한 토큰 수. 서버가 상한과 비교하는 값과 같다. */
export function countTokens(text: string, vocab: Set<string>): number {
  // 특수 토큰 문자열은 정규화 전에 그대로 한 토큰으로 떼어낸다(HF added_tokens 동작).
  let count = 2
  let rest = text
  while (rest) {
    let index = -1
    let token = ''
    for (const special of SPECIAL_TOKENS) {
      const found = rest.indexOf(special)
      if (found >= 0 && (index < 0 || found < index)) {
        index = found
        token = special
      }
    }
    if (index < 0) {
      count += countSegment(rest, vocab)
      break
    }
    count += countSegment(rest.slice(0, index), vocab) + 1
    rest = rest.slice(index + token.length)
  }
  return count
}

/** 사용자에게 보여줄 사용률. 513 토큰이 100% 로 보이지 않도록 올림한다. */
export function usagePercent(tokens: number): number {
  return Math.ceil((tokens / MAX_INPUT_TOKENS) * 100)
}

export function parseVocab(text: string): Set<string> {
  return new Set(text.split('\n').filter((line) => line.length > 0))
}

let vocabPromise: Promise<Set<string>> | null = null

/** 처음 부를 때 한 번만 받는다. 실패하면 다음 호출에서 다시 시도한다. */
export function loadVocab(): Promise<Set<string>> {
  if (!vocabPromise) {
    vocabPromise = fetch(VOCAB_URL)
      .then((response) => {
        if (!response.ok) throw new Error(`vocab ${response.status}`)
        return response.text()
      })
      .then(parseVocab)
      .catch((error) => {
        vocabPromise = null
        throw error
      })
  }
  return vocabPromise
}
