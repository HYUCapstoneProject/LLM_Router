import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { AnalyticsPeriod, AnalyticsSummary, ModelType, TierType } from '../types'
import { fetchAnalytics, ApiError } from '../api/chatApi'
import { RetrainingPanel } from './RetrainingPanel'
import { MODEL_CONFIG } from '../lib/models'

/** 티어 → 그 티어가 실제로 호출하는 모델. 배지 색을 채팅 화면과 맞춘다 */
const TIER_MODEL: Record<TierType, ModelType> = {
  easy: 'haiku',
  medium: 'sonnet',
  hard: 'opus',
}

const PERIODS: Array<{ value: AnalyticsPeriod; label: string }> = [
  { value: 'today', label: '오늘' },
  { value: '7d', label: '최근 7일' },
  { value: 'all', label: '전체' },
]

const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const usd = (v: number) => `$${v.toFixed(4)}`

export function AnalyticsPanel() {
  const [period, setPeriod] = useState<AnalyticsPeriod>('all')
  const [data, setData] = useState<AnalyticsSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (p: AnalyticsPeriod) => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchAnalytics(p))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '집계를 불러오지 못했습니다.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(period)
  }, [load, period])

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5">
      <div className="max-w-4xl mx-auto space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex gap-1">
            {PERIODS.map((p) => (
              <button
                key={p.value}
                onClick={() => setPeriod(p.value)}
                className={`text-xs px-3 py-1 rounded-md border transition-colors ${
                  period === p.value
                    ? 'bg-gray-800 text-gray-200 border-gray-600'
                    : 'text-gray-500 border-gray-800 hover:text-gray-300'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <button
            onClick={() => void load(period)}
            disabled={loading}
            className="text-xs px-3 py-1 rounded-md border border-gray-800 text-gray-500 hover:text-gray-300 disabled:opacity-40"
          >
            {loading ? '불러오는 중…' : '새로고침'}
          </button>
        </div>

        {error && (
          <div className="text-sm text-red-400 border border-red-900 bg-red-950 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        {data && data.requests.total === 0 && (
          <div className="text-sm text-gray-500 border border-gray-800 rounded-lg px-4 py-8 text-center">
            이 기간에 기록된 요청이 없습니다.
          </div>
        )}

        {data && data.requests.total > 0 && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="총 요청" value={data.requests.total.toLocaleString()}
                sub={`성공 ${pct(data.requests.successRate)}`} />
              <Stat label="캐시 히트율" value={pct(data.cache.hitRate)}
                sub={`LLM 호출 ${data.cache.llmCallsSkipped.toLocaleString()}회 생략`} />
              <Stat label="평균 응답" value={`${data.latency.avgMs.toLocaleString()} ms`}
                sub={`히트 ${data.latency.cacheHitAvgMs.toLocaleString()} / 미스 ${data.latency.cacheMissAvgMs.toLocaleString()} ms`} />
              <Stat label="단가 절감" value={pct(data.cost.savedRatio)}
                sub={`${usd(data.cost.actual)} vs ${usd(data.cost.alwaysOpus)}`} />
            </div>

            {data.cache.hits > 0 && (
              <Section title="캐시" hint="exact = 글자까지 같은 질문">
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <Stat label="시맨틱 히트" value={data.cache.semanticHits.toLocaleString()}
                    sub={`히트율 ${pct(data.cache.semanticHitRate)}`} />
                  <Stat label="exact 히트" value={data.cache.exactHits.toLocaleString()}
                    sub="임베딩 호출도 생략" />
                  <Stat label="전체" value={data.cache.hits.toLocaleString()}
                    sub={`히트율 ${pct(data.cache.hitRate)}`} />
                </div>
                {/* 두 값을 합친 히트율을 시맨틱 캐시 성능으로 제시하면 과대평가가 된다.
                    시연에서 나올 질문이라 화면에 미리 적어둔다. */}
                <p className="text-xs text-gray-600 leading-relaxed">
                  exact 히트는 질문이 글자까지 같아 임베딩 없이 답한 경우다. 시맨틱 캐시의
                  성능은 이를 분자·분모에서 모두 제외한 {pct(data.cache.semanticHitRate)} 로 본다.
                </p>
              </Section>
            )}

            <Section title="라우팅 분포" hint="LLM 을 실제 호출한 요청 기준 (캐시 히트 제외)">
              <div className="space-y-2">
                {data.byTier.map((t) => {
                  const cfg = MODEL_CONFIG[TIER_MODEL[t.tier]]
                  return (
                    // 실험 중 티어→모델 매핑을 바꾸면 같은 tier 가 모델별로 두 행이 된다.
                    <div key={`${t.tier}-${t.modelId}`} className="flex items-center gap-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full border font-medium w-20 text-center flex-shrink-0 ${cfg.badge}`}>
                        {cfg.label}
                      </span>
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gray-500 rounded-full"
                          style={{ width: `${t.share * 100}%` }} />
                      </div>
                      <span className="text-xs text-gray-400 w-16 text-right flex-shrink-0">
                        {pct(t.share)}
                      </span>
                      <span className="text-xs text-gray-600 w-32 text-right flex-shrink-0">
                        {t.count.toLocaleString()}건 · {usd(t.cost)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </Section>

            <Section title="비용" hint={`단가표 ${data.pricingVersion}`}>
              <div className="grid grid-cols-3 gap-3 mb-3">
                <Stat label="실제" value={usd(data.cost.actual)} />
                <Stat label="전부 Opus 였다면" value={usd(data.cost.alwaysOpus)} />
                <Stat label="차이" value={pct(data.cost.savedRatio)} />
              </div>
              {/* 이 숫자가 무엇을 비교한 것인지 화면에서 숨기지 않는다.
                  시연 때 가장 먼저 나올 질문이다. */}
              <p className="text-xs text-gray-600 leading-relaxed">{data.cost.note}</p>
            </Section>

            {(data.requests.escalation > 0 || data.errors.length > 0) && (
              <Section title="기타">
                <dl className="text-xs text-gray-400 space-y-1">
                  <Row label="에스컬레이션" value={`${data.requests.escalation.toLocaleString()}건`} />
                  {data.errors.map((e) => (
                    <Row key={e.status} label={`실패 (${e.status})`} value={`${e.count.toLocaleString()}건`} />
                  ))}
                </dl>
              </Section>
            )}
          </>
        )}

        {/* 재학습은 요청 집계와 달리 기간 필터를 받지 않는다. 배치는 누적 이력이라
            "오늘"로 잘라 보면 진행 중인 배치가 화면에서 사라진다. */}
        <Section title="재학습 피드백" hint="에스컬레이션 정정 라벨 → FastAPI">
          <RetrainingPanel />
        </Section>
      </div>
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border border-gray-800 rounded-lg px-4 py-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold text-gray-100 mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-1">{sub}</div>}
    </div>
  )
}

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <div className="border border-gray-800 rounded-lg px-4 py-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">{title}</h3>
        {hint && <span className="text-xs text-gray-600">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}
