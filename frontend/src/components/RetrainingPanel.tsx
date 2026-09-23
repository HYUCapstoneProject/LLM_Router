import { useCallback, useEffect, useState } from 'react'
import type { RetrainingBatch, RetrainingOverview } from '../types'
import { ApiError, fetchRetrainingStatus, submitRetraining } from '../api/chatApi'

/**
 * 재학습 피드백 배치 운영 패널.
 *
 * 사용자 기능이 아니라 우리가 제출 시점을 정하기 위한 화면이다. 실사용자가 없어
 * 이벤트가 천천히 쌓이므로 임계값(1,000건)을 기다리는 대신 시연·실험 직전에
 * 사람이 눌러 내보낸다.
 */

/** 진행 중과 종료를 색으로 구분한다. 종료 중에서도 성공/거부/실패가 갈린다. */
const STATUS_STYLE: Record<string, { label: string; className: string }> = {
  submitted: { label: '제출 중', className: 'text-gray-400 border-gray-700' },
  queued: { label: '대기', className: 'text-sky-300 border-sky-900' },
  running: { label: '학습 중', className: 'text-sky-300 border-sky-900' },
  cancelling: { label: '취소 중', className: 'text-amber-300 border-amber-900' },
  paused: { label: '일시정지', className: 'text-amber-300 border-amber-900' },
  succeeded: { label: '성공', className: 'text-emerald-300 border-emerald-900' },
  rejected: { label: 'gate 거부', className: 'text-amber-300 border-amber-900' },
  failed: { label: '실패', className: 'text-red-400 border-red-900' },
  cancelled: { label: '취소됨', className: 'text-gray-400 border-gray-700' },
  unknown: { label: '확인 불가', className: 'text-gray-400 border-gray-700' },
}

const STAGE_LABEL: Record<string, string> = {
  initializing: '초기화',
  feedbackPreparation: '피드백 준비',
  feedbackMerge: '피드백 병합',
  training: '학습',
  evaluation: '평가',
  packaging: '패키징',
  gate: '배포 게이트',
  publishing: '게시',
  complete: '완료',
}

const time = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString('ko-KR', { dateStyle: 'short', timeStyle: 'short' }) : '—'

export function RetrainingPanel() {
  const [data, setData] = useState<RetrainingOverview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchRetrainingStatus())
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '재학습 상태를 불러오지 못했습니다.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    // 학습은 분 단위로 걸린다. 서버가 계약의 backoff 로 폴링하므로
    // 화면은 그 결과를 30초마다 다시 읽기만 한다.
    const timer = setInterval(() => void load(), 30_000)
    return () => clearInterval(timer)
  }, [load])

  const onSubmit = async () => {
    // 되돌릴 수 없는 동작이라 확인을 받는다.
    // 계약상 성공한 배치 하나가 모델 버전을 한 칸 올린다.
    const pending = data?.pendingEvents ?? 0
    const ok = window.confirm(
      `대기 중인 ${pending.toLocaleString()}건을 지금 재학습으로 보냅니다.\n\n` +
        '제출이 성공하면 라우터 모델 버전이 한 칸 올라가고 되돌릴 수 없습니다.\n' +
        '계속할까요?',
    )
    if (!ok) return

    setSubmitting(true)
    setError(null)
    setNotice(null)
    try {
      const result = await submitRetraining()
      if (!result) {
        setNotice('보낼 이벤트가 없습니다.')
      } else if (result.accepted) {
        // 202 는 접수됐다는 뜻이지 학습이 끝났다는 뜻이 아니다. 문구로 구분해 둔다.
        setNotice(
          `${result.batchId} 접수됨 (${result.eventCount.toLocaleString()}건). ` +
            '학습 결과는 아래 상태로 확인합니다.',
        )
      } else {
        setNotice(`제출 실패 (${result.failureCode ?? '원인 미상'}). 로그를 확인하세요.`)
      }
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '재학습 제출에 실패했습니다.')
    } finally {
      setSubmitting(false)
    }
  }

  const pending = data?.pendingEvents ?? 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-2xl text-gray-200">{pending.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-0.5">제출 대기 중인 정정 라벨</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => void load()}
            disabled={loading}
            className="text-xs px-3 py-1 rounded-md border border-gray-800 text-gray-500 hover:text-gray-300 disabled:opacity-40"
          >
            {loading ? '불러오는 중…' : '새로고침'}
          </button>
          <button
            onClick={() => void onSubmit()}
            disabled={submitting || pending === 0}
            className="text-xs px-3 py-1 rounded-md border border-sky-800 text-sky-300 hover:bg-sky-950 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            {submitting ? '제출 중…' : '지금 재학습 보내기'}
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-600 leading-relaxed">
        에스컬레이션으로 얻은 정정 라벨을 FastAPI 재학습 파이프라인에 보냅니다. 캐시 히트에서
        나온 불만족은 라우터의 과소 라우팅이 아니라 캐시 오탐이므로 대기 건수에서 제외됩니다.
      </p>

      {error && (
        <div className="text-sm text-red-400 border border-red-900 bg-red-950 rounded-lg px-4 py-3">
          {error}
        </div>
      )}
      {notice && (
        <div className="text-sm text-gray-300 border border-gray-800 rounded-lg px-4 py-3">
          {notice}
        </div>
      )}

      {data && data.recentBatches.length === 0 && (
        <div className="text-sm text-gray-500 border border-gray-800 rounded-lg px-4 py-8 text-center">
          아직 제출한 배치가 없습니다.
        </div>
      )}

      {data && data.recentBatches.length > 0 && (
        <div className="space-y-2">
          {data.recentBatches.map((batch) => (
            <BatchRow key={batch.batchId} batch={batch} />
          ))}
        </div>
      )}
    </div>
  )
}

function BatchRow({ batch }: { batch: RetrainingBatch }) {
  const style = STATUS_STYLE[batch.status] ?? STATUS_STYLE.unknown
  // 학습 성공과 serving 반영 성공은 다른 사건이다.
  // succeeded 인데 active 가 아니면 새 모델이 아직 요청을 받지 않는다는 뜻이라 따로 알린다.
  const activationWarning =
    batch.status === 'succeeded' && batch.activationStatus !== 'active'

  return (
    <div className="border border-gray-800 rounded-lg px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm text-gray-300 truncate">{batch.batchId}</p>
          <p className="text-xs text-gray-600 mt-0.5">
            {batch.eventCount.toLocaleString()}건 ·{' '}
            {batch.triggeredBy === 'manual' ? '수동' : '자동'} · {time(batch.submittedAt)}
          </p>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded border whitespace-nowrap ${style.className}`}>
          {style.label}
          {batch.stage && batch.status === 'running'
            ? ` · ${STAGE_LABEL[batch.stage] ?? batch.stage}`
            : ''}
        </span>
      </div>

      {batch.failureCode && (
        <p className="text-xs text-red-400 mt-2">원인 코드: {batch.failureCode}</p>
      )}
      {activationWarning && (
        <p className="text-xs text-amber-300 mt-2">
          학습은 성공했지만 서빙 반영이 끝나지 않았습니다 (
          {batch.activationStatus ?? '상태 없음'}). 현재 요청은 아직 이전 모델이 처리합니다.
        </p>
      )}
      {batch.status === 'succeeded' && !activationWarning && batch.candidateModelVersion && (
        <p className="text-xs text-gray-600 mt-2 truncate">
          반영된 모델: {batch.candidateModelVersion}
        </p>
      )}
    </div>
  )
}
