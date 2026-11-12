import { useEffect, useMemo, useState } from 'react'
import { assetUrl, getRun, listRuns } from './api'
import { PRIORITY_COLOR } from './colors'
import { MapView } from './components/MapView'
import type { Candidate, Run } from './types'

export default function App() {
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    listRuns()
      .then((runs) => {
        const latest = runs[0]
        if (!latest) throw new Error('No runs yet — run `rescuelens analyze <folder>` first.')
        return getRun(latest.run_id)
      })
      .then(setRun)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const sorted = useMemo(() => (run ? [...run.candidates].sort(byPriority) : []), [run])

  if (error) return <Centered className="text-red-400">{error}</Centered>
  if (!run) return <Centered>loading…</Centered>

  const s = run.stats
  const simulated = run.config.geo_mode === 'simulated'
  return (
    <div className="flex h-full flex-col font-mono text-sm">
      <header className="flex items-center gap-6 border-b border-zinc-800 bg-zinc-950 px-4 py-2">
        <span className="tracking-widest text-amber-400">RESCUELENS</span>
        <Stat label="frames" value={s.n_frames} />
        <Stat label="candidates" value={s.n_candidates} />
        <Stat label="verified" value={s.n_verified} />
        <Stat label="people" value={s.n_confirmed} />
        <Stat label="time" value={`${s.duration_s.toFixed(1)}s`} />
        <Stat label="cost" value={`$${s.cost_usd.toFixed(3)}`} />
        {simulated && (
          <span className="ml-auto rounded border border-amber-500/50 px-2 py-0.5 text-xs text-amber-400">
            SIMULATED GPS
          </span>
        )}
      </header>
      <div className="flex min-h-0 flex-1">
        <main className="relative flex-1">
          <MapView
            frames={run.frames}
            candidates={run.candidates}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </main>
        <aside className="w-96 overflow-y-auto border-l border-zinc-800 bg-zinc-950">
          {sorted.map((c) => (
            <CandidateRow
              key={c.id}
              c={c}
              runId={run.run_id}
              selected={c.id === selectedId}
              onClick={() => setSelectedId(c.id)}
            />
          ))}
        </aside>
      </div>
    </div>
  )
}

function CandidateRow(props: {
  c: Candidate
  runId: string
  selected: boolean
  onClick: () => void
}) {
  const { c, runId, selected, onClick } = props
  const v = c.verification
  const color = v
    ? v.is_person
      ? PRIORITY_COLOR[v.priority]
      : PRIORITY_COLOR.low
    : PRIORITY_COLOR.unverified
  return (
    <button
      onClick={onClick}
      className={`flex w-full gap-3 border-b border-zinc-900 p-3 text-left hover:bg-zinc-900 ${selected ? 'bg-zinc-900' : ''}`}
    >
      <img src={assetUrl(runId, c.crop)} alt={c.id} className="h-20 w-20 shrink-0 rounded" />
      <div className="min-w-0 space-y-1">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: color }} />
          <span className="text-zinc-200">{c.id}</span>
          <span className="text-zinc-500">yolo {c.yolo_score.toFixed(2)}</span>
        </div>
        {v ? (
          <>
            <div className="text-xs text-zinc-300">
              {v.is_person ? 'PERSON' : 'not a person'} · {v.posture} ·{' '}
              {(v.confidence * 100).toFixed(0)}%
            </div>
            <p className="line-clamp-2 text-xs text-zinc-500">{v.evidence}</p>
          </>
        ) : (
          <div className="text-xs text-zinc-500">{c.verify_error ?? c.verify_status}</div>
        )}
      </div>
    </button>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="text-zinc-400">
      {label} <span className="text-zinc-100">{value}</span>
    </span>
  )
}

function Centered({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`flex h-full items-center justify-center font-mono ${className}`}>
      {children}
    </div>
  )
}

const RANK = { high: 0, medium: 1, low: 2 } as const
function byPriority(a: Candidate, b: Candidate): number {
  const ra = a.verification ? (a.verification.is_person ? RANK[a.verification.priority] : 3) : 2.5
  const rb = b.verification ? (b.verification.is_person ? RANK[b.verification.priority] : 3) : 2.5
  return ra - rb || b.yolo_score - a.yolo_score
}
