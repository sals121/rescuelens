import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { DEMO, followRun, getRun, getStatus, listRuns, listSources, startAnalysis } from './api'
import { FrameViewer } from './components/FrameViewer'
import { MapView } from './components/MapView'
import { ReportPanel } from './components/ReportPanel'
import { SearchList } from './components/SearchList'
import { replay } from './replay'
import { EMPTY, iou, ranked as rank, reduce, tier, type RunState } from './run-state'
import type { Run, RunSummary, Source } from './types'

const MODEL_NAME: Record<string, string> = {
  'claude-opus-5-5': 'Claude Opus 5.5',
  'claude-haiku-5-5': 'Claude Haiku 5.5',
  'claude-sonnet-5-5': 'Claude Sonnet 5.5',
}

export default function App() {
  const [state, dispatch] = useReducer(reduce, EMPTY)
  const [recorded, setRecorded] = useState<Run | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [source, setSource] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [frameId, setFrameId] = useState<string | null>(null)
  const [tab, setTab] = useState<'list' | 'report'>('list')
  const [replaying, setReplaying] = useState(false)
  const stop = useRef<() => void>(() => {})

  const stopAll = () => {
    stop.current()
    stop.current = () => {}
    setReplaying(false)
  }

  const startReplay = useCallback((run: Run) => {
    stop.current()
    const ctl = new AbortController()
    stop.current = () => ctl.abort()
    setSelectedId(null)
    setFrameId(null)
    setTab('list')
    setReplaying(true)
    replay(run, (ev) => dispatch({ kind: 'event', ev }), ctl.signal)
      .catch(() => {})
      .finally(() => setReplaying(false))
  }, [])

  const showRun = useCallback((run: Run) => {
    stop.current()
    setRecorded(run)
    setSelectedId(null)
    setFrameId(null)
    dispatch({ kind: 'load', run })
  }, [])

  const follow = useCallback((runId: string) => {
    stop.current()
    setSelectedId(null)
    setFrameId(null)
    setTab('list')
    stop.current = followRun(
      runId,
      (ev) => {
        dispatch({ kind: 'event', ev })
        if (ev.type === 'run_finished')
          void getRun(runId).then((r) => {
            setRecorded(r)
            void listRuns().then(setRuns)
          })
      },
      () => setNotice('Lost the connection to the analysis. Reload the page to see the result.'),
    )
  }, [])

  // first load
  useEffect(() => {
    if (DEMO) {
      getRun('demo')
        .then((r) => {
          setRecorded(r)
          startReplay(r)
        })
        .catch((e: unknown) => setNotice(String(e)))
      return () => stop.current()
    }
    Promise.all([getStatus(), listRuns(), listSources()])
      .then(([status, rs, srcs]) => {
        setRuns(rs)
        setSources(srcs)
        setSource(srcs[0]?.name ?? '')
        if (status.active_run_id) follow(status.active_run_id)
        else if (rs[0]) return getRun(rs[0].run_id).then(showRun)
      })
      .catch(() =>
        setNotice(
          'The RescueLens backend is not answering on port 8731. Start it with `just run`.',
        ),
      )
    return () => stop.current()
  }, [follow, showRun, startReplay])

  const analyse = async () => {
    setNotice(null)
    try {
      const { run_id } = await startAnalysis(source)
      follow(run_id)
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e))
    }
  }

  const ordered = useMemo(() => rank(state.candidates), [state.candidates])
  const rankOf = useMemo(() => new Map(ordered.map((c, i) => [c.id, i + 1])), [ordered])
  const framesById = useMemo(() => new Map(state.frames.map((f) => [f.id, f])), [state.frames])

  // keep something selected: the top candidate once the run is done
  const selected = ordered.find((c) => c.id === selectedId) ?? null
  const top = state.phase === 'done' ? ordered[0] : undefined
  const effective = selected ?? top ?? null
  const frame =
    framesById.get(effective?.frame_id ?? '') ??
    framesById.get(frameId ?? '') ??
    state.frames[state.frames.length - 1]

  const select = (id: string) => {
    setSelectedId(id)
    setFrameId(null)
  }
  const selectFrame = (id: string) => {
    setFrameId(id)
    setSelectedId(null)
  }

  return (
    <div className="grid h-full grid-rows-[auto_auto_1fr]">
      <Header
        state={state}
        notice={notice}
        controls={
          <Controls
            replaying={replaying}
            canReplay={!!recorded && state.phase !== 'running'}
            onReplay={() => recorded && startReplay(recorded)}
            onSkip={() => {
              stopAll()
              if (recorded) dispatch({ kind: 'load', run: recorded })
            }}
            runs={runs}
            currentRun={state.runId}
            onPickRun={(id) => void getRun(id).then(showRun)}
            sources={sources}
            source={source}
            onSource={setSource}
            onAnalyse={analyse}
            busy={state.phase === 'running' && !replaying}
          />
        }
      />
      <Progress state={state} />
      <div className="grid min-h-0 grid-rows-[45vh_60vh_auto] overflow-y-auto lg:grid-cols-[minmax(340px,420px)_1fr] lg:grid-rows-1 lg:overflow-hidden">
        <aside className="border-rule bg-paper order-3 flex min-h-0 flex-col lg:order-none lg:border-r">
          <nav className="flex gap-1 px-3 pt-3" aria-label="Panels">
            <TabButton active={tab === 'list'} onClick={() => setTab('list')}>
              Search order
            </TabButton>
            <TabButton active={tab === 'report'} onClick={() => setTab('report')}>
              Report
            </TabButton>
          </nav>
          <div className="min-h-0 flex-1 overflow-y-auto pt-2">
            {tab === 'list' ? (
              state.runId && ordered.length > 0 ? (
                <SearchList
                  runId={state.runId}
                  ranked={ordered}
                  frames={framesById}
                  selectedId={effective?.id ?? null}
                  onSelect={select}
                />
              ) : (
                <p className="text-ink-soft px-4 py-6 text-sm">
                  {state.phase === 'running'
                    ? 'Candidates appear here as each photo is scanned.'
                    : 'Pick a folder of drone photos and press Analyse.'}
                </p>
              )
            ) : (
              <ReportPanel
                markdown={state.report}
                candidateIds={new Set(state.candidates.map((c) => c.id))}
                onSelect={select}
              />
            )}
          </div>
        </aside>
        <main className="contents lg:grid lg:min-h-0 lg:grid-rows-[1fr_1fr]">
          <section className="relative min-h-0" aria-label="Map">
            <MapView
              frames={state.frames}
              ranked={ordered}
              selectedId={effective?.id ?? null}
              selectedFrameId={frame?.id ?? null}
              onSelect={select}
              onSelectFrame={selectFrame}
            />
            {state.config && (
              <p className="bg-ink/75 text-paper pointer-events-none absolute bottom-2 left-2 rounded px-2 py-1 text-xs">
                {state.config.geo_mode === 'simulated'
                  ? 'Simulated flight path: these photos carry no GPS'
                  : "Drone GPS at each photo. A person is within that photo's footprint."}
              </p>
            )}
          </section>
          <section className="border-rule bg-paper min-h-0 border-t" aria-label="Photo">
            {frame && state.runId ? (
              <FrameViewer
                runId={state.runId}
                frame={frame}
                candidates={state.candidates.filter((c) => c.frame_id === frame.id)}
                rankOf={rankOf}
                selected={effective}
                onSelect={select}
              />
            ) : (
              <div className="bg-ink text-paper/70 grid h-full place-items-center text-sm">
                Photos appear here as they are scanned.
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  )
}

function Header(props: { state: RunState; notice: string | null; controls: React.ReactNode }) {
  const { state, notice, controls } = props
  const people = state.candidates.filter((c) => c.verification?.is_person)
  const urgent = people.filter((c) => tier(c) === 'high').length
  const checked = state.candidates.filter((c) => c.verification).length
  const model = state.config?.model ? (MODEL_NAME[state.config.model] ?? state.config.model) : null

  let headline: string
  if (state.phase === 'running')
    headline = `Scanning photo ${Math.min(state.frames.length + 1, state.nFrames)} of ${state.nFrames}. ${checked} of ${state.candidates.length} candidates checked.`
  else if (state.phase === 'failed')
    headline = `The analysis stopped: ${state.error ?? 'unknown error'}`
  else if (state.runId)
    headline = `${people.length} likely ${people.length === 1 ? 'person' : 'people'} in ${state.nFrames} photos, ${urgent} urgent.`
  else headline = 'No analysis yet.'

  // honesty line: how many annotated people the run actually found
  const gt = state.frames.reduce((n, f) => n + (f.ground_truth?.length ?? 0), 0)
  const kept = state.candidates.filter((c) => tier(c) !== 'rejected')
  const found = state.frames
    .flatMap((f) => (f.ground_truth ?? []).map((g) => ({ f, g })))
    .filter(({ f, g }) => kept.some((c) => c.frame_id === f.id && iou(g, c.bbox) >= 0.5)).length

  return (
    <header className="border-rule bg-sheet flex flex-wrap items-center gap-x-6 gap-y-2 border-b px-4 py-3">
      <a
        href="https://github.com/sals121/rescuelens"
        className="wordmark text-ink mr-auto text-xl lg:mr-0"
      >
        RescueLens
      </a>
      <div className="order-last w-full min-w-0 lg:order-none lg:w-auto lg:flex-1">
        <p className="text-[0.95rem] font-semibold">{headline}</p>
        <p className="text-ink-soft text-xs">
          {notice ? (
            <span className="text-signal">{notice}</span>
          ) : (
            <>
              Detector: YOLO26s fine-tuned on HERIDAL, sliced with SAHI.
              {model &&
                (state.phase === 'done'
                  ? ` Each candidate checked by ${model} (${
                      state.config?.llm_backend === 'claude-code'
                        ? 'Claude Code subscription, '
                        : ''
                    }$${(state.stats?.cost_full_run_usd || state.costUsd).toFixed(2)} at API prices).`
                  : ` Each candidate checked by ${model}.`)}
              {state.phase === 'done' &&
                gt > 0 &&
                ` Dataset annotations: found ${found} of ${gt} people.`}
              {state.config?.source_dir.startsWith('heridal') && (
                <>
                  {' '}
                  Photos:{' '}
                  <a
                    className="underline underline-offset-2"
                    href="https://doi.org/10.5281/zenodo.5662351"
                  >
                    HERIDAL
                  </a>{' '}
                  (Božić-Štulić, Marušić, Gotovac), CC BY 3.0.
                </>
              )}
            </>
          )}
        </p>
      </div>
      {controls}
    </header>
  )
}

function Controls(props: {
  replaying: boolean
  canReplay: boolean
  onReplay: () => void
  onSkip: () => void
  runs: RunSummary[]
  currentRun: string | null
  onPickRun: (id: string) => void
  sources: Source[]
  source: string
  onSource: (s: string) => void
  onAnalyse: () => void
  busy: boolean
}) {
  const btn =
    'rounded-full border border-ink/25 px-3 py-1 text-sm font-medium hover:bg-ink hover:text-paper disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-ink'
  return (
    <div className="flex flex-wrap items-center gap-2">
      {props.replaying ? (
        <button type="button" className={btn} onClick={props.onSkip}>
          Skip to result
        </button>
      ) : (
        <button type="button" className={btn} onClick={props.onReplay} disabled={!props.canReplay}>
          Replay the search
        </button>
      )}
      {!DEMO && (
        <>
          {props.runs.length > 1 && (
            <select
              aria-label="Previous analyses"
              className="border-ink/25 rounded-full border bg-transparent px-2 py-1 text-sm"
              value={props.currentRun ?? ''}
              onChange={(e) => props.onPickRun(e.target.value)}
            >
              {props.runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id}
                </option>
              ))}
            </select>
          )}
          <select
            aria-label="Photo folder"
            className="border-ink/25 rounded-full border bg-transparent px-2 py-1 text-sm"
            value={props.source}
            onChange={(e) => props.onSource(e.target.value)}
          >
            {props.sources.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} ({s.n_images} photos)
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={props.onAnalyse}
            disabled={props.busy || !props.source}
            className="bg-signal rounded-full px-4 py-1 text-sm font-semibold text-white hover:brightness-95 disabled:opacity-40"
          >
            {props.busy ? 'Analysing…' : 'Analyse photos'}
          </button>
        </>
      )}
    </div>
  )
}

function Progress({ state }: { state: RunState }) {
  if (state.phase !== 'running' || state.nFrames === 0) return <div className="h-[3px]" />
  const frac = state.frames.length / state.nFrames
  return (
    <div
      className="bg-rule relative h-[3px] overflow-hidden"
      role="progressbar"
      aria-valuenow={Math.round(frac * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Photos scanned"
    >
      <div
        className="bg-signal h-full transition-[width] duration-300"
        style={{ width: `${frac * 100}%` }}
      />
      <div className="sweep absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-white/60 to-transparent" />
    </div>
  )
}

function TabButton(props: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={props.onClick}
      aria-pressed={props.active}
      className={`rounded-full px-3 py-1 text-sm font-medium ${
        props.active ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'
      }`}
    >
      {props.children}
    </button>
  )
}
