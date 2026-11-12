import type { Run, RunEvent, RunSummary, Source, Status } from './types'

/** Static build for GitHub Pages: no backend, one recorded run replayed in the browser. */
export const DEMO = import.meta.env.VITE_DEMO === '1'

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const detail = await res.json().then(
      (b: { detail?: string }) => b.detail,
      () => undefined,
    )
    throw new Error(detail ?? `${res.status} ${res.statusText}: ${url}`)
  }
  return (await res.json()) as T
}

export const listRuns = () => getJson<RunSummary[]>('/api/runs')
export const getStatus = () => getJson<Status>('/api/status')
export const listSources = () => getJson<Source[]>('/api/sources')

export const getRun = (id: string) =>
  DEMO
    ? getJson<Run>(`${import.meta.env.BASE_URL}demo/run.json`)
    : getJson<Run>(`/api/runs/${encodeURIComponent(id)}`)

export const assetUrl = (runId: string, relPath: string) =>
  DEMO
    ? `${import.meta.env.BASE_URL}demo/${relPath}`
    : `/api/runs/${encodeURIComponent(runId)}/${relPath}`

export const startAnalysis = (source: string) =>
  getJson<{ run_id: string }>('/api/analyze', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ source }),
  })

const EVENT_TYPES: RunEvent['type'][] = [
  'run_started',
  'frame_done',
  'candidate_verified',
  'report_ready',
  'run_finished',
  'error',
]

/** Follow a live analysis. Returns a function that closes the stream. */
export function followRun(
  runId: string,
  onEvent: (ev: RunEvent) => void,
  onBroken: () => void,
): () => void {
  const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`)
  let finished = false
  for (const type of EVENT_TYPES) {
    es.addEventListener(type, (msg) => {
      const ev = { type, data: JSON.parse((msg as MessageEvent<string>).data) } as RunEvent
      if (type === 'run_finished' || type === 'error') {
        finished = true
        es.close()
      }
      onEvent(ev)
    })
  }
  es.onerror = () => {
    if (finished) return
    es.close()
    onBroken()
  }
  return () => es.close()
}
