import type { Run, RunSummary } from './types'

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`)
  return (await res.json()) as T
}

export const listRuns = () => getJson<RunSummary[]>('/api/runs')
export const getRun = (id: string) => getJson<Run>(`/api/runs/${encodeURIComponent(id)}`)
export const assetUrl = (runId: string, relPath: string) =>
  `/api/runs/${encodeURIComponent(runId)}/${relPath}`
