import { iou } from './geometry'
import type { Candidate, Frame, Run, RunConfig, RunEvent, RunStats } from './types'

export type Phase = 'idle' | 'running' | 'done' | 'failed'

export interface RunState {
  runId: string | null
  phase: Phase
  config: RunConfig | null
  nFrames: number
  frames: Frame[]
  candidates: Candidate[]
  costUsd: number
  report: string | null
  stats: RunStats | null
  error: string | null
}

export const EMPTY: RunState = {
  runId: null,
  phase: 'idle',
  config: null,
  nFrames: 0,
  frames: [],
  candidates: [],
  costUsd: 0,
  report: null,
  stats: null,
  error: null,
}

export type Action =
  { kind: 'reset' } | { kind: 'load'; run: Run } | { kind: 'event'; ev: RunEvent }

function upsert<T extends { id: string }>(list: T[], items: T[]): T[] {
  const byId = new Map(list.map((x) => [x.id, x]))
  for (const it of items) byId.set(it.id, it)
  return [...byId.values()]
}

/** Events are applied idempotently: a reconnecting stream may resend the whole history. */
export function reduce(s: RunState, a: Action): RunState {
  if (a.kind === 'reset') return EMPTY
  if (a.kind === 'load') {
    const r = a.run
    return {
      runId: r.run_id,
      phase: r.status === 'running' ? 'running' : r.status,
      config: r.config,
      nFrames: r.frames.length,
      frames: r.frames,
      candidates: r.candidates,
      costUsd: r.stats.cost_usd,
      report: r.report_markdown,
      stats: r.stats,
      error: r.error,
    }
  }
  const ev = a.ev
  switch (ev.type) {
    case 'run_started':
      return {
        ...EMPTY,
        runId: ev.data.run_id,
        phase: 'running',
        config: ev.data.config,
        nFrames: ev.data.n_frames,
      }
    case 'frame_done':
      return {
        ...s,
        frames: upsert(s.frames, [ev.data.frame]),
        // keep any verdict that arrived first
        candidates: upsert(
          s.candidates,
          ev.data.candidates.filter((c) => !s.candidates.some((o) => o.id === c.id)),
        ),
      }
    case 'candidate_verified':
      return {
        ...s,
        candidates: upsert(s.candidates, [ev.data.candidate]),
        costUsd: Math.max(s.costUsd, ev.data.cost_usd),
      }
    case 'report_ready':
      return { ...s, report: ev.data.markdown }
    case 'run_finished':
      return { ...s, phase: 'done', stats: ev.data.stats, costUsd: ev.data.stats.cost_usd }
    case 'error':
      return { ...s, phase: 'failed', error: ev.data.message }
  }
}

// ---------- ranking ----------

export type Tier = 'high' | 'medium' | 'low' | 'rejected' | 'pending'

export function tier(c: Candidate): Tier {
  const v = c.verification
  if (!v) return c.verify_status === 'pending' ? 'pending' : 'low'
  if (!v.is_person) return 'rejected'
  return v.priority
}

const TIER_ORDER: Record<Tier, number> = { high: 0, medium: 1, pending: 2, low: 3, rejected: 4 }

/** Search order: Claude's priority, then its confidence, then the detector score. */
export function ranked(cands: Candidate[]): Candidate[] {
  return [...cands].sort(
    (a, b) =>
      TIER_ORDER[tier(a)] - TIER_ORDER[tier(b)] ||
      (b.verification?.confidence ?? 0) - (a.verification?.confidence ?? 0) ||
      b.yolo_score - a.yolo_score,
  )
}

/** Does this candidate overlap an annotated person? (null when the frame has no annotations) */
export function matchesAnnotation(c: Candidate, frame: Frame | undefined): boolean | null {
  if (!frame?.ground_truth) return null
  return frame.ground_truth.some((g) => iou(g, c.bbox) >= 0.5)
}

export { iou }
