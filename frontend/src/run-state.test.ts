import { describe, expect, it } from 'vitest'
import { EMPTY, matchesAnnotation, ranked, reduce, tier } from './run-state'
import type { Candidate, Frame, RunConfig, RunEvent, Verification } from './types'

const config: RunConfig = {
  source_dir: 'demo',
  weights: 'w.pt',
  use_sahi: true,
  confidence: 0.4,
  llm_enabled: true,
  model: 'claude-opus-5-5',
  llm_backend: 'claude-code',
  prompt_version: 'v1',
  geo_mode: 'exif',
}

const frame: Frame = {
  id: 'F001',
  filename: 'a.jpg',
  width: 4000,
  height: 3000,
  lat: 43.3,
  lon: 16.6,
  alt_m: 50,
  geo_source: 'exif',
  preview: 'frames/F001.jpg',
  n_candidates: 2,
  detect_s: 1,
  ground_truth: [[100, 100, 40, 40]],
}

function cand(id: string, score: number, v: Partial<Verification> | null = null): Candidate {
  return {
    id,
    frame_id: 'F001',
    bbox: [100, 100, 40, 40],
    yolo_score: score,
    lat: 43.3,
    lon: 16.6,
    geo_source: 'exif',
    crop: `crops/${id}.jpg`,
    verify_status: v ? 'verified' : 'pending',
    verification: v
      ? {
          is_person: true,
          confidence: 0.9,
          evidence: '',
          posture: 'lying',
          priority: 'high',
          ...v,
        }
      : null,
    verify_error: null,
    llm_cost_usd: 0,
    llm_latency_s: 0,
  }
}

describe('reduce', () => {
  it('builds a run from events and tolerates a replayed history', () => {
    const events: RunEvent[] = [
      { type: 'run_started', data: { run_id: 'r', config, n_frames: 1 } },
      { type: 'frame_done', data: { frame, candidates: [cand('F001-C01', 0.8)], progress: 1 } },
      {
        type: 'candidate_verified',
        data: { candidate: cand('F001-C01', 0.8, {}), cost_usd: 0.02 },
      },
    ]
    let s = EMPTY
    for (const ev of events) s = reduce(s, { kind: 'event', ev })
    // a reconnecting stream resends frame_done after the verdict: the verdict must survive
    s = reduce(s, { kind: 'event', ev: events[1]! })
    expect(s.phase).toBe('running')
    expect(s.candidates).toHaveLength(1)
    expect(s.candidates[0]?.verification?.is_person).toBe(true)
    expect(s.costUsd).toBe(0.02)
  })

  it('records failures', () => {
    const s = reduce(EMPTY, { kind: 'event', ev: { type: 'error', data: { message: 'boom' } } })
    expect(s.phase).toBe('failed')
    expect(s.error).toBe('boom')
  })
})

describe('ranking', () => {
  it('puts urgent people first and ruled-out candidates last', () => {
    const rock = cand('rock', 0.99, { is_person: false, priority: 'low' })
    const urgent = cand('urgent', 0.3, { priority: 'high' })
    const medium = cand('medium', 0.9, { priority: 'medium' })
    const pending = cand('pending', 0.5)
    expect(ranked([rock, medium, pending, urgent]).map((c) => c.id)).toEqual([
      'urgent',
      'medium',
      'pending',
      'rock',
    ])
    expect(tier(rock)).toBe('rejected')
  })

  it('matches candidates against dataset annotations', () => {
    expect(matchesAnnotation(cand('a', 0.5), frame)).toBe(true)
    expect(matchesAnnotation(cand('a', 0.5), { ...frame, ground_truth: null })).toBeNull()
  })
})
