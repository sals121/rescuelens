// Mirrors src/rescuelens/schemas.py and verify/schema.py.

export type GeoSource = 'exif' | 'simulated'
export type Posture = 'standing' | 'lying' | 'sitting' | 'unknown'
export type Priority = 'high' | 'medium' | 'low'
export type VerifyStatus = 'pending' | 'verified' | 'cached' | 'error' | 'skipped'

/** x, y, w, h in original-frame pixels */
export type Box = [number, number, number, number]

export interface Verification {
  is_person: boolean
  confidence: number
  evidence: string
  posture: Posture
  priority: Priority
}

export interface Frame {
  id: string
  filename: string
  width: number
  height: number
  lat: number
  lon: number
  alt_m: number | null
  geo_source: GeoSource
  preview: string
  n_candidates: number
  detect_s: number
  ground_truth: Box[] | null
}

export interface Candidate {
  id: string
  frame_id: string
  bbox: Box
  yolo_score: number
  lat: number
  lon: number
  geo_source: GeoSource
  crop: string
  verify_status: VerifyStatus
  verification: Verification | null
  verify_error: string | null
  llm_cost_usd: number
  llm_latency_s: number
}

export interface RunStats {
  n_frames: number
  n_candidates: number
  n_verified: number
  n_confirmed: number
  n_high_priority: number
  n_errors: number
  duration_s: number
  detect_s: number
  cost_usd: number
  cost_full_run_usd: number
  input_tokens: number
  output_tokens: number
  llm_calls: number
  cache_hits: number
}

export interface RunConfig {
  source_dir: string
  weights: string
  use_sahi: boolean
  confidence: number
  llm_enabled: boolean
  model: string | null
  llm_backend: string | null
  prompt_version: string | null
  geo_mode: GeoSource
}

export interface Run {
  run_id: string
  created_at: string
  status: 'running' | 'done' | 'failed'
  config: RunConfig
  stats: RunStats
  frames: Frame[]
  candidates: Candidate[]
  report_markdown: string | null
  error: string | null
}

export interface RunSummary {
  run_id: string
  created_at: string
  status: string
  stats: RunStats
  model: string | null
  geo_mode: string
}

export interface Source {
  name: string
  n_images: number
}

export interface Status {
  busy: boolean
  active_run_id: string | null
  weights: string
}

/** Server-sent events emitted by `run_analysis` (and re-created by the replay). */
export type RunEvent =
  | { type: 'run_started'; data: { run_id: string; config: RunConfig; n_frames: number } }
  | { type: 'frame_done'; data: { frame: Frame; candidates: Candidate[]; progress: number } }
  | { type: 'candidate_verified'; data: { candidate: Candidate; cost_usd: number } }
  | { type: 'report_ready'; data: { markdown: string } }
  | { type: 'run_finished'; data: { run_id: string; stats: RunStats } }
  | { type: 'error'; data: { message: string } }
