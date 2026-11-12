import type { Run, RunEvent } from './types'

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(t)
      reject(signal.reason)
    })
  })

/**
 * Re-play a finished run as the event stream the live pipeline produced: one frame at a time,
 * with Claude's verdicts landing shortly after each frame. `speed` > 1 is faster than real time.
 */
export async function replay(
  run: Run,
  emit: (ev: RunEvent) => void,
  signal: AbortSignal,
  speed = 3,
): Promise<void> {
  const pending = run.candidates.map((c) => ({
    ...c,
    verification: null,
    verify_status: 'pending' as const,
  }))
  emit({
    type: 'run_started',
    data: { run_id: run.run_id, config: run.config, n_frames: run.frames.length },
  })
  const timers: Promise<void>[] = []
  for (const [i, frame] of run.frames.entries()) {
    // the first frame also paid for model warm-up; clamp so the replay keeps a steady beat
    await sleep(Math.min(1100, Math.max(300, (frame.detect_s * 1000) / speed)), signal)
    emit({
      type: 'frame_done',
      data: {
        frame,
        candidates: pending.filter((c) => c.frame_id === frame.id),
        progress: (i + 1) / run.frames.length,
      },
    })
    run.candidates
      .filter((c) => c.frame_id === frame.id)
      .forEach((c, j) => {
        // cached verdicts have no latency on record; use a typical 4 s Claude call
        const latency = (c.llm_latency_s || 4) * 1000
        timers.push(
          sleep(latency / speed + j * 120, signal)
            .then(() => emit({ type: 'candidate_verified', data: { candidate: c, cost_usd: 0 } }))
            .catch(() => {}), // aborted: the caller already moved on
        )
      })
  }
  await Promise.all(timers)
  signal.throwIfAborted()
  if (run.report_markdown) {
    await sleep(600, signal)
    emit({ type: 'report_ready', data: { markdown: run.report_markdown } })
  }
  emit({ type: 'run_finished', data: { run_id: run.run_id, stats: run.stats } })
}
