import { useEffect, useRef } from 'react'
import { assetUrl } from '../api'
import { TIER_COLOR, TIER_LABEL } from '../colors'
import { matchesAnnotation, tier } from '../run-state'
import type { Candidate, Frame } from '../types'

interface Props {
  runId: string
  ranked: Candidate[]
  frames: Map<string, Frame>
  selectedId: string | null
  onSelect: (id: string) => void
}

const POSTURE: Record<string, string> = {
  lying: 'lying down',
  sitting: 'sitting',
  standing: 'standing',
  unknown: 'posture unclear',
}

export function SearchList({ runId, ranked, frames, selectedId, onSelect }: Props) {
  const active = ranked.filter((c) => tier(c) !== 'rejected')
  const rejected = ranked.filter((c) => tier(c) === 'rejected')
  return (
    <div>
      <ol>
        {active.map((c, i) => (
          <Row
            key={c.id}
            rank={i + 1}
            c={c}
            runId={runId}
            frame={frames.get(c.frame_id)}
            selected={c.id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </ol>
      {rejected.length > 0 && (
        <details className="group border-rule border-t">
          <summary className="text-ink-soft hover:text-ink cursor-pointer px-4 py-3 text-sm">
            {rejected.length} ruled out by Claude (rocks, bags, shadows…)
          </summary>
          <ol>
            {rejected.map((c, i) => (
              <Row
                key={c.id}
                rank={active.length + i + 1}
                c={c}
                runId={runId}
                frame={frames.get(c.frame_id)}
                selected={c.id === selectedId}
                onSelect={onSelect}
              />
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}

function Row(props: {
  rank: number
  c: Candidate
  runId: string
  frame: Frame | undefined
  selected: boolean
  onSelect: (id: string) => void
}) {
  const { rank, c, runId, frame, selected, onSelect } = props
  const ref = useRef<HTMLLIElement>(null)
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selected])

  const t = tier(c)
  const v = c.verification
  const annotated = matchesAnnotation(c, frame)
  const headline = v
    ? v.is_person
      ? `Person, ${POSTURE[v.posture]}`
      : 'Not a person'
    : c.verify_status === 'pending'
      ? 'Waiting for Claude'
      : 'Not verified'

  return (
    <li ref={ref}>
      <button
        type="button"
        onClick={() => onSelect(c.id)}
        aria-current={selected}
        className={`border-rule hover:bg-sheet grid w-full grid-cols-[2.25rem_4.5rem_1fr] gap-3 border-t px-4 py-3 text-left transition-colors ${
          selected ? 'bg-sheet shadow-[inset_3px_0_0_var(--color-ink)]' : ''
        }`}
      >
        <span
          className="wordmark text-2xl leading-none"
          style={{
            color: t === 'rejected' || t === 'low' ? 'var(--color-contour)' : TIER_COLOR[t],
          }}
        >
          {rank}
        </span>
        <img
          src={assetUrl(runId, c.crop)}
          alt=""
          className="h-[4.5rem] w-[4.5rem] rounded-sm object-cover"
          loading="lazy"
        />
        <span className="min-w-0 space-y-1">
          <span className="flex items-baseline gap-2">
            <span className="font-semibold">{headline}</span>
            <span
              className="condensed shrink-0 text-xs font-semibold"
              style={{ color: TIER_COLOR[t] }}
            >
              {TIER_LABEL[t]}
            </span>
          </span>
          {v && <span className="text-ink-soft line-clamp-2 text-sm">{v.evidence}</span>}
          <span className="condensed text-contour block text-xs">
            Photo {c.frame_id.replace(/^F0*/, '')}, detector {Math.round(c.yolo_score * 100)}%
            {v && `, Claude ${Math.round(v.confidence * 100)}%`}
            {annotated !== null && (annotated ? ', annotated person' : ', not annotated')}
          </span>
        </span>
      </button>
    </li>
  )
}
