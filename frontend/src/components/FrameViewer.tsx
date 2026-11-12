import { useEffect, useRef, useState } from 'react'
import { assetUrl } from '../api'
import { TIER_COLOR } from '../colors'
import { cropBox } from '../geometry'
import { tier } from '../run-state'
import type { Candidate, Frame } from '../types'

interface Props {
  runId: string
  frame: Frame
  candidates: Candidate[] // of this frame
  rankOf: Map<string, number>
  selected: Candidate | null
  onSelect: (id: string) => void
}

const GT_PAD = 10 // px in the original frame, so annotation and detection outlines stay apart

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => {
      if (e) setSize({ w: e.contentRect.width, h: e.contentRect.height })
    })
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}

/**
 * The full drone photo with every detection outlined. Selecting a candidate zooms the photo
 * in on it: a person is often under 0.1 % of the frame, so the whole-photo view alone hides it.
 */
export function FrameViewer({ runId, frame, candidates, rankOf, selected, onSelect }: Props) {
  const [box, { w, h }] = useSize<HTMLDivElement>()
  const [zoomed, setZoomed] = useState(true)
  const [prevSelected, setPrevSelected] = useState(selected?.id)
  if (selected?.id !== prevSelected) {
    setPrevSelected(selected?.id)
    setZoomed(true)
  }

  const fit = w && h ? Math.min(w / frame.width, h / frame.height) : 0
  const dw = frame.width * fit
  const dh = frame.height * fit
  const ox = (w - dw) / 2
  const oy = (h - dh) / 2

  let transform = 'translate(0px, 0px) scale(1)'
  let k = 1
  const focus = zoomed && selected && selected.frame_id === frame.id ? selected : null
  if (focus && fit) {
    const [x, y, bw, bh] = focus.bbox
    k = Math.min(8, Math.max(2, (h * 0.3) / (Math.max(bw, bh) * fit)))
    const cx = (x + bw / 2) * fit
    const cy = (y + bh / 2) * fit
    transform = `translate(${w / 2 - ox - k * cx}px, ${h / 2 - oy - k * cy}px) scale(${k})`
  }
  // stroke widths in original-frame pixels that render at a fixed width on screen
  const px = (n: number) => (fit ? n / (fit * k) : n)
  const sharp = focus ? cropBox(focus.bbox, frame.width, frame.height) : null

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline gap-3 px-4 pt-3 pb-2 text-sm">
        <h2 className="font-semibold">Photo {frame.id.replace(/^F0*/, '')}</h2>
        <span className="text-ink-soft hidden truncate sm:inline">{frame.filename}</span>
        <span className="text-ink-soft">
          {frame.width}×{frame.height}
          {frame.alt_m != null && `, ${Math.round(frame.alt_m)} m up`}
        </span>
        {selected?.frame_id === frame.id && (
          <button
            type="button"
            onClick={() => setZoomed((z) => !z)}
            className="border-ink/20 hover:bg-ink hover:text-paper ml-auto shrink-0 rounded-full border px-3 py-0.5"
          >
            {zoomed ? 'Show whole photo' : 'Zoom to candidate'}
          </button>
        )}
      </div>
      <div ref={box} className="bg-ink relative min-h-0 flex-1 overflow-hidden">
        {fit > 0 && (
          <div
            className="loupe absolute origin-top-left"
            style={{ left: ox, top: oy, width: dw, height: dh, transform }}
          >
            <img
              src={assetUrl(runId, frame.preview)}
              alt={`Drone photo ${frame.filename}`}
              className="h-full w-full select-none"
              draggable={false}
            />
            <svg
              viewBox={`0 0 ${frame.width} ${frame.height}`}
              className="absolute inset-0 h-full w-full"
            >
              {focus && sharp && (
                // the full-resolution close-up, laid over the downscaled preview
                <image
                  href={assetUrl(runId, focus.crop)}
                  x={sharp[0]}
                  y={sharp[1]}
                  width={sharp[2]}
                  height={sharp[3]}
                />
              )}
              {frame.ground_truth?.map(([x, y, bw, bh], i) => (
                <rect
                  key={`gt${i}`}
                  x={x - GT_PAD}
                  y={y - GT_PAD}
                  width={bw + 2 * GT_PAD}
                  height={bh + 2 * GT_PAD}
                  fill="none"
                  stroke="#fff"
                  strokeWidth={px(1.5)}
                  strokeDasharray={`${px(6)} ${px(4)}`}
                />
              ))}
              {!focus &&
                candidates.map((c) => (
                  // a person is a few pixels wide at this scale: ring it so the eye finds it
                  <circle
                    key={`ring${c.id}`}
                    cx={c.bbox[0] + c.bbox[2] / 2}
                    cy={c.bbox[1] + c.bbox[3] / 2}
                    r={px(16)}
                    fill="none"
                    stroke={TIER_COLOR[tier(c)]}
                    strokeOpacity={0.9}
                    strokeWidth={px(2)}
                  />
                ))}
              {candidates.map((c) => {
                const [x, y, bw, bh] = c.bbox
                const sel = c.id === selected?.id
                // outset covers the thin magenta outline drawn into Claude's close-up
                const out = (Math.max(bw, bh) * 4 * 3) / 384 + px(1)
                return (
                  <rect
                    key={c.id}
                    x={x - out}
                    y={y - out}
                    width={bw + 2 * out}
                    height={bh + 2 * out}
                    fill="transparent"
                    stroke={TIER_COLOR[tier(c)]}
                    strokeWidth={px(sel ? 3 : 2)}
                    className="cursor-pointer"
                    onClick={() => onSelect(c.id)}
                  >
                    <title>
                      #{rankOf.get(c.id)} {c.id}
                    </title>
                  </rect>
                )
              })}
            </svg>
          </div>
        )}
      </div>
      <p className="text-ink-soft flex flex-wrap gap-x-4 gap-y-1 px-4 py-2 text-xs">
        <Key color={TIER_COLOR.high}>urgent</Key>
        <Key color={TIER_COLOR.medium}>check soon</Key>
        <Key color={TIER_COLOR.rejected}>ruled out by Claude</Key>
        {frame.ground_truth && <Key dashed>person annotated in the dataset</Key>}
      </p>
    </div>
  )
}

function Key({
  color,
  dashed,
  children,
}: {
  color?: string
  dashed?: boolean
  children: React.ReactNode
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-3.5 border-2"
        style={{
          borderColor: color ?? 'var(--color-ink-soft)',
          borderStyle: dashed ? 'dashed' : 'solid',
        }}
      />
      {children}
    </span>
  )
}
