import type { Box } from './types'

/** Intersection over union of two (x, y, w, h) boxes. */
export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]))
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy
  return inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

/** Same window as `crop_box` in src/rescuelens/detect/crops.py (the close-up sent to Claude). */
export function cropBox([x, y, w, h]: Box, imgW: number, imgH: number): Box {
  let side = Math.trunc(Math.max(w, h) * 4)
  side = Math.max(96, Math.min(side, imgW, imgH))
  const left = Math.min(Math.max(Math.round(x + w / 2 - side / 2), 0), imgW - side)
  const top = Math.min(Math.max(Math.round(y + h / 2 - side / 2), 0), imgH - side)
  return [left, top, side, side]
}
