import { describe, expect, it } from 'vitest'
import { cropBox, iou } from './geometry'
import type { Box } from './types'

describe('cropBox', () => {
  // reference values from rescuelens.detect.crops.crop_box on a 4000x3000 frame
  it.each<[Box, Box]>([
    [
      [753, 1307, 73, 37],
      [644, 1180, 292, 292],
    ],
    [
      [5, 5, 20, 30],
      [0, 0, 120, 120],
    ],
    [
      [3950, 2980, 40, 15],
      [3840, 2840, 160, 160],
    ],
    [
      [1000, 1000, 1200, 900],
      [100, 0, 3000, 3000],
    ],
  ])('matches the Python crop window for %j', (det, expected) => {
    expect(cropBox(det, 4000, 3000)).toEqual(expected)
  })
})

describe('iou', () => {
  it('is 1 for identical boxes and 0 for disjoint ones', () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1)
    expect(iou([0, 0, 10, 10], [20, 20, 5, 5])).toBe(0)
  })
  it('handles partial overlap', () => {
    expect(iou([0, 0, 10, 10], [5, 0, 10, 10])).toBeCloseTo(50 / 150)
  })
})
