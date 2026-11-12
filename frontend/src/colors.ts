import type { Tier } from './run-state'

export const TIER_COLOR: Record<Tier, string> = {
  high: '#ff5a1f',
  medium: '#c38f22',
  pending: '#3a86d4',
  low: '#7e918a',
  rejected: '#7e918a',
}

export const TIER_LABEL: Record<Tier, string> = {
  high: 'Urgent',
  medium: 'Check soon',
  pending: 'Checking…',
  low: 'Unverified',
  rejected: 'Ruled out',
}
