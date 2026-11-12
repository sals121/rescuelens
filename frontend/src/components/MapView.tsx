import {
  LngLatBounds,
  Map as MlMap,
  Marker,
  NavigationControl,
  setWorkerUrl,
  type GeoJSONSource,
  type MapLayerMouseEvent,
  type StyleSpecification,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url'
import { useEffect, useRef, useState } from 'react'
import type { FeatureCollection } from 'geojson'
import { TIER_COLOR } from '../colors'
import { tier } from '../run-state'
import type { Candidate, Frame } from '../types'

// MapLibre finds its worker next to its own module; a bundled build must point to it explicitly
setWorkerUrl(workerUrl)

const SATELLITE_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    imagery: {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      // Esri serves 'map data not yet available' tiles past z17 in rural areas; overzoom instead
      maxzoom: 17,
      attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    },
  },
  layers: [{ id: 'imagery', type: 'raster', source: 'imagery' }],
}

interface Props {
  frames: Frame[]
  /** already in search order; index + 1 is the rank shown on the pin */
  ranked: Candidate[]
  selectedId: string | null
  selectedFrameId: string | null
  onSelect: (id: string) => void
  onSelectFrame: (frameId: string) => void
}

/**
 * Real EXIF only gives the drone's position, so every candidate of a photo shares one point.
 * Pins are fanned out around it in screen space (the coordinates themselves stay honest).
 */
function fanOffset(index: number, count: number): [number, number] {
  if (count <= 1) return [0, 0]
  const r = 16 + 3 * count
  const a = (2 * Math.PI * index) / count - Math.PI / 2
  return [Math.round(r * Math.cos(a)), Math.round(r * Math.sin(a))]
}

export function MapView(props: Props) {
  const { frames, ranked, selectedId, selectedFrameId } = props
  const container = useRef<HTMLDivElement>(null)
  const [map, setMap] = useState<MlMap | null>(null)
  const markers = useRef(new Map<string, { marker: Marker; el: HTMLElement }>())
  const fittedFrames = useRef(0)
  const cb = useRef(props)
  useEffect(() => {
    cb.current = props
  })

  useEffect(() => {
    if (!container.current) return
    const m = new MlMap({
      container: container.current,
      style: SATELLITE_STYLE,
      center: [16.6, 43.3],
      zoom: 4,
      attributionControl: { compact: true },
    })
    m.addControl(new NavigationControl({ showCompass: false }), 'top-right')
    // 'load' waits for every satellite tile; the style is all we need to add our layers
    m.once('style.load', () => {
      m.addSource('track', { type: 'geojson', data: emptyFC() })
      m.addLayer({
        id: 'track-line',
        type: 'line',
        source: 'track',
        filter: ['==', '$type', 'LineString'],
        paint: { 'line-color': '#ffffff', 'line-width': 2, 'line-opacity': 0.85 },
      })
      m.addLayer({
        id: 'track-points',
        type: 'circle',
        source: 'track',
        filter: ['==', '$type', 'Point'],
        paint: {
          'circle-radius': ['case', ['get', 'selected'], 7, 4],
          'circle-color': ['case', ['get', 'selected'], '#18221f', '#ffffff'],
          'circle-stroke-color': '#18221f',
          'circle-stroke-width': 1.5,
        },
      })
      m.on('click', 'track-points', (e: MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id as string | undefined
        if (id) cb.current.onSelectFrame(id)
      })
      m.on('mouseenter', 'track-points', () => (m.getCanvas().style.cursor = 'pointer'))
      m.on('mouseleave', 'track-points', () => (m.getCanvas().style.cursor = ''))
      setMap(m)
    })
    const live = markers.current
    return () => {
      live.forEach(({ marker }) => marker.remove())
      live.clear()
      setMap(null)
      m.remove()
    }
  }, [])

  // flight track
  useEffect(() => {
    if (!map) return
    const coords = frames.map((f) => [f.lon, f.lat])
    ;(map.getSource('track') as GeoJSONSource | undefined)?.setData({
      type: 'FeatureCollection',
      features: [
        ...(coords.length > 1
          ? [
              {
                type: 'Feature' as const,
                properties: {},
                geometry: { type: 'LineString' as const, coordinates: coords },
              },
            ]
          : []),
        ...frames.map((f) => ({
          type: 'Feature' as const,
          properties: { id: f.id, selected: f.id === selectedFrameId },
          geometry: { type: 'Point' as const, coordinates: [f.lon, f.lat] },
        })),
      ],
    })
    if (frames.length > 0 && frames.length !== fittedFrames.current) {
      const b = new LngLatBounds()
      frames.forEach((f) => b.extend([f.lon, f.lat]))
      map.fitBounds(b, {
        padding: { top: 70, bottom: 70, left: 70, right: 70 },
        maxZoom: 18.5,
        duration: fittedFrames.current ? 600 : 0,
      })
      fittedFrames.current = frames.length
    }
  }, [map, frames, selectedFrameId])

  // candidate pins
  useEffect(() => {
    if (!map) return
    const perFrame = new Map<string, Candidate[]>()
    ranked.forEach((c) => perFrame.set(c.frame_id, [...(perFrame.get(c.frame_id) ?? []), c]))
    const seen = new Set<string>()
    ranked.forEach((c, i) => {
      seen.add(c.id)
      const t = tier(c)
      const siblings = perFrame.get(c.frame_id) ?? []
      const offset = fanOffset(siblings.indexOf(c), siblings.length)
      let entry = markers.current.get(c.id)
      if (!entry) {
        // MapLibre owns the element's transform and classes; we style a child instead
        const el = document.createElement('button')
        el.type = 'button'
        el.appendChild(document.createElement('span'))
        el.addEventListener('click', (e) => {
          e.stopPropagation()
          cb.current.onSelect(c.id)
        })
        const marker = new Marker({ element: el, offset }).setLngLat([c.lon, c.lat]).addTo(map)
        entry = { marker, el }
        markers.current.set(c.id, entry)
      }
      const { el, marker } = entry
      marker.setOffset(offset)
      const big = t === 'high' || t === 'medium'
      const shape = el.firstElementChild as HTMLSpanElement
      shape.className = big ? 'pin' : 'pin-dot'
      shape.style.background = TIER_COLOR[t]
      shape.dataset.selected = String(c.id === selectedId)
      shape.textContent = big ? String(i + 1) : ''
      el.style.zIndex = c.id === selectedId ? '10' : big ? '2' : '1'
      el.setAttribute('aria-label', `Candidate ${i + 1}, ${c.id}`)
    })
    markers.current.forEach(({ marker }, id) => {
      if (!seen.has(id)) {
        marker.remove()
        markers.current.delete(id)
      }
    })
  }, [map, ranked, selectedId])

  return <div ref={container} className="h-full w-full" />
}

function emptyFC(): FeatureCollection {
  return { type: 'FeatureCollection', features: [] }
}
