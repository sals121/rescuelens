import {
  LngLatBounds,
  Map as MlMap,
  NavigationControl,
  type GeoJSONSource,
  type MapLayerMouseEvent,
  type StyleSpecification,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useEffect, useRef, useState } from 'react'
import type { FeatureCollection } from 'geojson'
import { PRIORITY_COLOR } from '../colors'
import type { Candidate, Frame } from '../types'

const SATELLITE_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    imagery: {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      maxzoom: 19,
      attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    },
  },
  layers: [
    { id: 'imagery', type: 'raster', source: 'imagery', paint: { 'raster-saturation': -0.35 } },
  ],
}

function candidateColorKey(c: Candidate): keyof typeof PRIORITY_COLOR {
  if (!c.verification) return 'unverified'
  return c.verification.is_person ? c.verification.priority : 'low'
}

interface Props {
  frames: Frame[]
  candidates: Candidate[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function MapView({ frames, candidates, selectedId, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const map = useRef<MlMap | null>(null)
  // The map instance whose style + sources are ready; data effects depend on it.
  const [ready, setReady] = useState<MlMap | null>(null)
  const fitted = useRef(false)
  const onSelectRef = useRef(onSelect)
  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    if (!container.current) return
    const m = new MlMap({
      container: container.current,
      style: SATELLITE_STYLE,
      center: [0, 42],
      zoom: 3,
      attributionControl: { compact: true },
    })
    m.addControl(new NavigationControl({ showCompass: false }), 'bottom-right')
    m.on('load', () => {
      m.addSource('track', { type: 'geojson', data: emptyFC() })
      m.addSource('candidates', { type: 'geojson', data: emptyFC() })
      m.addLayer({
        id: 'track-line',
        type: 'line',
        source: 'track',
        filter: ['==', '$type', 'LineString'],
        paint: { 'line-color': '#e4e4e7', 'line-width': 1.5, 'line-dasharray': [2, 2] },
      })
      m.addLayer({
        id: 'track-points',
        type: 'circle',
        source: 'track',
        filter: ['==', '$type', 'Point'],
        paint: { 'circle-radius': 3, 'circle-color': '#e4e4e7', 'circle-opacity': 0.7 },
      })
      m.addLayer({
        id: 'candidate-dots',
        type: 'circle',
        source: 'candidates',
        paint: {
          'circle-radius': ['case', ['get', 'selected'], 10, 7],
          'circle-color': ['get', 'color'],
          'circle-stroke-color': '#09090b',
          'circle-stroke-width': 2,
        },
      })
      m.on('click', 'candidate-dots', (e: MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id as string | undefined
        if (id) onSelectRef.current(id)
      })
      m.on('mouseenter', 'candidate-dots', () => (m.getCanvas().style.cursor = 'pointer'))
      m.on('mouseleave', 'candidate-dots', () => (m.getCanvas().style.cursor = ''))
      setReady(m)
    })
    map.current = m
    fitted.current = false
    return () => {
      setReady(null)
      m.remove()
    }
  }, [])

  // Track + candidates data
  useEffect(() => {
    const m = ready
    if (!m) return
    {
      const coords = frames.map((f) => [f.lon, f.lat])
      ;(m.getSource('track') as GeoJSONSource | undefined)?.setData({
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
          ...coords.map((c) => ({
            type: 'Feature' as const,
            properties: {},
            geometry: { type: 'Point' as const, coordinates: c },
          })),
        ],
      })
      ;(m.getSource('candidates') as GeoJSONSource | undefined)?.setData({
        type: 'FeatureCollection',
        features: candidates.map((c) => ({
          type: 'Feature' as const,
          properties: {
            id: c.id,
            color: PRIORITY_COLOR[candidateColorKey(c)],
            selected: c.id === selectedId,
          },
          geometry: { type: 'Point' as const, coordinates: [c.lon, c.lat] },
        })),
      })
    }
  }, [ready, frames, candidates, selectedId])

  // Fit to the flight once frames arrive
  useEffect(() => {
    const m = ready
    if (!m || frames.length === 0 || fitted.current) return
    const bounds = new LngLatBounds()
    frames.forEach((f) => bounds.extend([f.lon, f.lat]))
    m.fitBounds(bounds, { padding: 80, maxZoom: 17, duration: 1200 })
    fitted.current = true
  }, [ready, frames])

  return <div ref={container} className="h-full w-full" />
}

function emptyFC(): FeatureCollection {
  return { type: 'FeatureCollection', features: [] }
}
