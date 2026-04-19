import { useState, useEffect, useCallback, useRef } from 'react'
import { createDefaultLayer } from '../types/layers'
import type { Layer } from '../types/layers'

const API = ''
const POLL_INTERVAL_MS = 2000

async function apiGet(): Promise<Layer[]> {
  const res = await fetch(`${API}/layers`)
  if (!res.ok) throw new Error(`GET /layers ${res.status}`)
  return res.json()
}

async function apiPost(layer: Layer): Promise<Layer> {
  const res = await fetch(`${API}/layers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(layer),
  })
  if (!res.ok) throw new Error(`POST /layers ${res.status}`)
  return res.json()
}

async function apiPatch(id: string, updates: Partial<Layer>): Promise<void> {
  const res = await fetch(`${API}/layers/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  })
  if (!res.ok) throw new Error(`PATCH /layers/${id} ${res.status}`)
}

async function apiDelete(id: string): Promise<void> {
  const res = await fetch(`${API}/layers/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE /layers/${id} ${res.status}`)
}

async function apiDeleteAll(): Promise<void> {
  const res = await fetch(`${API}/layers`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE /layers ${res.status}`)
}

async function apiReorder(order: string[]): Promise<void> {
  const res = await fetch(`${API}/layers/reorder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ order }),
  })
  if (!res.ok) throw new Error(`POST /layers/reorder ${res.status}`)
}

export function useLayerStack() {
  const [layers, setLayers] = useState<Layer[]>([])
  const pendingOps = useRef(0)
  const layersRef = useRef<Layer[]>([])
  layersRef.current = layers

  const trackOp = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | null> => {
    pendingOps.current++
    try {
      return await fn()
    } catch (err) {
      console.error('Layer op failed:', err)
      return null
    } finally {
      pendingOps.current--
    }
  }, [])

  // Hydrate + poll
  useEffect(() => {
    let cancelled = false
    const sync = async () => {
      if (pendingOps.current > 0) return
      try {
        const server = await apiGet()
        if (cancelled || pendingOps.current > 0) return
        // Only update if actually different (shallow id+updatedAt-ish check)
        const sameLength = server.length === layersRef.current.length
        const sameIds = sameLength && server.every((l, i) => l.id === layersRef.current[i]?.id)
        const sameJson = sameIds && JSON.stringify(server) === JSON.stringify(layersRef.current)
        if (!sameJson) setLayers(server)
      } catch {
        /* ignore transient failures */
      }
    }
    sync()
    const interval = setInterval(sync, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const addLayer = useCallback(() => {
    const layer = createDefaultLayer(layersRef.current.length)
    setLayers(prev => [...prev, layer])
    void trackOp(() => apiPost(layer))
  }, [trackOp])

  const removeLayer = useCallback((id: string) => {
    setLayers(prev => prev.filter(l => l.id !== id))
    void trackOp(() => apiDelete(id))
  }, [trackOp])

  const updateLayer = useCallback((id: string, updates: Partial<Layer>) => {
    setLayers(prev => prev.map(l => l.id === id ? { ...l, ...updates } : l))
    void trackOp(() => apiPatch(id, updates))
  }, [trackOp])

  const moveLayer = useCallback((id: string, direction: 'up' | 'down') => {
    const prev = layersRef.current
    const idx = prev.findIndex(l => l.id === id)
    if (idx < 0) return
    const targetIdx = direction === 'up' ? idx + 1 : idx - 1
    if (targetIdx < 0 || targetIdx >= prev.length) return
    const next = [...prev]
    ;[next[idx], next[targetIdx]] = [next[targetIdx], next[idx]]
    setLayers(next)
    void trackOp(() => apiReorder(next.map(l => l.id)))
  }, [trackOp])

  const duplicateLayer = useCallback((id: string) => {
    const prev = layersRef.current
    const idx = prev.findIndex(l => l.id === id)
    if (idx < 0) return
    const source = prev[idx]
    const copy: Layer = {
      ...source,
      id: `layer-dup-${Date.now()}`,
      name: `${source.name} copy`,
    }
    const next = [...prev]
    next.splice(idx + 1, 0, copy)
    setLayers(next)
    // Duplicate on server requires source to exist server-side; POST the copy
    // at the end, then reorder to get it in the right spot.
    void trackOp(async () => {
      await apiPost(copy)
      await apiReorder(next.map(l => l.id))
    })
  }, [trackOp])

  const clearLayers = useCallback(() => {
    setLayers([])
    void trackOp(() => apiDeleteAll())
  }, [trackOp])

  return { layers, addLayer, removeLayer, updateLayer, moveLayer, duplicateLayer, clearLayers }
}
