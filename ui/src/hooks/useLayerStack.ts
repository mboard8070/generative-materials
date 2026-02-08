import { useState, useCallback } from 'react'
import { createDefaultLayer } from '../types/layers'
import type { Layer } from '../types/layers'

export function useLayerStack() {
  const [layers, setLayers] = useState<Layer[]>([])

  const addLayer = useCallback(() => {
    setLayers(prev => [...prev, createDefaultLayer(prev.length)])
  }, [])

  const removeLayer = useCallback((id: string) => {
    setLayers(prev => prev.filter(l => l.id !== id))
  }, [])

  const updateLayer = useCallback((id: string, updates: Partial<Layer>) => {
    setLayers(prev => prev.map(l => l.id === id ? { ...l, ...updates } : l))
  }, [])

  const moveLayer = useCallback((id: string, direction: 'up' | 'down') => {
    setLayers(prev => {
      const idx = prev.findIndex(l => l.id === id)
      if (idx < 0) return prev
      const targetIdx = direction === 'up' ? idx + 1 : idx - 1
      if (targetIdx < 0 || targetIdx >= prev.length) return prev
      const next = [...prev]
      ;[next[idx], next[targetIdx]] = [next[targetIdx], next[idx]]
      return next
    })
  }, [])

  const duplicateLayer = useCallback((id: string) => {
    setLayers(prev => {
      const idx = prev.findIndex(l => l.id === id)
      if (idx < 0) return prev
      const source = prev[idx]
      const copy: Layer = {
        ...source,
        id: `layer-dup-${Date.now()}`,
        name: `${source.name} copy`,
      }
      const next = [...prev]
      next.splice(idx + 1, 0, copy)
      return next
    })
  }, [])

  const clearLayers = useCallback(() => {
    setLayers([])
  }, [])

  return { layers, addLayer, removeLayer, updateLayer, moveLayer, duplicateLayer, clearLayers }
}
