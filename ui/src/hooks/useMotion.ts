import { useState, useEffect, useCallback, useRef } from 'react'
import type { MotionState, MotionConfig, AnimatedProp } from '../types/motion'

const API = ''
const POLL_INTERVAL_MS = 2000

async function apiGet(): Promise<MotionState> {
  const res = await fetch(`${API}/motion`)
  if (!res.ok) throw new Error(`GET /motion ${res.status}`)
  return res.json()
}

async function apiPatch(updates: Partial<Record<AnimatedProp, MotionConfig | null>>): Promise<void> {
  const res = await fetch(`${API}/motion`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  })
  if (!res.ok) throw new Error(`PATCH /motion ${res.status}`)
}

async function apiClear(): Promise<void> {
  const res = await fetch(`${API}/motion`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE /motion ${res.status}`)
}

export function useMotion() {
  const [motion, setMotion] = useState<MotionState>({})
  const pendingOps = useRef(0)
  const motionRef = useRef<MotionState>({})
  motionRef.current = motion

  const trackOp = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | null> => {
    pendingOps.current++
    try {
      return await fn()
    } catch (err) {
      console.error('Motion op failed:', err)
      return null
    } finally {
      pendingOps.current--
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const sync = async () => {
      if (pendingOps.current > 0) return
      try {
        const server = await apiGet()
        if (cancelled || pendingOps.current > 0) return
        if (JSON.stringify(server) !== JSON.stringify(motionRef.current)) {
          setMotion(server)
        }
      } catch { /* ignore */ }
    }
    sync()
    const interval = setInterval(sync, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const setProp = useCallback((prop: AnimatedProp, cfg: MotionConfig | null) => {
    setMotion(prev => {
      const next = { ...prev }
      if (cfg === null) delete next[prop]
      else next[prop] = cfg
      return next
    })
    void trackOp(() => apiPatch({ [prop]: cfg }))
  }, [trackOp])

  const updateProp = useCallback((prop: AnimatedProp, updates: Partial<MotionConfig>) => {
    setMotion(prev => {
      const current = prev[prop]
      if (!current) return prev
      const merged: MotionConfig = { ...current, ...updates }
      return { ...prev, [prop]: merged }
    })
    const current = motionRef.current[prop]
    if (current) {
      const merged: MotionConfig = { ...current, ...updates }
      void trackOp(() => apiPatch({ [prop]: merged }))
    }
  }, [trackOp])

  const clearAll = useCallback(() => {
    setMotion({})
    void trackOp(() => apiClear())
  }, [trackOp])

  return { motion, setProp, updateProp, clearAll }
}
