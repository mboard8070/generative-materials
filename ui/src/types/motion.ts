export type AnimMode = 'static' | 'sin' | 'cos' | 'linear'

export interface MotionConfig {
  mode: AnimMode
  base: number   // center (for sin/cos) or start value (for linear)
  amp: number    // amplitude (sin/cos) — ignored for linear/static
  freq: number   // cycles/sec (sin/cos) or rate per sec (linear)
  phase: number  // radians (sin/cos) — ignored for linear/static
}

export const ANIMATED_PROPS = [
  'pan_x',
  'pan_y',
  'displacement',
  'transmission',
  'ior',
  'emissive',
  'roughness',
  'metalness',
] as const

export type AnimatedProp = typeof ANIMATED_PROPS[number]

export type MotionState = Partial<Record<AnimatedProp, MotionConfig>>

export const PROP_LABELS: Record<AnimatedProp, string> = {
  pan_x: 'Pan X',
  pan_y: 'Pan Y',
  displacement: 'Displacement',
  transmission: 'Transmission',
  ior: 'IOR',
  emissive: 'Emissive',
  roughness: 'Roughness',
  metalness: 'Metalness',
}

export function defaultMotion(prop: AnimatedProp): MotionConfig {
  // Pan defaults to linear scroll, everything else to static sin with small amp
  if (prop === 'pan_x' || prop === 'pan_y') {
    return { mode: 'linear', base: 0, amp: 0, freq: 0.05, phase: 0 }
  }
  return { mode: 'sin', base: 0, amp: 0.1, freq: 0.5, phase: 0 }
}

export function evaluateMotion(cfg: MotionConfig | undefined, t: number, fallback: number): number {
  if (!cfg || cfg.mode === 'static') return fallback
  const TAU = Math.PI * 2
  switch (cfg.mode) {
    case 'sin':
      return cfg.base + cfg.amp * Math.sin(TAU * cfg.freq * t + cfg.phase)
    case 'cos':
      return cfg.base + cfg.amp * Math.cos(TAU * cfg.freq * t + cfg.phase)
    case 'linear':
      return cfg.base + cfg.freq * t
  }
}
