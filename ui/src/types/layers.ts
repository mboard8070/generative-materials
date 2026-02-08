export type LayerType = 'solid' | 'noise'
export type NoiseType = 'perlin' | 'voronoi'
export type BlendMode = 'normal' | 'multiply' | 'screen' | 'overlay' | 'soft-light' | 'difference'

export interface LayerMask {
  enabled: boolean
  noiseType: NoiseType
  scale: number
  seed: number
  invert: boolean
}

export interface Layer {
  id: string
  name: string
  visible: boolean
  opacity: number
  blendMode: BlendMode
  type: LayerType
  // Solid fill
  color: string
  // Noise fill
  noiseType: NoiseType
  noiseScale: number
  noiseSeed: number
  noiseColor1: string
  noiseColor2: string
  // Mask
  mask: LayerMask
}

export const BLEND_MODES: { value: BlendMode; label: string }[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'multiply', label: 'Multiply' },
  { value: 'screen', label: 'Screen' },
  { value: 'overlay', label: 'Overlay' },
  { value: 'soft-light', label: 'Soft Light' },
  { value: 'difference', label: 'Difference' },
]

let nextId = 1

export function createDefaultLayer(index: number): Layer {
  const id = `layer-${nextId++}-${Date.now()}`
  return {
    id,
    name: `Layer ${index + 1}`,
    visible: true,
    opacity: 1.0,
    blendMode: 'normal',
    type: 'solid',
    color: '#888888',
    noiseType: 'perlin',
    noiseScale: 4,
    noiseSeed: Math.floor(Math.random() * 10000),
    noiseColor1: '#000000',
    noiseColor2: '#ffffff',
    mask: {
      enabled: false,
      noiseType: 'perlin',
      scale: 4,
      seed: Math.floor(Math.random() * 10000),
      invert: false,
    },
  }
}
