export type LayerType = 'material' | 'solid' | 'noise'
export type NoiseType = 'perlin' | 'voronoi'

export type BlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'soft-light'
  | 'difference'
  | 'add'
  | 'subtract'
  | 'divide'
  | 'darken'
  | 'lighten'
  | 'color-dodge'
  | 'color-burn'

export const BLEND_MODES: { value: BlendMode; label: string }[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'add', label: 'Add' },
  { value: 'subtract', label: 'Subtract' },
  { value: 'multiply', label: 'Multiply' },
  { value: 'divide', label: 'Divide' },
  { value: 'screen', label: 'Screen' },
  { value: 'overlay', label: 'Overlay' },
  { value: 'soft-light', label: 'Soft Light' },
  { value: 'darken', label: 'Darken' },
  { value: 'lighten', label: 'Lighten' },
  { value: 'color-dodge', label: 'Color Dodge' },
  { value: 'color-burn', label: 'Color Burn' },
  { value: 'difference', label: 'Difference' },
]

export const PBR_CHANNELS = ['basecolor', 'normal', 'roughness', 'metalness', 'height', 'translucency', 'subsurface'] as const
export type PbrChannel = typeof PBR_CHANNELS[number]

export interface PbrMaps {
  basecolor?: string
  normal?: string
  roughness?: string
  metalness?: string
  height?: string
  translucency?: string
  subsurface?: string
}

export interface ChannelSettings {
  enabled: boolean
}

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

  // Material layer — generated PBR maps
  materialPrompt: string
  materialMaps: PbrMaps
  generating: boolean

  // Per-channel enable/disable
  channels: Record<PbrChannel, ChannelSettings>

  // Solid fill (used for solid type, also tints material basecolor)
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

let nextId = 1

export function createDefaultLayer(index: number): Layer {
  const id = `layer-${nextId++}-${Date.now()}`
  return {
    id,
    name: `Layer ${index + 1}`,
    visible: true,
    opacity: 1.0,
    blendMode: 'normal',
    type: 'material',

    materialPrompt: '',
    materialMaps: {},
    generating: false,

    channels: {
      basecolor: { enabled: true },
      normal: { enabled: true },
      roughness: { enabled: true },
      metalness: { enabled: true },
      height: { enabled: true },
      translucency: { enabled: false },
      subsurface: { enabled: false },
    },

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
