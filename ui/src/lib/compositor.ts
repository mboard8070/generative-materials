import type { Layer, BlendMode, PbrChannel, PbrMaps } from '../types/layers'
import { PBR_CHANNELS } from '../types/layers'
import { generatePerlinNoise, generateVoronoiNoise } from './noise'

const SIZE = 512

// Canvas-native blend modes
const NATIVE_BLENDS: Partial<Record<BlendMode, GlobalCompositeOperation>> = {
  'normal': 'source-over',
  'multiply': 'multiply',
  'screen': 'screen',
  'overlay': 'overlay',
  'soft-light': 'soft-light',
  'difference': 'difference',
  'darken': 'darken',
  'lighten': 'lighten',
  'color-dodge': 'color-dodge',
  'color-burn': 'color-burn',
}

// --- Pixel-level blend functions for non-native modes ---

function blendPixels(
  base: Uint8ClampedArray,
  layer: Uint8ClampedArray,
  mode: BlendMode,
  opacity: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(base.length)
  for (let i = 0; i < base.length; i += 4) {
    const a = layer[i + 3] / 255 * opacity
    for (let c = 0; c < 3; c++) {
      const bv = base[i + c]
      const lv = layer[i + c]
      let result: number
      switch (mode) {
        case 'add':
          result = bv + lv
          break
        case 'subtract':
          result = bv - lv
          break
        case 'divide':
          result = lv > 0 ? (bv / lv) * 255 : 255
          break
        default:
          result = lv
      }
      result = Math.max(0, Math.min(255, result))
      out[i + c] = Math.round(bv * (1 - a) + result * a)
    }
    out[i + 3] = 255
  }
  return out
}

function isCustomBlend(mode: BlendMode): boolean {
  return mode === 'add' || mode === 'subtract' || mode === 'divide'
}

// --- Helpers ---

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff]
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    if (!url.startsWith('data:')) img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load: ${url.slice(0, 80)}`))
    img.src = url
  })
}

function generateMask(layer: Layer): Float32Array {
  const mask = layer.mask.noiseType === 'voronoi'
    ? generateVoronoiNoise(SIZE, SIZE, Math.max(4, Math.round(layer.mask.scale * 10)), layer.mask.seed)
    : generatePerlinNoise(SIZE, SIZE, Math.max(1, layer.mask.scale * 16), layer.mask.seed)
  if (layer.mask.invert) {
    for (let i = 0; i < mask.length; i++) mask[i] = 1 - mask[i]
  }
  return mask
}

function applyMaskToImageData(data: Uint8ClampedArray, mask: Float32Array) {
  for (let i = 0; i < SIZE * SIZE; i++) {
    data[i * 4 + 3] = Math.round(data[i * 4 + 3] * mask[i])
  }
}

// Render solid/noise layer content as ImageData
function renderFillContent(layer: Layer): ImageData {
  const data = new Uint8ClampedArray(SIZE * SIZE * 4)
  if (layer.type === 'solid') {
    const [r, g, b] = hexToRgb(layer.color)
    for (let i = 0; i < SIZE * SIZE; i++) {
      data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255
    }
  } else {
    const noiseData = layer.noiseType === 'voronoi'
      ? generateVoronoiNoise(SIZE, SIZE, Math.max(4, Math.round(layer.noiseScale * 10)), layer.noiseSeed)
      : generatePerlinNoise(SIZE, SIZE, Math.max(1, layer.noiseScale * 16), layer.noiseSeed)
    const [r1, g1, b1] = hexToRgb(layer.noiseColor1)
    const [r2, g2, b2] = hexToRgb(layer.noiseColor2)
    for (let i = 0; i < SIZE * SIZE; i++) {
      const t = noiseData[i]
      data[i * 4] = r1 + (r2 - r1) * t
      data[i * 4 + 1] = g1 + (g2 - g1) * t
      data[i * 4 + 2] = b1 + (b2 - b1) * t
      data[i * 4 + 3] = 255
    }
  }
  return new ImageData(data, SIZE, SIZE)
}

// --- Blend one layer onto a canvas for a specific channel ---

async function blendLayerOntoCanvas(
  _mainCanvas: HTMLCanvasElement,
  mainCtx: CanvasRenderingContext2D,
  layer: Layer,
  channel: PbrChannel,
  mask: Float32Array | null,
) {
  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = SIZE; tempCanvas.height = SIZE
  const tempCtx = tempCanvas.getContext('2d')!

  let content: ImageData

  if (layer.type === 'material' && layer.materialMaps[channel]) {
    // Load the generated map image
    const img = await loadImage(layer.materialMaps[channel]!)
    tempCtx.drawImage(img, 0, 0, SIZE, SIZE)
    content = tempCtx.getImageData(0, 0, SIZE, SIZE)
  } else if (layer.type === 'solid' || layer.type === 'noise') {
    content = renderFillContent(layer)
  } else {
    return // material layer with no map for this channel — skip
  }

  // Apply mask
  if (mask) applyMaskToImageData(content.data, mask)

  if (isCustomBlend(layer.blendMode)) {
    // Pixel-level blend
    const baseData = mainCtx.getImageData(0, 0, SIZE, SIZE)
    const result = blendPixels(baseData.data, content.data, layer.blendMode, layer.opacity)
    const imgData = mainCtx.createImageData(SIZE, SIZE)
    imgData.data.set(result)
    mainCtx.putImageData(imgData, 0, 0)
  } else {
    // Canvas-native blend
    tempCtx.clearRect(0, 0, SIZE, SIZE)
    tempCtx.putImageData(content, 0, 0)
    mainCtx.globalCompositeOperation = NATIVE_BLENDS[layer.blendMode] ?? 'source-over'
    mainCtx.globalAlpha = layer.opacity
    mainCtx.drawImage(tempCanvas, 0, 0)
    mainCtx.globalCompositeOperation = 'source-over'
    mainCtx.globalAlpha = 1
  }
}

// --- Public: composite all layers per PBR channel ---

export interface CompositedMaps {
  basecolor: string | null
  normal: string | null
  roughness: string | null
  metalness: string | null
  height: string | null
  translucency: string | null
  subsurface: string | null
}

export async function compositePbrLayers(
  layers: Layer[],
  baseMaps: PbrMaps,
): Promise<CompositedMaps> {
  const visible = layers.filter(l => l.visible)
  const result: CompositedMaps = {
    basecolor: null, normal: null, roughness: null, metalness: null, height: null,
    translucency: null, subsurface: null,
  }

  // Pre-compute masks (same mask used across all channels for a layer)
  const masks = new Map<string, Float32Array | null>()
  for (const layer of visible) {
    masks.set(layer.id, layer.mask.enabled ? generateMask(layer) : null)
  }

  for (const channel of PBR_CHANNELS) {
    const canvas = document.createElement('canvas')
    canvas.width = SIZE; canvas.height = SIZE
    const ctx = canvas.getContext('2d')!

    // Draw base map
    const baseUrl = baseMaps[channel]
    if (baseUrl) {
      try {
        const img = await loadImage(baseUrl)
        ctx.drawImage(img, 0, 0, SIZE, SIZE)
      } catch { /* no base for this channel */ }
    }

    // Layer each visible layer that has this channel enabled
    let hasLayerContent = false
    for (const layer of visible) {
      if (!layer.channels[channel].enabled) continue

      // For material layers, skip if no map generated for this channel
      if (layer.type === 'material' && !layer.materialMaps[channel]) continue

      await blendLayerOntoCanvas(canvas, ctx, layer, channel, masks.get(layer.id) ?? null)
      hasLayerContent = true
    }

    if (baseUrl || hasLayerContent) {
      result[channel] = canvas.toDataURL('image/png')
    }
  }

  return result
}

// Legacy: albedo-only compositing for backward compat
export async function compositeLayerStack(
  layers: Layer[],
  baseImageUrl?: string | null,
): Promise<string | null> {
  const maps: PbrMaps = { basecolor: baseImageUrl ?? undefined }
  const result = await compositePbrLayers(layers, maps)
  return result.basecolor
}
