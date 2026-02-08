import type { Layer, BlendMode } from '../types/layers'
import { generatePerlinNoise, generateVoronoiNoise } from './noise'

const SIZE = 512

const BLEND_MAP: Record<BlendMode, GlobalCompositeOperation> = {
  'normal': 'source-over',
  'multiply': 'multiply',
  'screen': 'screen',
  'overlay': 'overlay',
  'soft-light': 'soft-light',
  'difference': 'difference',
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff]
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    // data URLs don't need CORS; cross-origin server URLs do
    if (!url.startsWith('data:')) {
      img.crossOrigin = 'anonymous'
    }
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image: ${url.slice(0, 80)}`))
    img.src = url
  })
}

function renderLayerContent(layer: Layer): ImageData {
  const data = new Uint8ClampedArray(SIZE * SIZE * 4)

  if (layer.type === 'solid') {
    const [r, g, b] = hexToRgb(layer.color)
    for (let i = 0; i < SIZE * SIZE; i++) {
      data[i * 4] = r
      data[i * 4 + 1] = g
      data[i * 4 + 2] = b
      data[i * 4 + 3] = 255
    }
  } else {
    // Noise layer — generate noise and map to color gradient
    const noiseData =
      layer.noiseType === 'voronoi'
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

function applyMask(imageData: ImageData, layer: Layer): ImageData {
  if (!layer.mask.enabled) return imageData

  const mask =
    layer.mask.noiseType === 'voronoi'
      ? generateVoronoiNoise(SIZE, SIZE, Math.max(4, Math.round(layer.mask.scale * 10)), layer.mask.seed)
      : generatePerlinNoise(SIZE, SIZE, Math.max(1, layer.mask.scale * 16), layer.mask.seed)

  const data = imageData.data
  for (let i = 0; i < SIZE * SIZE; i++) {
    let m = mask[i]
    if (layer.mask.invert) m = 1 - m
    data[i * 4 + 3] = Math.round(data[i * 4 + 3] * m)
  }

  return imageData
}

/**
 * Composite all visible layers onto an optional base image.
 * Returns a data URL (png) or null if nothing to draw.
 */
export async function compositeLayerStack(
  layers: Layer[],
  baseImageUrl?: string | null
): Promise<string | null> {
  const visible = layers.filter(l => l.visible)
  // No layers → return null so preview falls back to textureUrl directly
  if (visible.length === 0) return null

  const mainCanvas = document.createElement('canvas')
  mainCanvas.width = SIZE
  mainCanvas.height = SIZE
  const mainCtx = mainCanvas.getContext('2d')!

  // Draw the generated texture as the base layer
  let hasContent = false
  if (baseImageUrl) {
    try {
      const img = await loadImage(baseImageUrl)
      mainCtx.drawImage(img, 0, 0, SIZE, SIZE)
      hasContent = true
    } catch {
      // base image failed to load — continue without it
    }
  }

  if (visible.length === 0) {
    // No layers — just return the base image (or null)
    return hasContent ? mainCanvas.toDataURL('image/png') : null
  }

  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = SIZE
  tempCanvas.height = SIZE
  const tempCtx = tempCanvas.getContext('2d')!

  for (const layer of visible) {
    // Render content
    let content = renderLayerContent(layer)
    // Apply mask
    content = applyMask(content, layer)

    // Draw content onto temp canvas
    tempCtx.clearRect(0, 0, SIZE, SIZE)
    tempCtx.putImageData(content, 0, 0)

    // Blend modes like multiply against transparent = invisible,
    // so force source-over when the main canvas is still empty
    if (!hasContent) {
      mainCtx.globalCompositeOperation = 'source-over'
      hasContent = true
    } else {
      mainCtx.globalCompositeOperation = BLEND_MAP[layer.blendMode]
    }
    mainCtx.globalAlpha = layer.opacity
    mainCtx.drawImage(tempCanvas, 0, 0)
  }

  // Reset
  mainCtx.globalCompositeOperation = 'source-over'
  mainCtx.globalAlpha = 1

  return mainCanvas.toDataURL('image/png')
}
