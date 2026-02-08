// Seeded PRNG — mulberry32
function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Smooth interpolation
function smoothstep(t: number): number {
  return t * t * (3 - 2 * t)
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Generate value noise with bilinear interpolation, 4 octaves.
 * Returns Float32Array of width*height values in [0,1].
 */
export function generatePerlinNoise(
  width: number,
  height: number,
  scale: number,
  seed: number
): Float32Array {
  const rng = mulberry32(seed)
  const result = new Float32Array(width * height)

  // Pre-generate a large enough grid of random values for all octaves
  const maxGridSize = Math.ceil(Math.max(width, height) / scale) + 2
  const gridSize = maxGridSize * 16 // enough for 4 octaves
  const grid = new Float32Array(gridSize * gridSize)
  for (let i = 0; i < grid.length; i++) {
    grid[i] = rng()
  }

  function sampleGrid(gx: number, gy: number): number {
    const ix = ((gx % gridSize) + gridSize) % gridSize
    const iy = ((gy % gridSize) + gridSize) % gridSize
    return grid[iy * gridSize + ix]
  }

  function noise2d(x: number, y: number): number {
    const ix = Math.floor(x)
    const iy = Math.floor(y)
    const fx = x - ix
    const fy = y - iy
    const sx = smoothstep(fx)
    const sy = smoothstep(fy)

    const v00 = sampleGrid(ix, iy)
    const v10 = sampleGrid(ix + 1, iy)
    const v01 = sampleGrid(ix, iy + 1)
    const v11 = sampleGrid(ix + 1, iy + 1)

    return lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sy)
  }

  // 4 octaves of value noise
  const octaves = 4
  let maxAmp = 0
  for (let o = 0; o < octaves; o++) maxAmp += 1 / (1 << o)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = 0
      let amp = 1
      let freq = scale > 0 ? 1 / scale : 1
      for (let o = 0; o < octaves; o++) {
        value += noise2d(x * freq + o * 97, y * freq + o * 131) * amp
        freq *= 2
        amp *= 0.5
      }
      result[y * width + x] = value / maxAmp
    }
  }

  return result
}

/**
 * Generate Voronoi noise (distance-to-nearest-point field).
 * Returns Float32Array of width*height values in [0,1].
 */
export function generateVoronoiNoise(
  width: number,
  height: number,
  numPoints: number,
  seed: number
): Float32Array {
  const rng = mulberry32(seed)
  const result = new Float32Array(width * height)

  // Grid acceleration
  const cellSize = Math.max(1, Math.floor(Math.sqrt((width * height) / Math.max(1, numPoints))))
  const gridW = Math.ceil(width / cellSize)
  const gridH = Math.ceil(height / cellSize)
  const cells: number[][][] = Array.from({ length: gridH }, () =>
    Array.from({ length: gridW }, () => [])
  )

  // Place random points
  const points: [number, number][] = []
  for (let i = 0; i < numPoints; i++) {
    const px = rng() * width
    const py = rng() * height
    points.push([px, py])
    const cx = Math.min(Math.floor(px / cellSize), gridW - 1)
    const cy = Math.min(Math.floor(py / cellSize), gridH - 1)
    cells[cy][cx].push(i)
  }

  // Find max distance for normalization
  let maxDist = 0

  for (let y = 0; y < height; y++) {
    const cy = Math.min(Math.floor(y / cellSize), gridH - 1)
    for (let x = 0; x < width; x++) {
      const cx = Math.min(Math.floor(x / cellSize), gridW - 1)

      let minDist = Infinity
      // Check surrounding cells
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const ny = cy + dy
          const nx = cx + dx
          if (ny < 0 || ny >= gridH || nx < 0 || nx >= gridW) continue
          for (const pi of cells[ny][nx]) {
            const [px, py] = points[pi]
            const ddx = x - px
            const ddy = y - py
            const d = Math.sqrt(ddx * ddx + ddy * ddy)
            if (d < minDist) minDist = d
          }
        }
      }

      if (minDist === Infinity) minDist = 0
      result[y * width + x] = minDist
      if (minDist > maxDist) maxDist = minDist
    }
  }

  // Normalize to [0,1]
  if (maxDist > 0) {
    for (let i = 0; i < result.length; i++) {
      result[i] /= maxDist
    }
  }

  return result
}
