import { useRef, useState, useEffect, useCallback } from 'react'

const CANVAS_SIZE = 512
const STAMP_SIZE = 128

type PbrChannel = 'basecolor' | 'normal' | 'roughness' | 'metalness' | 'height' | 'ao' | 'emissive' | 'translucency' | 'subsurface'
type StampType = 'round' | 'square' | 'splatter' | 'noise' | 'scratches' | 'custom'

const ALL_CHANNELS: { value: PbrChannel; label: string }[] = [
  { value: 'basecolor', label: 'Color' },
  { value: 'normal', label: 'Normal' },
  { value: 'roughness', label: 'Rough' },
  { value: 'metalness', label: 'Metal' },
  { value: 'height', label: 'Height' },
  { value: 'ao', label: 'AO' },
  { value: 'emissive', label: 'Emiss' },
  { value: 'translucency', label: 'Trans' },
  { value: 'subsurface', label: 'SSS' },
]

interface MapPainterProps {
  channels: PbrChannel[]
  sourceUrls: Record<string, string | null>
  onUpdate: (channel: PbrChannel, dataUrl: string) => void
  onUpload: (channel: PbrChannel, dataUrl: string) => void
  onChannelsChange: (channels: PbrChannel[]) => void
}

// --- Stamp generators ---

function generateRoundStamp(hardness: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  const cx = STAMP_SIZE / 2, cy = STAMP_SIZE / 2, r = STAMP_SIZE / 2
  if (hardness >= 0.99) {
    ctx.fillStyle = 'white'
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill()
  } else {
    const innerR = r * hardness
    const grad = ctx.createRadialGradient(cx, cy, innerR, cx, cy, r)
    grad.addColorStop(0, 'rgba(255,255,255,1)')
    grad.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = grad
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill()
  }
  return c
}

function generateSquareStamp(hardness: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  if (hardness >= 0.99) {
    ctx.fillStyle = 'white'; ctx.fillRect(0, 0, STAMP_SIZE, STAMP_SIZE)
  } else {
    const edge = STAMP_SIZE * (1 - hardness) * 0.5
    for (let y = 0; y < STAMP_SIZE; y++) {
      for (let x = 0; x < STAMP_SIZE; x++) {
        const dx = Math.max(0, Math.max(edge - x, x - (STAMP_SIZE - edge))) / edge
        const dy = Math.max(0, Math.max(edge - y, y - (STAMP_SIZE - edge))) / edge
        const a = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy))
        ctx.fillStyle = `rgba(255,255,255,${a})`
        ctx.fillRect(x, y, 1, 1)
      }
    }
  }
  return c
}

function generateSplatterStamp(): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  const cx = STAMP_SIZE / 2, cy = STAMP_SIZE / 2
  ctx.fillStyle = 'white'
  ctx.beginPath(); ctx.arc(cx, cy, STAMP_SIZE * 0.25, 0, Math.PI * 2); ctx.fill()
  let seed = 42
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
  for (let i = 0; i < 40; i++) {
    const angle = rand() * Math.PI * 2, dist = rand() * STAMP_SIZE * 0.45
    const r = rand() * STAMP_SIZE * 0.08 + 2
    ctx.fillStyle = `rgba(255,255,255,${Math.max(0, 1 - dist / (STAMP_SIZE * 0.5))})`
    ctx.beginPath(); ctx.arc(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist, r, 0, Math.PI * 2); ctx.fill()
  }
  return c
}

function generateNoiseStamp(): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  const imgData = ctx.createImageData(STAMP_SIZE, STAMP_SIZE)
  const cx = STAMP_SIZE / 2, cy = STAMP_SIZE / 2, maxR = STAMP_SIZE / 2
  let seed = 123
  for (let i = 0; i < STAMP_SIZE * STAMP_SIZE; i++) {
    seed = (seed * 16807) % 2147483647
    const v = (seed / 2147483647)
    const x = i % STAMP_SIZE, y = Math.floor(i / STAMP_SIZE)
    const falloff = Math.max(0, 1 - Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / maxR)
    imgData.data[i * 4] = 255; imgData.data[i * 4 + 1] = 255; imgData.data[i * 4 + 2] = 255
    imgData.data[i * 4 + 3] = Math.round(v * falloff * 255)
  }
  ctx.putImageData(imgData, 0, 0)
  return c
}

function generateScratchesStamp(): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  let seed = 77
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
  ctx.strokeStyle = 'white'; ctx.lineCap = 'round'
  for (let i = 0; i < 12; i++) {
    ctx.lineWidth = rand() * 3 + 0.5; ctx.globalAlpha = rand() * 0.6 + 0.4
    const x1 = rand() * STAMP_SIZE, y1 = rand() * STAMP_SIZE
    const angle = rand() * Math.PI, len = rand() * STAMP_SIZE * 0.6 + 10
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 + Math.cos(angle) * len, y1 + Math.sin(angle) * len); ctx.stroke()
  }
  ctx.globalAlpha = 1
  return c
}

function applyHardnessFalloff(stamp: HTMLCanvasElement, hardness: number): HTMLCanvasElement {
  if (hardness >= 0.99) return stamp
  const c = document.createElement('canvas')
  c.width = STAMP_SIZE; c.height = STAMP_SIZE
  const ctx = c.getContext('2d')!
  ctx.drawImage(stamp, 0, 0)
  const imgData = ctx.getImageData(0, 0, STAMP_SIZE, STAMP_SIZE)
  const cx = STAMP_SIZE / 2, cy = STAMP_SIZE / 2, maxR = STAMP_SIZE / 2, innerR = maxR * hardness
  for (let i = 0; i < STAMP_SIZE * STAMP_SIZE; i++) {
    const x = i % STAMP_SIZE, y = Math.floor(i / STAMP_SIZE)
    const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if (dist > maxR) imgData.data[i * 4 + 3] = 0
    else if (dist > innerR) imgData.data[i * 4 + 3] = Math.round(imgData.data[i * 4 + 3] * (1 - (dist - innerR) / (maxR - innerR)))
  }
  ctx.putImageData(imgData, 0, 0)
  return c
}

// --- Component ---

export default function MapPainter({ channels, sourceUrls, onUpdate, onUpload, onChannelsChange }: MapPainterProps) {
  // One canvas per active channel
  const canvasMapRef = useRef<Map<string, HTMLCanvasElement>>(new Map())
  const strokeMapRef = useRef<Map<string, HTMLCanvasElement>>(new Map())
  const snapshotMapRef = useRef<Map<string, ImageData>>(new Map())
  const stampRef = useRef<HTMLCanvasElement | null>(null)
  const painting = useRef(false)
  const lastPos = useRef<{ x: number; y: number } | null>(null)

  // Which channel to show in the preview canvas
  const [viewChannel, setViewChannel] = useState<PbrChannel>(channels[0] || 'basecolor')
  const displayCanvasRef = useRef<HTMLCanvasElement>(null)

  const [brushSize, setBrushSize] = useState(30)
  const [brushValue, setBrushValue] = useState(255)
  const [brushColor, setBrushColor] = useState('#ffffff')
  const [brushOpacity, setBrushOpacity] = useState(0.8)
  const [brushHardness, setBrushHardness] = useState(0.8)
  const [stampType, setStampType] = useState<StampType>('round')
  const [erasing, setErasing] = useState(false)
  const [customStampUrl, setCustomStampUrl] = useState<string | null>(null)

  // Ensure view channel is in selected channels
  useEffect(() => {
    if (!channels.includes(viewChannel)) setViewChannel(channels[0] || 'basecolor')
  }, [channels, viewChannel])

  // Init/update canvases per channel
  useEffect(() => {
    for (const ch of channels) {
      if (!canvasMapRef.current.has(ch)) {
        const c = document.createElement('canvas')
        c.width = CANVAS_SIZE; c.height = CANVAS_SIZE
        canvasMapRef.current.set(ch, c)
      }
      if (!strokeMapRef.current.has(ch)) {
        const s = document.createElement('canvas')
        s.width = CANVAS_SIZE; s.height = CANVAS_SIZE
        strokeMapRef.current.set(ch, s)
      }
    }
  }, [channels])

  // Load source images onto channel canvases
  useEffect(() => {
    for (const ch of channels) {
      const canvas = canvasMapRef.current.get(ch)
      if (!canvas) continue
      const ctx = canvas.getContext('2d')!
      const url = sourceUrls[ch]
      if (!url) {
        ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      } else {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => { ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE); syncDisplay() }
        img.onerror = () => { ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE) }
        img.src = url
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceUrls, channels])

  // Build stamp
  useEffect(() => {
    let raw: HTMLCanvasElement
    switch (stampType) {
      case 'round': stampRef.current = generateRoundStamp(brushHardness); return
      case 'square': raw = generateSquareStamp(brushHardness); break
      case 'splatter': raw = generateSplatterStamp(); break
      case 'noise': raw = generateNoiseStamp(); break
      case 'scratches': raw = generateScratchesStamp(); break
      case 'custom':
        if (customStampUrl) {
          const img = new Image()
          img.onload = () => {
            const c = document.createElement('canvas')
            c.width = STAMP_SIZE; c.height = STAMP_SIZE
            c.getContext('2d')!.drawImage(img, 0, 0, STAMP_SIZE, STAMP_SIZE)
            stampRef.current = applyHardnessFalloff(c, brushHardness)
          }
          img.src = customStampUrl
        }
        return
      default: return
    }
    stampRef.current = applyHardnessFalloff(raw, brushHardness)
  }, [stampType, brushHardness, customStampUrl])

  // Sync the display canvas to show the currently viewed channel
  const syncDisplay = useCallback(() => {
    const display = displayCanvasRef.current
    const source = canvasMapRef.current.get(viewChannel)
    if (!display || !source) return
    display.getContext('2d')!.drawImage(source, 0, 0)
  }, [viewChannel])

  useEffect(() => { syncDisplay() }, [syncDisplay])

  const getCanvasPos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = displayCanvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (CANVAS_SIZE / rect.width),
      y: (e.clientY - rect.top) * (CANVAS_SIZE / rect.height),
    }
  }, [])

  // Dab onto a single channel's stroke buffer
  const dabChannel = useCallback((ch: PbrChannel, x: number, y: number) => {
    const stroke = strokeMapRef.current.get(ch)
    const stamp = stampRef.current
    if (!stroke || !stamp) return
    const ctx = stroke.getContext('2d')!
    const half = brushSize / 2

    const tinted = document.createElement('canvas')
    tinted.width = brushSize; tinted.height = brushSize
    const tCtx = tinted.getContext('2d')!
    tCtx.drawImage(stamp, 0, 0, brushSize, brushSize)
    tCtx.globalCompositeOperation = 'source-in'

    if (ch === 'basecolor') {
      tCtx.fillStyle = brushColor
    } else {
      const v = brushValue
      tCtx.fillStyle = `rgb(${v},${v},${v})`
    }
    tCtx.fillRect(0, 0, brushSize, brushSize)
    ctx.drawImage(tinted, x - half, y - half)
  }, [brushSize, brushValue, brushColor])

  // Dab all active channels
  const dab = useCallback((x: number, y: number) => {
    for (const ch of channels) dabChannel(ch, x, y)
  }, [channels, dabChannel])

  const strokeTo = useCallback((from: { x: number; y: number }, to: { x: number; y: number }) => {
    const dx = to.x - from.x, dy = to.y - from.y
    const dist = Math.sqrt(dx * dx + dy * dy)
    const spacing = Math.max(1, brushSize * 0.15)
    const steps = Math.max(1, Math.ceil(dist / spacing))
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      dab(from.x + dx * t, from.y + dy * t)
    }
  }, [dab, brushSize])

  // Composite all stroke buffers onto their canvases
  const compositeAll = useCallback(() => {
    for (const ch of channels) {
      const canvas = canvasMapRef.current.get(ch)
      const stroke = strokeMapRef.current.get(ch)
      const snapshot = snapshotMapRef.current.get(ch)
      if (!canvas || !stroke) continue
      const ctx = canvas.getContext('2d')!
      if (snapshot) ctx.putImageData(snapshot, 0, 0)
      ctx.save()
      ctx.globalAlpha = brushOpacity
      ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over'
      ctx.drawImage(stroke, 0, 0)
      ctx.restore()
    }
    syncDisplay()
  }, [channels, brushOpacity, erasing, syncDisplay])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    painting.current = true
    const pos = getCanvasPos(e)
    lastPos.current = pos

    // Snapshot all channel canvases
    for (const ch of channels) {
      const canvas = canvasMapRef.current.get(ch)
      if (canvas) snapshotMapRef.current.set(ch, canvas.getContext('2d')!.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE))
      const stroke = strokeMapRef.current.get(ch)
      if (stroke) stroke.getContext('2d')!.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    }

    dab(pos.x, pos.y)
    compositeAll()
  }, [getCanvasPos, channels, dab, compositeAll])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!painting.current) return
    const pos = getCanvasPos(e)
    if (lastPos.current) strokeTo(lastPos.current, pos)
    lastPos.current = pos
    compositeAll()
  }, [getCanvasPos, strokeTo, compositeAll])

  const handleMouseUp = useCallback(() => {
    if (!painting.current) return
    painting.current = false
    lastPos.current = null
    snapshotMapRef.current.clear()

    // Emit all updated channels
    for (const ch of channels) {
      const canvas = canvasMapRef.current.get(ch)
      if (canvas) onUpdate(ch, canvas.toDataURL('image/png'))
    }
  }, [channels, onUpdate])

  const handleFill = useCallback(() => {
    for (const ch of channels) {
      const canvas = canvasMapRef.current.get(ch)
      if (!canvas) continue
      const ctx = canvas.getContext('2d')!
      ctx.globalAlpha = brushOpacity
      if (ch === 'basecolor') {
        ctx.fillStyle = brushColor
      } else {
        ctx.fillStyle = `rgb(${brushValue},${brushValue},${brushValue})`
      }
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      ctx.globalAlpha = 1
      onUpdate(ch, canvas.toDataURL('image/png'))
    }
    syncDisplay()
  }, [channels, brushColor, brushValue, brushOpacity, onUpdate, syncDisplay])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const img = new Image()
      img.onload = () => {
        // Upload replaces the currently viewed channel
        const canvas = canvasMapRef.current.get(viewChannel)
        if (!canvas) return
        canvas.getContext('2d')!.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE)
        onUpload(viewChannel, canvas.toDataURL('image/png'))
        syncDisplay()
      }
      img.src = reader.result as string
    }
    reader.readAsDataURL(file)
  }, [viewChannel, onUpload, syncDisplay])

  const handleStampUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => { setCustomStampUrl(reader.result as string); setStampType('custom') }
    reader.readAsDataURL(file)
  }, [])

  const toggleChannel = (ch: PbrChannel) => {
    if (channels.includes(ch)) {
      if (channels.length > 1) onChannelsChange(channels.filter(c => c !== ch))
    } else {
      onChannelsChange([...channels, ch])
    }
  }

  const stampPreviewUrl = stampRef.current?.toDataURL('image/png')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Channel toggles */}
      <div>
        <label style={{ fontSize: '0.75rem', opacity: 0.7 }}>Paint channels:</label>
        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginTop: 4 }}>
          {ALL_CHANNELS.map(({ value, label }) => (
            <button key={value} className="preset-btn"
              onClick={() => toggleChannel(value)}
              style={{
                fontSize: '0.65rem', padding: '3px 6px',
                background: channels.includes(value) ? '#6366f1' : undefined,
                color: channels.includes(value) ? 'white' : undefined,
                opacity: channels.includes(value) ? 1 : 0.5,
              }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* View channel selector */}
      {channels.length > 1 && (
        <div>
          <label style={{ fontSize: '0.75rem', opacity: 0.7 }}>Viewing:</label>
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginTop: 2 }}>
            {channels.map(ch => (
              <button key={ch} className="preset-btn"
                onClick={() => { setViewChannel(ch); syncDisplay() }}
                style={{
                  fontSize: '0.65rem', padding: '2px 6px',
                  background: viewChannel === ch ? '#818cf8' : undefined,
                  color: viewChannel === ch ? 'white' : undefined,
                }}>
                {ALL_CHANNELS.find(c => c.value === ch)?.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Paint canvas (shows viewed channel) */}
      <canvas
        ref={displayCanvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          width: '100%', aspectRatio: '1', borderRadius: 4,
          cursor: 'crosshair', border: '1px solid rgba(255,255,255,0.1)',
        }}
      />

      {/* Stamp selection */}
      <div>
        <label style={{ fontSize: '0.75rem', opacity: 0.7 }}>Brush Stamp:</label>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
          {(['round', 'square', 'splatter', 'noise', 'scratches'] as StampType[]).map(st => (
            <button key={st} className="preset-btn"
              onClick={() => setStampType(st)}
              style={{
                fontSize: '0.7rem', padding: '4px 8px',
                background: stampType === st ? '#6366f1' : undefined,
                color: stampType === st ? 'white' : undefined,
              }}>
              {st.charAt(0).toUpperCase() + st.slice(1)}
            </button>
          ))}
          <button className="preset-btn"
            onClick={() => document.getElementById('stamp-upload')?.click()}
            style={{
              fontSize: '0.7rem', padding: '4px 8px',
              background: stampType === 'custom' ? '#6366f1' : undefined,
              color: stampType === 'custom' ? 'white' : undefined,
            }}>
            Custom
          </button>
          <input id="stamp-upload" type="file" accept="image/*" onChange={handleStampUpload} style={{ display: 'none' }} />
        </div>
        {stampPreviewUrl && (
          <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
            <img src={stampPreviewUrl} alt="stamp" style={{ width: 32, height: 32, borderRadius: 2, border: '1px solid #555', background: '#222' }} />
          </div>
        )}
      </div>

      <label>
        Size: {brushSize}px
        <input type="range" min="2" max="200" step="1" value={brushSize}
          onChange={(e) => setBrushSize(parseInt(e.target.value))} />
      </label>

      {/* Color picker for basecolor, value slider for grayscale channels */}
      {channels.includes('basecolor') && (
        <label>
          Color
          <input type="color" value={brushColor} onChange={(e) => setBrushColor(e.target.value)} />
        </label>
      )}

      {channels.some(ch => ch !== 'basecolor') && (
        <label>
          Value: {brushValue}
          <input type="range" min="0" max="255" step="1" value={brushValue}
            onChange={(e) => setBrushValue(parseInt(e.target.value))} />
          <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
            <div style={{ width: 20, height: 20, background: `rgb(${brushValue},${brushValue},${brushValue})`, border: '1px solid #555', borderRadius: 2 }} />
            <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>
              {channels.includes('metalness') ? (brushValue > 127 ? 'metallic' : 'dielectric') :
               channels.includes('roughness') ? (brushValue > 127 ? 'rough' : 'smooth') :
               channels.includes('height') ? (brushValue > 127 ? 'raised' : 'recessed') : ''}
            </span>
          </div>
        </label>
      )}

      <label>
        Opacity: {Math.round(brushOpacity * 100)}%
        <input type="range" min="0.01" max="1" step="0.01" value={brushOpacity}
          onChange={(e) => setBrushOpacity(parseFloat(e.target.value))} />
      </label>

      <label>
        Hardness: {Math.round(brushHardness * 100)}%
        <input type="range" min="0" max="1" step="0.01" value={brushHardness}
          onChange={(e) => setBrushHardness(parseFloat(e.target.value))} />
      </label>

      <div style={{ display: 'flex', gap: 6 }}>
        <label style={{ flex: 1 }}>
          <input type="checkbox" checked={erasing} onChange={(e) => setErasing(e.target.checked)} />
          Erase
        </label>
        <button className="preset-btn" onClick={handleFill} style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
          Fill
        </button>
      </div>

      <div>
        <label style={{ fontSize: '0.75rem', opacity: 0.6 }}>Replace {ALL_CHANNELS.find(c => c.value === viewChannel)?.label} map:</label>
        <input type="file" accept="image/*" onChange={handleFileUpload} style={{ fontSize: '0.7rem' }} />
      </div>
    </div>
  )
}
