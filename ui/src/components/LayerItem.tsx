import { useState } from 'react'
import { BLEND_MODES, PBR_CHANNELS } from '../types/layers'
import type { Layer, LayerType, NoiseType, PbrChannel } from '../types/layers'

interface LayerItemProps {
  layer: Layer
  onUpdate: (id: string, updates: Partial<Layer>) => void
  onRemove: (id: string) => void
  onMove: (id: string, direction: 'up' | 'down') => void
  onDuplicate: (id: string) => void
  onGenerate: (id: string, prompt: string) => void
  onPaint: (id: string) => void
  isPainting: boolean
  isFirst: boolean
  isLast: boolean
}

const CHANNEL_LABELS: Record<PbrChannel, string> = {
  basecolor: 'Color',
  normal: 'Normal',
  roughness: 'Rough',
  metalness: 'Metal',
  height: 'Height',
  translucency: 'Trans',
  subsurface: 'SSS',
}

export default function LayerItem({
  layer,
  onUpdate,
  onRemove,
  onMove,
  onDuplicate,
  onGenerate,
  onPaint,
  isPainting,
  isFirst,
  isLast,
}: LayerItemProps) {
  const [expanded, setExpanded] = useState(true)

  const toggleChannel = (ch: PbrChannel) => {
    onUpdate(layer.id, {
      channels: {
        ...layer.channels,
        [ch]: { enabled: !layer.channels[ch].enabled },
      },
    })
  }

  const hasMaps = Object.values(layer.materialMaps).some(Boolean)

  return (
    <div className={`layer-item ${expanded ? 'expanded' : ''}`}>
      <div className="layer-header" onClick={() => setExpanded(!expanded)}>
        <button
          className="layer-eye"
          onClick={(e) => { e.stopPropagation(); onUpdate(layer.id, { visible: !layer.visible }) }}
          title={layer.visible ? 'Hide layer' : 'Show layer'}
        >
          {layer.visible ? '\u{1F441}' : '\u25CB'}
        </button>
        <input
          className="layer-name-input"
          value={layer.name}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => onUpdate(layer.id, { name: e.target.value })}
          style={{ background: 'transparent', border: 'none', color: 'inherit', fontSize: 'inherit', width: '100%' }}
        />
        <div className="layer-actions">
          <button onClick={(e) => { e.stopPropagation(); onMove(layer.id, 'up') }} disabled={isLast} title="Move up">
            {'\u25B2'}
          </button>
          <button onClick={(e) => { e.stopPropagation(); onMove(layer.id, 'down') }} disabled={isFirst} title="Move down">
            {'\u25BC'}
          </button>
          <button onClick={(e) => { e.stopPropagation(); onDuplicate(layer.id) }} title="Duplicate">
            {'\u2398'}
          </button>
          <button onClick={(e) => { e.stopPropagation(); onRemove(layer.id) }} title="Delete">
            {'\u2715'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="layer-controls">
          {/* Blend & Opacity */}
          <div className="control-row">
            <label>
              Blend
              <select
                value={layer.blendMode}
                onChange={(e) => onUpdate(layer.id, { blendMode: e.target.value as any })}
              >
                {BLEND_MODES.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </label>
            <label>
              Opacity: {Math.round(layer.opacity * 100)}%
              <input type="range" min="0" max="1" step="0.01" value={layer.opacity}
                onChange={(e) => onUpdate(layer.id, { opacity: parseFloat(e.target.value) })} />
            </label>
          </div>

          {/* Channel toggles */}
          <div className="control-row" style={{ flexWrap: 'wrap', gap: 4 }}>
            <span style={{ fontSize: '0.75rem', opacity: 0.7, width: '100%' }}>Channels:</span>
            {PBR_CHANNELS.map(ch => (
              <label key={ch} className="channel-flip" style={{ fontSize: '0.75rem' }}>
                <input
                  type="checkbox"
                  checked={layer.channels[ch].enabled}
                  onChange={() => toggleChannel(ch)}
                />
                {CHANNEL_LABELS[ch]}
              </label>
            ))}
          </div>

          {/* Layer type */}
          <div className="control-row">
            <label>
              Type
              <select
                value={layer.type}
                onChange={(e) => onUpdate(layer.id, { type: e.target.value as LayerType })}
              >
                <option value="material">Material (PATINA)</option>
                <option value="solid">Solid</option>
                <option value="noise">Noise</option>
              </select>
            </label>
          </div>

          {/* MATERIAL type */}
          {layer.type === 'material' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <input
                type="text"
                placeholder="Material prompt... e.g., 'rusted iron'"
                value={layer.materialPrompt}
                onChange={(e) => onUpdate(layer.id, { materialPrompt: e.target.value })}
                style={{ width: '100%', padding: '6px 8px', fontSize: '0.85rem' }}
              />
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  className={`generate-btn ${layer.generating ? 'processing' : ''}`}
                  onClick={() => onGenerate(layer.id, layer.materialPrompt)}
                  disabled={layer.generating || !layer.materialPrompt.trim()}
                  style={{ fontSize: '0.8rem', padding: '6px 12px', flex: 1 }}
                >
                  {layer.generating ? 'Generating...' : 'Generate'}
                </button>
                <button
                  className="generate-btn"
                  onClick={() => onPaint(layer.id)}
                  style={{
                    fontSize: '0.8rem', padding: '6px 12px',
                    background: isPainting ? '#6366f1' : undefined,
                    color: isPainting ? 'white' : undefined,
                  }}
                >
                  {isPainting ? 'Painting...' : 'Paint'}
                </button>
              </div>
              {hasMaps && (
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {PBR_CHANNELS.map(ch => {
                    const url = layer.materialMaps[ch]
                    if (!url) return null
                    return (
                      <div key={ch} style={{
                        width: 48, height: 48, borderRadius: 4, overflow: 'hidden',
                        border: layer.channels[ch].enabled ? '2px solid #6366f1' : '2px solid transparent',
                        opacity: layer.channels[ch].enabled ? 1 : 0.4,
                        cursor: 'pointer',
                      }}
                        onClick={() => toggleChannel(ch)}
                        title={`${CHANNEL_LABELS[ch]} — click to toggle`}
                      >
                        <img src={url} alt={ch} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* SOLID type */}
          {layer.type === 'solid' && (
            <div className="color-row">
              <label>
                Color
                <input type="color" value={layer.color}
                  onChange={(e) => onUpdate(layer.id, { color: e.target.value })} />
              </label>
            </div>
          )}

          {/* NOISE type */}
          {layer.type === 'noise' && (
            <>
              <div className="control-row">
                <label>
                  Noise
                  <select value={layer.noiseType}
                    onChange={(e) => onUpdate(layer.id, { noiseType: e.target.value as NoiseType })}>
                    <option value="perlin">Perlin</option>
                    <option value="voronoi">Voronoi</option>
                  </select>
                </label>
                <label>
                  Scale: {layer.noiseScale.toFixed(1)}
                  <input type="range" min="0.5" max="20" step="0.5" value={layer.noiseScale}
                    onChange={(e) => onUpdate(layer.id, { noiseScale: parseFloat(e.target.value) })} />
                </label>
              </div>
              <div className="control-row">
                <label>
                  Seed: {layer.noiseSeed}
                  <input type="range" min="0" max="9999" step="1" value={layer.noiseSeed}
                    onChange={(e) => onUpdate(layer.id, { noiseSeed: parseInt(e.target.value) })} />
                </label>
              </div>
              <div className="color-row">
                <label>Color 1 <input type="color" value={layer.noiseColor1}
                  onChange={(e) => onUpdate(layer.id, { noiseColor1: e.target.value })} /></label>
                <label>Color 2 <input type="color" value={layer.noiseColor2}
                  onChange={(e) => onUpdate(layer.id, { noiseColor2: e.target.value })} /></label>
              </div>
            </>
          )}

          {/* Mask section */}
          <div className="mask-section">
            <label className="mask-toggle">
              <input type="checkbox" checked={layer.mask.enabled}
                onChange={(e) => onUpdate(layer.id, { mask: { ...layer.mask, enabled: e.target.checked } })} />
              Mask
            </label>
            {layer.mask.enabled && (
              <>
                <div className="control-row">
                  <label>
                    Mask Noise
                    <select value={layer.mask.noiseType}
                      onChange={(e) => onUpdate(layer.id, { mask: { ...layer.mask, noiseType: e.target.value as NoiseType } })}>
                      <option value="perlin">Perlin</option>
                      <option value="voronoi">Voronoi</option>
                    </select>
                  </label>
                  <label>
                    Scale: {layer.mask.scale.toFixed(1)}
                    <input type="range" min="0.5" max="20" step="0.5" value={layer.mask.scale}
                      onChange={(e) => onUpdate(layer.id, { mask: { ...layer.mask, scale: parseFloat(e.target.value) } })} />
                  </label>
                </div>
                <div className="control-row">
                  <label>
                    Seed: {layer.mask.seed}
                    <input type="range" min="0" max="9999" step="1" value={layer.mask.seed}
                      onChange={(e) => onUpdate(layer.id, { mask: { ...layer.mask, seed: parseInt(e.target.value) } })} />
                  </label>
                  <label className="mask-invert">
                    <input type="checkbox" checked={layer.mask.invert}
                      onChange={(e) => onUpdate(layer.id, { mask: { ...layer.mask, invert: e.target.checked } })} />
                    Invert
                  </label>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
