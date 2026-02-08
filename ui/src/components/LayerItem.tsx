import { useState } from 'react'
import { BLEND_MODES } from '../types/layers'
import type { Layer, LayerType, NoiseType } from '../types/layers'

interface LayerItemProps {
  layer: Layer
  onUpdate: (id: string, updates: Partial<Layer>) => void
  onRemove: (id: string) => void
  onMove: (id: string, direction: 'up' | 'down') => void
  onDuplicate: (id: string) => void
  isFirst: boolean
  isLast: boolean
}

export default function LayerItem({
  layer,
  onUpdate,
  onRemove,
  onMove,
  onDuplicate,
  isFirst,
  isLast,
}: LayerItemProps) {
  const [expanded, setExpanded] = useState(true)

  return (
    <div className={`layer-item ${expanded ? 'expanded' : ''}`}>
      <div className="layer-header" onClick={() => setExpanded(!expanded)}>
        <button
          className="layer-eye"
          onClick={(e) => {
            e.stopPropagation()
            onUpdate(layer.id, { visible: !layer.visible })
          }}
          title={layer.visible ? 'Hide layer' : 'Show layer'}
        >
          {layer.visible ? '\u{1F441}' : '\u25CB'}
        </button>
        <span className="layer-name">{layer.name}</span>
        <div className="layer-actions">
          <button
            onClick={(e) => { e.stopPropagation(); onMove(layer.id, 'up') }}
            disabled={isLast}
            title="Move up"
          >
            {'\u25B2'}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onMove(layer.id, 'down') }}
            disabled={isFirst}
            title="Move down"
          >
            {'\u25BC'}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDuplicate(layer.id) }}
            title="Duplicate"
          >
            {'\u2398'}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(layer.id) }}
            title="Delete"
          >
            {'\u2715'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="layer-controls">
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
              <input
                type="range"
                min="0" max="1" step="0.01"
                value={layer.opacity}
                onChange={(e) => onUpdate(layer.id, { opacity: parseFloat(e.target.value) })}
              />
            </label>
          </div>

          <div className="control-row">
            <label>
              Type
              <select
                value={layer.type}
                onChange={(e) => onUpdate(layer.id, { type: e.target.value as LayerType })}
              >
                <option value="solid">Solid</option>
                <option value="noise">Noise</option>
              </select>
            </label>
          </div>

          {layer.type === 'solid' ? (
            <div className="color-row">
              <label>
                Color
                <input
                  type="color"
                  value={layer.color}
                  onChange={(e) => onUpdate(layer.id, { color: e.target.value })}
                />
              </label>
            </div>
          ) : (
            <>
              <div className="control-row">
                <label>
                  Noise
                  <select
                    value={layer.noiseType}
                    onChange={(e) => onUpdate(layer.id, { noiseType: e.target.value as NoiseType })}
                  >
                    <option value="perlin">Perlin</option>
                    <option value="voronoi">Voronoi</option>
                  </select>
                </label>
                <label>
                  Scale: {layer.noiseScale.toFixed(1)}
                  <input
                    type="range"
                    min="0.5" max="20" step="0.5"
                    value={layer.noiseScale}
                    onChange={(e) => onUpdate(layer.id, { noiseScale: parseFloat(e.target.value) })}
                  />
                </label>
              </div>
              <div className="control-row">
                <label>
                  Seed: {layer.noiseSeed}
                  <input
                    type="range"
                    min="0" max="9999" step="1"
                    value={layer.noiseSeed}
                    onChange={(e) => onUpdate(layer.id, { noiseSeed: parseInt(e.target.value) })}
                  />
                </label>
              </div>
              <div className="color-row">
                <label>
                  Color 1
                  <input
                    type="color"
                    value={layer.noiseColor1}
                    onChange={(e) => onUpdate(layer.id, { noiseColor1: e.target.value })}
                  />
                </label>
                <label>
                  Color 2
                  <input
                    type="color"
                    value={layer.noiseColor2}
                    onChange={(e) => onUpdate(layer.id, { noiseColor2: e.target.value })}
                  />
                </label>
              </div>
            </>
          )}

          {/* Mask section */}
          <div className="mask-section">
            <label className="mask-toggle">
              <input
                type="checkbox"
                checked={layer.mask.enabled}
                onChange={(e) =>
                  onUpdate(layer.id, { mask: { ...layer.mask, enabled: e.target.checked } })
                }
              />
              Mask
            </label>
            {layer.mask.enabled && (
              <>
                <div className="control-row">
                  <label>
                    Mask Noise
                    <select
                      value={layer.mask.noiseType}
                      onChange={(e) =>
                        onUpdate(layer.id, {
                          mask: { ...layer.mask, noiseType: e.target.value as NoiseType },
                        })
                      }
                    >
                      <option value="perlin">Perlin</option>
                      <option value="voronoi">Voronoi</option>
                    </select>
                  </label>
                  <label>
                    Scale: {layer.mask.scale.toFixed(1)}
                    <input
                      type="range"
                      min="0.5" max="20" step="0.5"
                      value={layer.mask.scale}
                      onChange={(e) =>
                        onUpdate(layer.id, {
                          mask: { ...layer.mask, scale: parseFloat(e.target.value) },
                        })
                      }
                    />
                  </label>
                </div>
                <div className="control-row">
                  <label>
                    Seed: {layer.mask.seed}
                    <input
                      type="range"
                      min="0" max="9999" step="1"
                      value={layer.mask.seed}
                      onChange={(e) =>
                        onUpdate(layer.id, {
                          mask: { ...layer.mask, seed: parseInt(e.target.value) },
                        })
                      }
                    />
                  </label>
                  <label className="mask-invert">
                    <input
                      type="checkbox"
                      checked={layer.mask.invert}
                      onChange={(e) =>
                        onUpdate(layer.id, {
                          mask: { ...layer.mask, invert: e.target.checked },
                        })
                      }
                    />
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
