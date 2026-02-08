import type { Layer } from '../types/layers'
import LayerItem from './LayerItem'

interface LayerStackProps {
  layers: Layer[]
  onAddLayer: () => void
  onRemoveLayer: (id: string) => void
  onUpdateLayer: (id: string, updates: Partial<Layer>) => void
  onMoveLayer: (id: string, direction: 'up' | 'down') => void
  onDuplicateLayer: (id: string) => void
  onApply: () => void
  applying: boolean
}

export default function LayerStack({
  layers,
  onAddLayer,
  onRemoveLayer,
  onUpdateLayer,
  onMoveLayer,
  onDuplicateLayer,
  onApply,
  applying,
}: LayerStackProps) {
  return (
    <div className="layer-stack">
      <section className="section">
        <h2>Layer Stack</h2>
        <div className="layer-stack-toolbar">
          <button className="generate-btn" onClick={onAddLayer}>
            + Add Layer
          </button>
          <button
            className={`generate-btn ${applying ? 'processing' : ''}`}
            onClick={onApply}
            disabled={layers.length === 0 || applying}
          >
            {applying ? 'Applying...' : 'Apply to PBR'}
          </button>
        </div>
      </section>

      <div className="layer-list">
        {layers.map((layer, index) => (
          <LayerItem
            key={layer.id}
            layer={layer}
            onUpdate={onUpdateLayer}
            onRemove={onRemoveLayer}
            onMove={onMoveLayer}
            onDuplicate={onDuplicateLayer}
            isFirst={index === 0}
            isLast={index === layers.length - 1}
          />
        ))}
        {layers.length === 0 && (
          <p className="hint" style={{ textAlign: 'center', padding: '1rem' }}>
            No layers yet. Click "+ Add Layer" to start.
          </p>
        )}
      </div>
    </div>
  )
}
