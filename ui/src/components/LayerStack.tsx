import type { Layer } from '../types/layers'
import LayerItem from './LayerItem'

interface LayerStackProps {
  layers: Layer[]
  onAddLayer: () => void
  onRemoveLayer: (id: string) => void
  onUpdateLayer: (id: string, updates: Partial<Layer>) => void
  onMoveLayer: (id: string, direction: 'up' | 'down') => void
  onDuplicateLayer: (id: string) => void
  onGenerateLayer: (id: string, prompt: string) => void
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
  onGenerateLayer,
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
            {applying ? 'Compositing...' : 'Composite Layers'}
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
            onGenerate={onGenerateLayer}
            isFirst={index === 0}
            isLast={index === layers.length - 1}
          />
        ))}
        {layers.length === 0 && (
          <p className="hint" style={{ textAlign: 'center', padding: '1rem' }}>
            No layers yet. Add a layer, enter a material prompt, and generate.
            <br />Each layer gets its own PBR maps that blend per-channel.
          </p>
        )}
      </div>
    </div>
  )
}
