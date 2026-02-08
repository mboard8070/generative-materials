import { useState, useEffect, useRef, Suspense } from 'react'
import MaterialPreview from './components/MaterialPreview'
import LayerStack from './components/LayerStack'
import { materialPresets, promptModifiers } from './data/materialPresets'
import { useLayerStack } from './hooks/useLayerStack'
import { compositeLayerStack } from './lib/compositor'
import './App.css'

const API_URL = 'http://localhost:8001'

type Mode = 'generate' | 'edit' | 'layers'

function App() {
  // Mode state
  const [mode, setMode] = useState<Mode>('generate')

  // Generation state
  const [prompt, setPrompt] = useState('')
  const [generating, setGenerating] = useState(false)
  const [textureUrl, setTextureUrl] = useState<string | null>(null)
  const [normalMapUrl, setNormalMapUrl] = useState<string | null>(null)
  const [heightMapUrl, setHeightMapUrl] = useState<string | null>(null)
  const [roughnessMapUrl, setRoughnessMapUrl] = useState<string | null>(null)
  const [emissiveMapUrl, setEmissiveMapUrl] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState('')

  // Edit state
  const [editPrompt, setEditPrompt] = useState('')
  const [editStrength, setEditStrength] = useState(0.7)

  // Material properties
  const [roughness, setRoughness] = useState(1.0)
  const [metalness, setMetalness] = useState(0.0)
  const [normalScale, setNormalScale] = useState(1.0)
  const [displacementScale, setDisplacementScale] = useState(0.1)
  const [emissiveIntensity, setEmissiveIntensity] = useState(0.0)

  // Viewport settings
  const [autoRotate, setAutoRotate] = useState(true)
  const [environment, setEnvironment] = useState('studio')

  // Tiling settings
  const [tilingMode, setTilingMode] = useState('multiscale')

  // Modifiers
  const [weathering, setWeathering] = useState(1)

  // Layer stack
  // Normal map channel flips
  const [normalFlipR, setNormalFlipR] = useState(false)
  const [normalFlipG, setNormalFlipG] = useState(false)
  const [normalFlipB, setNormalFlipB] = useState(false)

  const { layers, addLayer, removeLayer, updateLayer, moveLayer, duplicateLayer, clearLayers } = useLayerStack()
  const [compositedTextureUrl, setCompositedTextureUrl] = useState<string | null>(null)
  const [applyingLayers, setApplyingLayers] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Composite layers on change (debounced)
  // Draws textureUrl as the base, then layers on top
  useEffect(() => {
    if (mode !== 'layers') return
    let cancelled = false
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      compositeLayerStack(layers, textureUrl)
        .then(result => {
          if (!cancelled) setCompositedTextureUrl(result)
        })
        .catch(err => {
          console.warn('Layer compositing failed:', err)
        })
    }, 33)
    return () => {
      cancelled = true
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [layers, mode, textureUrl])

  // Determine active texture for preview
  // In layers mode: show composite (which includes the base texture + layers)
  // Otherwise: show the generated/edited texture
  const activeTextureUrl = mode === 'layers'
    ? (compositedTextureUrl ?? textureUrl)
    : textureUrl

  const handlePresetSelect = (preset: typeof materialPresets[0]['materials'][0]) => {
    setPrompt(preset.prompt)
    setRoughness(preset.roughness)
    setMetalness(preset.metalness)
  }

  const buildFullPrompt = () => {
    let fullPrompt = `seamless tileable PBR texture of ${prompt}`
    if (weathering !== 1) {
      fullPrompt += `, ${promptModifiers.weathering[weathering]}`
    }
    fullPrompt += ', game texture, photorealistic material, top-down view, even lighting'
    return fullPrompt
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setGenerating(true)
    try {
      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: buildFullPrompt(),
          tiling_mode: tilingMode
        })
      })

      if (response.ok) {
        const data = await response.json()
        setTextureUrl(`${API_URL}${data.texture_url}`)
        setNormalMapUrl(data.normal_map_url ? `${API_URL}${data.normal_map_url}` : null)
        setHeightMapUrl(data.height_map_url ? `${API_URL}${data.height_map_url}` : null)
        setRoughnessMapUrl(data.roughness_map_url ? `${API_URL}${data.roughness_map_url}` : null)
        setEmissiveMapUrl(data.emissive_map_url ? `${API_URL}${data.emissive_map_url}` : null)
      }
    } catch (error) {
      console.error('Generation failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  const handleEdit = async () => {
    if (!textureUrl || !editPrompt.trim()) return

    setGenerating(true)
    try {
      const response = await fetch(`${API_URL}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: textureUrl,
          prompt: editPrompt,
          strength: editStrength,
          tiling_mode: tilingMode
        })
      })

      if (response.ok) {
        const data = await response.json()
        setTextureUrl(`${API_URL}${data.texture_url}`)
        setNormalMapUrl(data.normal_map_url ? `${API_URL}${data.normal_map_url}` : null)
        setHeightMapUrl(data.height_map_url ? `${API_URL}${data.height_map_url}` : null)
        setRoughnessMapUrl(data.roughness_map_url ? `${API_URL}${data.roughness_map_url}` : null)
        setEmissiveMapUrl(data.emissive_map_url ? `${API_URL}${data.emissive_map_url}` : null)
      }
    } catch (error) {
      console.error('Edit failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  const handleExport = async () => {
    if (!textureUrl) return

    try {
      const response = await fetch(`${API_URL}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texture_url: textureUrl,
          roughness,
          metalness,
          format: 'unreal'
        })
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'material_export.zip'
        a.click()
      }
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const handleApplyLayers = async () => {
    if (!compositedTextureUrl) return

    setApplyingLayers(true)
    try {
      const response = await fetch(`${API_URL}/composite-pbr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: compositedTextureUrl })
      })

      if (response.ok) {
        const data = await response.json()
        // Flatten: clear layers first so the compositor won't double-apply,
        // then set the composite as the new base texture with fresh PBR maps.
        clearLayers()
        setCompositedTextureUrl(null)
        setTextureUrl(`${API_URL}${data.texture_url}`)
        setNormalMapUrl(data.normal_map_url ? `${API_URL}${data.normal_map_url}` : null)
        setHeightMapUrl(data.height_map_url ? `${API_URL}${data.height_map_url}` : null)
        setRoughnessMapUrl(data.roughness_map_url ? `${API_URL}${data.roughness_map_url}` : null)
        setEmissiveMapUrl(data.emissive_map_url ? `${API_URL}${data.emissive_map_url}` : null)
      }
    } catch (error) {
      console.error('Apply layers failed:', error)
    } finally {
      setApplyingLayers(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🎨 Text to Material</h1>
        <p>AI-powered PBR texture generation</p>
      </header>

      <main className="main">
        {/* Left Panel - Generation */}
        <aside className="panel left-panel">
          {/* Mode Toggle */}
          <div className="mode-toggle">
            <button
              className={mode === 'generate' ? 'active' : ''}
              onClick={() => setMode('generate')}
            >
              ✨ Generate
            </button>
            <button
              className={mode === 'edit' ? 'active' : ''}
              onClick={() => setMode('edit')}
              disabled={!textureUrl}
            >
              🖌️ Edit
            </button>
            <button
              className={mode === 'layers' ? 'active' : ''}
              onClick={() => setMode('layers')}
            >
              🧩 Layers
            </button>
          </div>

          {mode === 'generate' ? (
            <>
              <section className="section">
                <h2>Material Presets</h2>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="category-select"
                >
                  <option value="">Select category...</option>
                  {materialPresets.map(cat => (
                    <option key={cat.category} value={cat.category}>
                      {cat.category}
                    </option>
                  ))}
                </select>

                {selectedCategory && (
                  <div className="preset-grid">
                    {materialPresets
                      .find(c => c.category === selectedCategory)
                      ?.materials.map(mat => (
                        <button
                          key={mat.name}
                          className="preset-btn"
                          onClick={() => handlePresetSelect(mat)}
                        >
                          {mat.name}
                        </button>
                      ))
                    }
                  </div>
                )}
              </section>

              <section className="section">
                <h2>Prompt</h2>
                <textarea
                  className="prompt-input"
                  placeholder="Describe your material... e.g., 'weathered copper with green patina'"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={3}
                />

                <label>
                  Weathering: {promptModifiers.weathering[weathering]}
                  <input
                    type="range"
                    min="0" max="4" step="1"
                    value={weathering}
                    onChange={(e) => setWeathering(parseInt(e.target.value))}
                  />
                </label>
              </section>
            </>
          ) : mode === 'edit' ? (
            <section className="section">
              <h2>Edit Material</h2>
              <textarea
                className="prompt-input"
                placeholder="Describe the edit... e.g., 'add rust spots', 'make it more blue'"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                rows={3}
              />
              <label>
                Edit Strength: {editStrength.toFixed(2)}
                <input
                  type="range"
                  min="0.1" max="1" step="0.05"
                  value={editStrength}
                  onChange={(e) => setEditStrength(parseFloat(e.target.value))}
                />
              </label>
              <p className="hint">Lower = subtle changes, Higher = dramatic changes</p>
            </section>
          ) : (
            <LayerStack
              layers={layers}
              onAddLayer={addLayer}
              onRemoveLayer={removeLayer}
              onUpdateLayer={updateLayer}
              onMoveLayer={moveLayer}
              onDuplicateLayer={duplicateLayer}
              onApply={handleApplyLayers}
              applying={applyingLayers}
            />
          )}

          {mode !== 'layers' && (
            <>
              <section className="section">
                <h2>Tiling</h2>
                <select value={tilingMode} onChange={(e) => setTilingMode(e.target.value)}>
                  <option value="none">None</option>
                  <option value="basic">Basic</option>
                  <option value="multipass">Multi-pass</option>
                  <option value="multiscale">Multi-scale</option>
                </select>
              </section>

              <button
                className={`generate-btn ${generating ? 'processing' : ''}`}
                onClick={mode === 'generate' ? handleGenerate : handleEdit}
                disabled={generating || (mode === 'generate' ? !prompt.trim() : !editPrompt.trim())}
              >
                {generating ? '⏳ Generating... Please wait' : mode === 'generate' ? '✨ Generate Material' : '🖌️ Apply Edit'}
              </button>
            </>
          )}
        </aside>

        {/* Center - 3D Preview */}
        <div className="preview-panel">
          {generating && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
              <div className="loading-text">
                {mode === 'generate' ? '✨ Generating Material...' : '🖌️ Applying Edit...'}
              </div>
              <div className="loading-subtext">This may take 30-60 seconds</div>
            </div>
          )}
          <Suspense fallback={<div className="loading">Loading 3D Preview...</div>}>
            <MaterialPreview
              textureUrl={activeTextureUrl}
              normalMapUrl={normalMapUrl}
              heightMapUrl={heightMapUrl}
              roughnessMapUrl={roughnessMapUrl}
              emissiveMapUrl={emissiveMapUrl}
              roughness={roughness}
              metalness={metalness}
              normalScale={normalScale}
              displacementScale={displacementScale}
              emissiveIntensity={emissiveIntensity}
              autoRotate={autoRotate}
              environment={environment}
              normalFlipR={normalFlipR}
              normalFlipG={normalFlipG}
              normalFlipB={normalFlipB}
            />
          </Suspense>
          {activeTextureUrl && (
            <div className="texture-preview">
              <img src={activeTextureUrl} alt="Generated texture" />
            </div>
          )}
        </div>

        {/* Right Panel - Viewport & Export */}
        <aside className="panel right-panel">
          <section className="section">
            <h2>Material Properties</h2>
            <label>
              Roughness: {roughness.toFixed(2)}
              <input
                type="range"
                min="0" max="1" step="0.01"
                value={roughness}
                onChange={(e) => setRoughness(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Metalness: {metalness.toFixed(2)}
              <input
                type="range"
                min="0" max="1" step="0.01"
                value={metalness}
                onChange={(e) => setMetalness(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Normal Strength: {normalScale.toFixed(2)}
              <input
                type="range"
                min="0" max="2" step="0.01"
                value={normalScale}
                onChange={(e) => setNormalScale(parseFloat(e.target.value))}
              />
            </label>
            <div className="channel-flip-row">
              <span>Flip Normal:</span>
              <label className="channel-flip">
                <input
                  type="checkbox"
                  checked={normalFlipR}
                  onChange={(e) => setNormalFlipR(e.target.checked)}
                />
                R
              </label>
              <label className="channel-flip">
                <input
                  type="checkbox"
                  checked={normalFlipG}
                  onChange={(e) => setNormalFlipG(e.target.checked)}
                />
                G
              </label>
              <label className="channel-flip">
                <input
                  type="checkbox"
                  checked={normalFlipB}
                  onChange={(e) => setNormalFlipB(e.target.checked)}
                />
                B
              </label>
            </div>
            <label>
              Displacement: {displacementScale.toFixed(3)}
              <input
                type="range"
                min="0" max="0.3" step="0.005"
                value={displacementScale}
                onChange={(e) => setDisplacementScale(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Emissive: {emissiveIntensity.toFixed(2)}
              <input
                type="range"
                min="0" max="2" step="0.01"
                value={emissiveIntensity}
                onChange={(e) => setEmissiveIntensity(parseFloat(e.target.value))}
              />
            </label>
          </section>

          <section className="section">
            <h2>Viewport</h2>
            <label>
              <input
                type="checkbox"
                checked={autoRotate}
                onChange={(e) => setAutoRotate(e.target.checked)}
              />
              Auto-rotate
            </label>
            <label>
              Environment:
              <select value={environment} onChange={(e) => setEnvironment(e.target.value)}>
                <option value="studio">Studio</option>
                <option value="sunset">Sunset</option>
                <option value="dawn">Dawn</option>
                <option value="night">Night</option>
                <option value="warehouse">Warehouse</option>
                <option value="forest">Forest</option>
                <option value="apartment">Apartment</option>
                <option value="city">City</option>
                <option value="park">Park</option>
                <option value="lobby">Lobby</option>
              </select>
            </label>
          </section>

          <section className="section">
            <h2>Export</h2>
            <button
              className="export-btn"
              onClick={handleExport}
              disabled={!textureUrl}
            >
              📦 Export for Unreal
            </button>
            <p className="export-info">
              Includes: Textures, Python script for UE5
            </p>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App
