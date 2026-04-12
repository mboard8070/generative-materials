import { useState, useEffect, useRef, Suspense, useCallback } from 'react'
import MaterialPreview from './components/MaterialPreview'
import LayerStack from './components/LayerStack'
import { materialPresets, promptModifiers } from './data/materialPresets'
import { useLayerStack } from './hooks/useLayerStack'
import { compositePbrLayers } from './lib/compositor'
import type { PbrMaps } from './types/layers'
import './App.css'

const API_URL = ''

type Mode = 'generate' | 'image-to-pbr' | 'extract' | 'edit' | 'layers' | 'library'

interface SavedMaterial {
  id: string
  name: string
  prompt: string
  engine: string
  seed: number | null
  thumbnail_url?: string
  created: string
}

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
  const [aoMapUrl, setAoMapUrl] = useState<string | null>(null)
  const [metallicMapUrl, setMetallicMapUrl] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState('')
  const [textureId, setTextureId] = useState<string | null>(null)

  // Seed
  const [seed, setSeed] = useState<string>('')
  const [lastSeed, setLastSeed] = useState<number | null>(null)

  // Edit state
  const [editPrompt, setEditPrompt] = useState('')
  const [editStrength, setEditStrength] = useState(0.7)

  // Image-to-PBR state
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadPreview, setUploadPreview] = useState<string | null>(null)

  // Extract state
  const [extractFile, setExtractFile] = useState<File | null>(null)
  const [extractPreview, setExtractPreview] = useState<string | null>(null)
  const [extractLabel, setExtractLabel] = useState('')

  // Library
  const [libraryItems, setLibraryItems] = useState<SavedMaterial[]>([])
  const [saveName, setSaveName] = useState('')
  const [saving, setSaving] = useState(false)

  // Material properties
  const [roughness, setRoughness] = useState(1.0)
  const [metalness, setMetalness] = useState(0.0)
  const [normalScale, setNormalScale] = useState(1.0)
  const [displacementScale, setDisplacementScale] = useState(0.1)
  const [emissiveIntensity, setEmissiveIntensity] = useState(0.0)
  const [aoIntensity, setAoIntensity] = useState(1.0)

  // Translucency & SSS
  const [transmission, setTransmission] = useState(0.0)
  const [thickness, setThickness] = useState(0.0)
  const [subsurfaceColor, setSubsurfaceColor] = useState('#ffffff')
  const [translucencyMapUrl, _setTranslucencyMapUrl] = useState<string | null>(null)
  const [subsurfaceMapUrl, _setSubsurfaceMapUrl] = useState<string | null>(null)

  // Geometry
  const [geometry, setGeometry] = useState<'sphere' | 'plane' | 'cube' | 'custom'>('sphere')
  const [customMeshUrl, setCustomMeshUrl] = useState<string | null>(null)

  // Height adjustment
  const [heightContrast, setHeightContrast] = useState(1.0)
  const [heightBrightness, setHeightBrightness] = useState(0.0)
  const [heightInvert, setHeightInvert] = useState(false)
  const [heightBlur, setHeightBlur] = useState(0.0)
  const heightDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Specular
  const [specularIntensity, setSpecularIntensity] = useState(0.5)

  // Lighting
  const [envIntensity, setEnvIntensity] = useState(1.0)
  const [keyLightIntensity, setKeyLightIntensity] = useState(1.8)
  const [fillLightIntensity, setFillLightIntensity] = useState(0.4)
  const [rimLightIntensity, setRimLightIntensity] = useState(0.25)

  // Viewport settings
  const [autoRotate, setAutoRotate] = useState(true)
  const [environment, setEnvironment] = useState('studio')

  // Engine selection
  const [engine, setEngine] = useState<'flux' | 'patina'>('patina')

  // PATINA options
  const [upscaleFactor, setUpscaleFactor] = useState(0)
  const [promptExpansion, setPromptExpansion] = useState(true)

  // Tiling settings
  const [tilingMode, setTilingMode] = useState('basic')
  const [tileRepeat, setTileRepeat] = useState(2)

  // API status
  const [apiReady, setApiReady] = useState(false)
  const [patinaReady, setPatinaReady] = useState(false)
  const [apiStep, setApiStep] = useState('Connecting...')
  const [apiProgress, setApiProgress] = useState(0)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const res = await fetch(`${API_URL}/status`)
        const data = await res.json()
        if (cancelled) return
        setApiProgress(data.progress ?? 0)
        setApiStep(data.step ?? 'Loading...')
        if (data.patina_ready) setPatinaReady(true)
        if (data.ready) {
          setApiReady(true)
          return
        }
      } catch {
        if (cancelled) return
        setApiStep('Connecting to server...')
        setApiProgress(0)
      }
      if (!cancelled) setTimeout(poll, 2000)
    }
    poll()
    return () => { cancelled = true }
  }, [])

  // Modifiers
  const [weathering, setWeathering] = useState(1)

  // Normal map channel flips
  const [normalFlipR, setNormalFlipR] = useState(false)
  const [normalFlipG, setNormalFlipG] = useState(false)
  const [normalFlipB, setNormalFlipB] = useState(false)

  // Layer stack
  const { layers, addLayer, removeLayer, updateLayer, moveLayer, duplicateLayer, clearLayers } = useLayerStack()
  const [compositedMaps, setCompositedMaps] = useState<{
    basecolor: string | null; normal: string | null; roughness: string | null;
    metalness: string | null; height: string | null;
  }>({ basecolor: null, normal: null, roughness: null, metalness: null, height: null })
  const [applyingLayers, setApplyingLayers] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Height adjustment — debounced API call
  const adjustHeight = useCallback(() => {
    if (!textureId) return
    if (heightContrast === 1.0 && heightBrightness === 0.0 && !heightInvert && heightBlur === 0.0) {
      // Reset to original
      setHeightMapUrl(`${API_URL}/outputs/${textureId}_height.png`)
      return
    }
    if (heightDebounce.current) clearTimeout(heightDebounce.current)
    heightDebounce.current = setTimeout(async () => {
      try {
        const res = await fetch(`${API_URL}/adjust-height`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            texture_id: textureId,
            contrast: heightContrast,
            brightness: heightBrightness,
            invert: heightInvert,
            blur_radius: heightBlur,
          })
        })
        if (res.ok) {
          const data = await res.json()
          setHeightMapUrl(`${API_URL}${data.height_map_url}?t=${Date.now()}`)
        }
      } catch (e) {
        console.error('Height adjust failed:', e)
      }
    }, 300)
  }, [textureId, heightContrast, heightBrightness, heightInvert, heightBlur])

  useEffect(() => {
    adjustHeight()
  }, [adjustHeight])

  // Active maps: in layers mode use composited, otherwise use generated
  const activeTextureUrl = mode === 'layers' ? (compositedMaps.basecolor ?? textureUrl) : textureUrl
  const activeNormalUrl = mode === 'layers' ? (compositedMaps.normal ?? normalMapUrl) : normalMapUrl
  const activeHeightUrl = mode === 'layers' ? (compositedMaps.height ?? heightMapUrl) : heightMapUrl
  const activeRoughnessUrl = mode === 'layers' ? (compositedMaps.roughness ?? roughnessMapUrl) : roughnessMapUrl
  const activeMetallicUrl = mode === 'layers' ? (compositedMaps.metalness ?? metallicMapUrl) : metallicMapUrl

  // Generate material for a specific layer via PATINA
  const handleGenerateLayer = useCallback(async (layerId: string, prompt: string) => {
    if (!prompt.trim()) return
    updateLayer(layerId, { generating: true })
    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, engine: 'patina', tiling_mode: 'none' }),
      })
      if (res.ok) {
        const data = await res.json()
        updateLayer(layerId, {
          generating: false,
          materialMaps: {
            basecolor: data.texture_url ? `${API_URL}${data.texture_url}` : undefined,
            normal: data.normal_map_url ? `${API_URL}${data.normal_map_url}` : undefined,
            roughness: data.roughness_map_url ? `${API_URL}${data.roughness_map_url}` : undefined,
            metalness: data.metallic_map_url ? `${API_URL}${data.metallic_map_url}` : undefined,
            height: data.height_map_url ? `${API_URL}${data.height_map_url}` : undefined,
          },
        })
      } else {
        updateLayer(layerId, { generating: false })
      }
    } catch {
      updateLayer(layerId, { generating: false })
    }
  }, [updateLayer])

  // Composite all PBR channels when layers change (debounced)
  const runComposite = useCallback(() => {
    if (mode !== 'layers') return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      const baseMaps: PbrMaps = {
        basecolor: textureUrl ?? undefined,
        normal: normalMapUrl ?? undefined,
        roughness: roughnessMapUrl ?? undefined,
        metalness: metallicMapUrl ?? undefined,
        height: heightMapUrl ?? undefined,
      }
      try {
        const result = await compositePbrLayers(layers, baseMaps)
        setCompositedMaps(result)
      } catch (err) {
        console.warn('PBR compositing failed:', err)
      }
    }, 100)
  }, [mode, layers, textureUrl, normalMapUrl, roughnessMapUrl, metallicMapUrl, heightMapUrl])

  useEffect(() => {
    runComposite()
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [runComposite])

  // --- Shared response handler ---
  const applyResult = (data: any) => {
    setTextureUrl(data.texture_url ? `${API_URL}${data.texture_url}` : null)
    setNormalMapUrl(data.normal_map_url ? `${API_URL}${data.normal_map_url}` : null)
    setHeightMapUrl(data.height_map_url ? `${API_URL}${data.height_map_url}` : null)
    setRoughnessMapUrl(data.roughness_map_url ? `${API_URL}${data.roughness_map_url}` : null)
    setEmissiveMapUrl(data.emissive_map_url ? `${API_URL}${data.emissive_map_url}` : null)
    setMetallicMapUrl(data.metallic_map_url ? `${API_URL}${data.metallic_map_url}` : null)
    setAoMapUrl(data.ao_map_url ? `${API_URL}${data.ao_map_url}` : null)
    setTextureId(data.texture_id ?? null)
    if (data.seed != null) setLastSeed(data.seed)
    // Reset height adjustments
    setHeightContrast(1.0)
    setHeightBrightness(0.0)
    setHeightInvert(false)
    setHeightBlur(0.0)
  }

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

  // --- Generate (text to material) ---
  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setGenerating(true)
    try {
      const body: any = {
        prompt: engine === 'patina' ? prompt : buildFullPrompt(),
        tiling_mode: tilingMode,
        engine,
      }
      if (seed) body.seed = parseInt(seed)
      if (engine === 'patina') {
        body.upscale_factor = upscaleFactor
        body.enable_prompt_expansion = promptExpansion
      }
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      if (res.ok) applyResult(await res.json())
    } catch (error) {
      console.error('Generation failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  // --- Image to PBR ---
  const handleImageToPbr = async () => {
    if (!uploadFile) return
    setGenerating(true)
    try {
      const form = new FormData()
      form.append('image', uploadFile)
      if (seed) form.append('seed', seed)
      const res = await fetch(`${API_URL}/image-to-pbr`, { method: 'POST', body: form })
      if (res.ok) applyResult(await res.json())
    } catch (error) {
      console.error('Image-to-PBR failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  // --- Extract material ---
  const handleExtract = async () => {
    if (!extractFile || !extractLabel.trim()) return
    setGenerating(true)
    try {
      const form = new FormData()
      form.append('image', extractFile)
      form.append('label', extractLabel)
      form.append('upscale_factor', upscaleFactor.toString())
      if (seed) form.append('seed', seed)
      const res = await fetch(`${API_URL}/extract-material`, { method: 'POST', body: form })
      if (res.ok) applyResult(await res.json())
    } catch (error) {
      console.error('Extract failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  // --- Edit ---
  const handleEdit = async () => {
    if (!textureUrl || !editPrompt.trim()) return
    setGenerating(true)
    try {
      const res = await fetch(`${API_URL}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: textureUrl,
          prompt: editPrompt,
          strength: editStrength,
          tiling_mode: tilingMode
        })
      })
      if (res.ok) applyResult(await res.json())
    } catch (error) {
      console.error('Edit failed:', error)
    } finally {
      setGenerating(false)
    }
  }

  // --- Export ---
  const handleExport = async () => {
    if (!textureUrl) return
    try {
      const res = await fetch(`${API_URL}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texture_url: textureUrl, roughness, metalness, format: 'unreal' })
      })
      if (res.ok) {
        const blob = await res.blob()
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

  // --- Download individual map ---
  const downloadMap = (url: string | null, name: string) => {
    if (!url) return
    const a = document.createElement('a')
    a.href = url
    a.download = `${textureId}_${name}.png`
    a.click()
  }

  // --- Download all maps ---
  const handleDownloadAll = async () => {
    if (!textureId) return
    const a = document.createElement('a')
    a.href = `${API_URL}/download-all/${textureId}`
    a.download = `${textureId}_maps.zip`
    a.click()
  }

  // --- Apply layers: flatten composited maps into base ---
  const handleApplyLayers = () => {
    setApplyingLayers(true)
    // Promote composited data URLs to base map state
    if (compositedMaps.basecolor) setTextureUrl(compositedMaps.basecolor)
    if (compositedMaps.normal) setNormalMapUrl(compositedMaps.normal)
    if (compositedMaps.roughness) setRoughnessMapUrl(compositedMaps.roughness)
    if (compositedMaps.metalness) setMetallicMapUrl(compositedMaps.metalness)
    if (compositedMaps.height) setHeightMapUrl(compositedMaps.height)
    clearLayers()
    setCompositedMaps({ basecolor: null, normal: null, roughness: null, metalness: null, height: null })
    setApplyingLayers(false)
  }

  // --- File upload handler ---
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, target: 'upload' | 'extract') => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    if (target === 'upload') {
      setUploadFile(file)
      setUploadPreview(url)
    } else {
      setExtractFile(file)
      setExtractPreview(url)
    }
  }

  // --- Library ---
  const loadLibrary = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/library`)
      if (res.ok) setLibraryItems(await res.json())
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    if (mode === 'library') loadLibrary()
  }, [mode, loadLibrary])

  const handleSaveMaterial = async () => {
    if (!textureId || !saveName.trim()) return
    setSaving(true)
    try {
      await fetch(`${API_URL}/library/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texture_id: textureId,
          name: saveName,
          prompt,
          engine,
          seed: lastSeed,
        })
      })
      setSaveName('')
      loadLibrary()
    } catch (e) {
      console.error('Save failed:', e)
    } finally {
      setSaving(false)
    }
  }

  const handleLoadMaterial = async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/library/${id}`)
      if (res.ok) {
        const data = await res.json()
        applyResult(data)
        if (data.prompt) setPrompt(data.prompt)
        setMode('generate')
      }
    } catch (e) {
      console.error('Load failed:', e)
    }
  }

  const handleDeleteMaterial = async (id: string) => {
    try {
      await fetch(`${API_URL}/library/${id}`, { method: 'DELETE' })
      loadLibrary()
    } catch (e) {
      console.error('Delete failed:', e)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="2" width="12" height="12" rx="2" fill="#6366f1"/>
            <rect x="18" y="2" width="12" height="12" rx="2" fill="#818cf8"/>
            <rect x="2" y="18" width="12" height="12" rx="2" fill="#818cf8"/>
            <rect x="18" y="18" width="12" height="12" rx="2" fill="#a5b4fc"/>
            <path d="M8 8L24 24M24 8L8 24" stroke="white" strokeWidth="2" strokeLinecap="round" opacity="0.6"/>
          </svg>
          <h1>Surfaced</h1>
        </div>
        <p>AI-powered PBR material generation</p>
        {!apiReady && (
          <div className="api-loading">
            <div className="api-loading-bar">
              <div className="api-loading-fill" style={{ width: `${apiProgress}%` }} />
            </div>
            <span className="api-loading-text">{apiStep}</span>
          </div>
        )}
      </header>

      <main className="main">
        {/* Left Panel */}
        <aside className="panel left-panel">
          {/* Mode Selector */}
          <select
            className="category-select"
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
          >
            <option value="generate">Generate</option>
            <option value="image-to-pbr">Image to PBR</option>
            <option value="extract">Extract Material</option>
            <option value="edit" disabled={!textureUrl}>Edit</option>
            <option value="layers">Layers</option>
            <option value="library">Library</option>
          </select>

          {/* ---- GENERATE MODE ---- */}
          {mode === 'generate' && (
            <>
              <section className="section">
                <h2>Material Presets</h2>
                <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)} className="category-select">
                  <option value="">Select category...</option>
                  {materialPresets.map(cat => (
                    <option key={cat.category} value={cat.category}>{cat.category}</option>
                  ))}
                </select>
                {selectedCategory && (
                  <div className="preset-grid">
                    {materialPresets.find(c => c.category === selectedCategory)?.materials.map(mat => (
                      <button key={mat.name} className="preset-btn" onClick={() => handlePresetSelect(mat)}>
                        {mat.name}
                      </button>
                    ))}
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
                  <input type="range" min="0" max="4" step="1" value={weathering}
                    onChange={(e) => setWeathering(parseInt(e.target.value))} />
                </label>
              </section>

              <section className="section">
                <h2>Engine</h2>
                <div className="mode-toggle">
                  <button className={engine === 'patina' ? 'active' : ''} onClick={() => setEngine('patina')}>
                    PATINA (fal.ai)
                  </button>
                  <button className={engine === 'flux' ? 'active' : ''} onClick={() => setEngine('flux')}>
                    Flux + LoRA
                  </button>
                </div>
                <p className="hint">
                  {engine === 'patina'
                    ? 'Fast cloud API (~15s, ~$0.08) — native PBR maps'
                    : 'Local Flux.1-dev + LoRA (~45s, free) — grid split maps'}
                </p>
              </section>

              {engine === 'patina' && (
                <section className="section">
                  <h2>PATINA Options</h2>
                  <label>
                    Upscale:
                    <select value={upscaleFactor} onChange={(e) => setUpscaleFactor(parseInt(e.target.value))}>
                      <option value={0}>None (1024px)</option>
                      <option value={2}>2x (2048px)</option>
                      <option value={4}>4x (4096px)</option>
                    </select>
                  </label>
                  <label>
                    <input type="checkbox" checked={promptExpansion}
                      onChange={(e) => setPromptExpansion(e.target.checked)} />
                    Prompt expansion
                  </label>
                  <p className="hint">Auto-enhance terse prompts for better results</p>
                </section>
              )}
            </>
          )}

          {/* ---- IMAGE-TO-PBR MODE ---- */}
          {mode === 'image-to-pbr' && (
            <section className="section">
              <h2>Image to PBR Maps</h2>
              <p className="hint">Upload a texture image to generate PBR maps from it</p>
              <input type="file" accept="image/*" onChange={(e) => handleFileSelect(e, 'upload')} />
              {uploadPreview && (
                <img src={uploadPreview} alt="Upload preview" style={{ width: '100%', marginTop: 8, borderRadius: 4 }} />
              )}
            </section>
          )}

          {/* ---- EXTRACT MODE ---- */}
          {mode === 'extract' && (
            <section className="section">
              <h2>Extract Material</h2>
              <p className="hint">Upload a scene photo and label the material to extract</p>
              <input type="file" accept="image/*" onChange={(e) => handleFileSelect(e, 'extract')} />
              {extractPreview && (
                <img src={extractPreview} alt="Extract preview" style={{ width: '100%', marginTop: 8, borderRadius: 4 }} />
              )}
              <input
                type="text"
                className="prompt-input"
                placeholder="Material label... e.g., 'brick wall', 'wood floor'"
                value={extractLabel}
                onChange={(e) => setExtractLabel(e.target.value)}
                style={{ marginTop: 8 }}
              />
              <label style={{ marginTop: 8 }}>
                Upscale:
                <select value={upscaleFactor} onChange={(e) => setUpscaleFactor(parseInt(e.target.value))}>
                  <option value={0}>None</option>
                  <option value={2}>2x</option>
                  <option value={4}>4x</option>
                </select>
              </label>
            </section>
          )}

          {/* ---- EDIT MODE ---- */}
          {mode === 'edit' && (
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
                <input type="range" min="0.1" max="1" step="0.05" value={editStrength}
                  onChange={(e) => setEditStrength(parseFloat(e.target.value))} />
              </label>
              <p className="hint">Lower = subtle changes, Higher = dramatic changes</p>
            </section>
          )}

          {/* ---- LAYERS MODE ---- */}
          {mode === 'layers' && (
            <LayerStack
              layers={layers}
              onAddLayer={addLayer}
              onRemoveLayer={removeLayer}
              onUpdateLayer={updateLayer}
              onMoveLayer={moveLayer}
              onDuplicateLayer={duplicateLayer}
              onGenerateLayer={handleGenerateLayer}
              onApply={handleApplyLayers}
              applying={applyingLayers}
            />
          )}

          {/* ---- LIBRARY MODE ---- */}
          {mode === 'library' && (
            <>
              {textureId && (
                <section className="section">
                  <h2>Save Current Material</h2>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input
                      type="text"
                      placeholder="Material name..."
                      value={saveName}
                      onChange={(e) => setSaveName(e.target.value)}
                      style={{ flex: 1 }}
                    />
                    <button
                      className="generate-btn"
                      onClick={handleSaveMaterial}
                      disabled={saving || !saveName.trim()}
                      style={{ padding: '6px 16px', whiteSpace: 'nowrap' }}
                    >
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </section>
              )}

              <section className="section">
                <h2>Saved Materials ({libraryItems.length})</h2>
              </section>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflowY: 'auto' }}>
                {libraryItems.map(item => (
                  <div key={item.id} style={{
                    display: 'flex', gap: 8, alignItems: 'center',
                    padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: 6,
                  }}>
                    {item.thumbnail_url && (
                      <img
                        src={`${API_URL}${item.thumbnail_url}`}
                        alt={item.name}
                        style={{ width: 56, height: 56, borderRadius: 4, objectFit: 'cover', cursor: 'pointer' }}
                        onClick={() => handleLoadMaterial(item.id)}
                      />
                    )}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.85rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.name}
                      </div>
                      <div style={{ fontSize: '0.7rem', opacity: 0.6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.prompt || 'No prompt'}
                      </div>
                      <div style={{ fontSize: '0.65rem', opacity: 0.4 }}>
                        {item.engine} &middot; {item.created}
                      </div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <button className="preset-btn" onClick={() => handleLoadMaterial(item.id)} style={{ fontSize: '0.7rem', padding: '3px 8px' }}>
                        Load
                      </button>
                      <button className="preset-btn" onClick={() => handleDeleteMaterial(item.id)} style={{ fontSize: '0.7rem', padding: '3px 8px', opacity: 0.6 }}>
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
                {libraryItems.length === 0 && (
                  <p className="hint" style={{ textAlign: 'center', padding: '1rem' }}>
                    No saved materials yet. Generate a material, then come here to save it.
                  </p>
                )}
              </div>
            </>
          )}

          {/* ---- Shared bottom controls ---- */}
          {mode !== 'layers' && mode !== 'library' && (
            <>
              {(mode === 'generate' || mode === 'edit') && (
                <section className="section">
                  <h2>Tiling</h2>
                  <select value={tilingMode} onChange={(e) => setTilingMode(e.target.value)}>
                    <option value="none">None</option>
                    <option value="basic">Basic</option>
                    <option value="multipass">Multi-pass</option>
                    <option value="multiscale">Multi-scale</option>
                  </select>
                </section>
              )}

              <section className="section">
                <h2>Seed</h2>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="text"
                    placeholder="Random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value.replace(/\D/g, ''))}
                    style={{ flex: 1 }}
                  />
                  {lastSeed != null && (
                    <button className="preset-btn" onClick={() => setSeed(lastSeed.toString())} title="Reuse last seed">
                      {lastSeed}
                    </button>
                  )}
                </div>
              </section>

              <button
                className={`generate-btn ${generating ? 'processing' : ''}`}
                onClick={
                  mode === 'generate' ? handleGenerate
                  : mode === 'image-to-pbr' ? handleImageToPbr
                  : mode === 'extract' ? handleExtract
                  : handleEdit
                }
                disabled={
                  generating ||
                  (mode === 'generate' && ((engine === 'flux' && !apiReady) || (engine === 'patina' && !patinaReady) || !prompt.trim())) ||
                  (mode === 'image-to-pbr' && (!patinaReady || !uploadFile)) ||
                  (mode === 'extract' && (!patinaReady || !extractFile || !extractLabel.trim())) ||
                  (mode === 'edit' && !editPrompt.trim())
                }
              >
                {generating ? 'Generating... Please wait'
                  : mode === 'generate' ? 'Generate Material'
                  : mode === 'image-to-pbr' ? 'Generate PBR Maps'
                  : mode === 'extract' ? 'Extract Material'
                  : 'Apply Edit'}
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
                {mode === 'generate' ? 'Generating Material...'
                  : mode === 'image-to-pbr' ? 'Analyzing Image...'
                  : mode === 'extract' ? 'Extracting Material...'
                  : 'Applying Edit...'}
              </div>
              <div className="loading-subtext">
                {engine === 'patina' || mode === 'image-to-pbr' || mode === 'extract'
                  ? 'This takes about 15 seconds'
                  : 'This may take 30-60 seconds'}
              </div>
            </div>
          )}
          <Suspense fallback={<div className="loading">Loading 3D Preview...</div>}>
            <MaterialPreview
              textureUrl={activeTextureUrl}
              normalMapUrl={activeNormalUrl}
              heightMapUrl={activeHeightUrl}
              roughnessMapUrl={activeRoughnessUrl}
              emissiveMapUrl={emissiveMapUrl}
              aoMapUrl={aoMapUrl}
              metallicMapUrl={activeMetallicUrl}
              translucencyMapUrl={translucencyMapUrl}
              subsurfaceMapUrl={subsurfaceMapUrl}
              roughness={roughness}
              metalness={metalness}
              specularIntensity={specularIntensity}
              normalScale={normalScale}
              displacementScale={displacementScale}
              emissiveIntensity={emissiveIntensity}
              aoIntensity={aoIntensity}
              transmission={transmission}
              thickness={thickness}
              subsurfaceColor={subsurfaceColor}
              autoRotate={autoRotate}
              environment={environment}
              envIntensity={envIntensity}
              keyLightIntensity={keyLightIntensity}
              fillLightIntensity={fillLightIntensity}
              rimLightIntensity={rimLightIntensity}
              normalFlipR={normalFlipR}
              normalFlipG={normalFlipG}
              normalFlipB={normalFlipB}
              tileRepeat={tileRepeat}
              geometry={geometry}
              customMeshUrl={customMeshUrl}
            />
          </Suspense>
          {activeTextureUrl && (
            <div className="texture-preview">
              <img src={activeTextureUrl} alt="Generated texture" />
            </div>
          )}
        </div>

        {/* Right Panel */}
        <aside className="panel right-panel">
          <section className="section">
            <h2>Material Properties</h2>
            <label>
              Roughness: {roughness.toFixed(2)}
              <input type="range" min="0" max="1" step="0.01" value={roughness}
                onChange={(e) => setRoughness(parseFloat(e.target.value))} />
            </label>
            <label>
              Metalness: {metalness.toFixed(2)}
              <input type="range" min="0" max="1" step="0.01" value={metalness}
                onChange={(e) => setMetalness(parseFloat(e.target.value))} />
            </label>
            <label>
              Specular: {specularIntensity.toFixed(2)}
              <input type="range" min="0" max="1" step="0.01" value={specularIntensity}
                onChange={(e) => setSpecularIntensity(parseFloat(e.target.value))} />
            </label>
            <label>
              Normal Strength: {normalScale.toFixed(2)}
              <input type="range" min="0" max="2" step="0.01" value={normalScale}
                onChange={(e) => setNormalScale(parseFloat(e.target.value))} />
            </label>
            <div className="channel-flip-row">
              <span>Flip Normal:</span>
              <label className="channel-flip">
                <input type="checkbox" checked={normalFlipR} onChange={(e) => setNormalFlipR(e.target.checked)} /> R
              </label>
              <label className="channel-flip">
                <input type="checkbox" checked={normalFlipG} onChange={(e) => setNormalFlipG(e.target.checked)} /> G
              </label>
              <label className="channel-flip">
                <input type="checkbox" checked={normalFlipB} onChange={(e) => setNormalFlipB(e.target.checked)} /> B
              </label>
            </div>
            <label>
              Displacement: {displacementScale.toFixed(3)}
              <input type="range" min="0" max="0.3" step="0.005" value={displacementScale}
                onChange={(e) => setDisplacementScale(parseFloat(e.target.value))} />
            </label>
            <label>
              Emissive: {emissiveIntensity.toFixed(2)}
              <input type="range" min="0" max="2" step="0.01" value={emissiveIntensity}
                onChange={(e) => setEmissiveIntensity(parseFloat(e.target.value))} />
            </label>
            <label>
              AO Intensity: {aoIntensity.toFixed(2)}
              <input type="range" min="0" max="2" step="0.01" value={aoIntensity}
                onChange={(e) => setAoIntensity(parseFloat(e.target.value))} />
            </label>
          </section>

          {/* Translucency & SSS */}
          <section className="section">
            <h2>Translucency / SSS</h2>
            <label>
              <input type="checkbox" checked={transmission > 0}
                onChange={(e) => setTransmission(e.target.checked ? 0.5 : 0)} />
              Enable Translucency
            </label>
            {transmission > 0 && (
              <label>
                Transmission: {transmission.toFixed(2)}
                <input type="range" min="0.01" max="1" step="0.01" value={transmission}
                  onChange={(e) => setTransmission(parseFloat(e.target.value))} />
              </label>
            )}
            <label>
              <input type="checkbox" checked={thickness > 0}
                onChange={(e) => setThickness(e.target.checked ? 1.0 : 0)} />
              Enable SSS
            </label>
            {thickness > 0 && (
              <>
                <label>
                  Thickness: {thickness.toFixed(2)}
                  <input type="range" min="0.05" max="5" step="0.05" value={thickness}
                    onChange={(e) => setThickness(parseFloat(e.target.value))} />
                </label>
                <label>
                  SSS Color
                  <input type="color" value={subsurfaceColor}
                    onChange={(e) => setSubsurfaceColor(e.target.value)} />
                </label>
              </>
            )}
          </section>

          {/* Height Map Controls */}
          <section className="section">
            <h2>Height Map</h2>
            <label>
              Contrast: {heightContrast.toFixed(2)}
              <input type="range" min="0.1" max="3.0" step="0.05" value={heightContrast}
                onChange={(e) => setHeightContrast(parseFloat(e.target.value))} />
            </label>
            <label>
              Brightness: {heightBrightness.toFixed(2)}
              <input type="range" min="-0.5" max="0.5" step="0.01" value={heightBrightness}
                onChange={(e) => setHeightBrightness(parseFloat(e.target.value))} />
            </label>
            <label>
              Blur: {heightBlur.toFixed(1)}
              <input type="range" min="0" max="10" step="0.5" value={heightBlur}
                onChange={(e) => setHeightBlur(parseFloat(e.target.value))} />
            </label>
            <label>
              <input type="checkbox" checked={heightInvert} onChange={(e) => setHeightInvert(e.target.checked)} />
              Invert height
            </label>
          </section>

          <section className="section">
            <h2>Lighting</h2>
            <label>
              Environment: {envIntensity.toFixed(2)}
              <input type="range" min="0" max="3" step="0.05" value={envIntensity}
                onChange={(e) => setEnvIntensity(parseFloat(e.target.value))} />
            </label>
            <label>
              Key Light: {keyLightIntensity.toFixed(2)}
              <input type="range" min="0" max="5" step="0.05" value={keyLightIntensity}
                onChange={(e) => setKeyLightIntensity(parseFloat(e.target.value))} />
            </label>
            <label>
              Fill Light: {fillLightIntensity.toFixed(2)}
              <input type="range" min="0" max="3" step="0.05" value={fillLightIntensity}
                onChange={(e) => setFillLightIntensity(parseFloat(e.target.value))} />
            </label>
            <label>
              Rim Light: {rimLightIntensity.toFixed(2)}
              <input type="range" min="0" max="3" step="0.05" value={rimLightIntensity}
                onChange={(e) => setRimLightIntensity(parseFloat(e.target.value))} />
            </label>
          </section>

          <section className="section">
            <h2>Viewport</h2>
            <label>
              Geometry:
              <select value={geometry} onChange={(e) => setGeometry(e.target.value as any)}>
                <option value="sphere">Sphere</option>
                <option value="plane">Plane</option>
                <option value="cube">Cube</option>
                <option value="custom">Custom Mesh</option>
              </select>
            </label>
            {geometry === 'custom' && (
              <div>
                <input type="file" accept=".glb,.gltf,.obj" onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) setCustomMeshUrl(URL.createObjectURL(file))
                }} />
                <p className="hint">Upload .glb, .gltf, or .obj</p>
              </div>
            )}
            <label>
              Tile Repeat: {tileRepeat}x{tileRepeat}
              <input type="range" min="1" max="8" step="1" value={tileRepeat}
                onChange={(e) => setTileRepeat(parseInt(e.target.value))} />
            </label>
            <label>
              <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} />
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

          {/* Download Maps */}
          <section className="section">
            <h2>Download Maps</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              <button className="preset-btn" disabled={!textureUrl} onClick={() => downloadMap(textureUrl, 'basecolor')}>Base Color</button>
              <button className="preset-btn" disabled={!normalMapUrl} onClick={() => downloadMap(normalMapUrl, 'normal')}>Normal</button>
              <button className="preset-btn" disabled={!roughnessMapUrl} onClick={() => downloadMap(roughnessMapUrl, 'roughness')}>Roughness</button>
              <button className="preset-btn" disabled={!metallicMapUrl} onClick={() => downloadMap(metallicMapUrl, 'metallic')}>Metallic</button>
              <button className="preset-btn" disabled={!heightMapUrl} onClick={() => downloadMap(heightMapUrl, 'height')}>Height</button>
              <button className="preset-btn" disabled={!emissiveMapUrl} onClick={() => downloadMap(emissiveMapUrl, 'emissive')}>Emissive</button>
            </div>
            <button className="export-btn" onClick={handleDownloadAll} disabled={!textureId} style={{ marginTop: 8 }}>
              Download All (ZIP)
            </button>
          </section>

          <section className="section">
            <h2>Export</h2>
            <button className="export-btn" onClick={handleExport} disabled={!textureUrl}>
              Export for Unreal
            </button>
            <p className="export-info">Includes: Textures, Python script for UE5</p>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App
