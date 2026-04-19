import { useRef, useEffect, useState, useMemo, useCallback, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Environment, useTexture } from '@react-three/drei'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { evaluateMotion } from '../types/motion'
import type { MotionState } from '../types/motion'

/**
 * Safely load an optional texture via THREE.TextureLoader.
 * Always calls the same hooks regardless of whether url is null,
 * avoiding the conditional-hook bug with drei's useTexture.
 */
function useOptionalTexture(url: string | null, tileRepeat: number = 2): THREE.Texture | null {
  const [texture, setTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    if (!url) {
      setTexture((prev) => {
        prev?.dispose()
        return null
      })
      return
    }
    const loader = new THREE.TextureLoader()
    let cancelled = false
    loader.load(url, (tex) => {
      if (cancelled) {
        tex.dispose()
        return
      }
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping
      tex.repeat.set(tileRepeat, tileRepeat)
      setTexture((prev) => {
        prev?.dispose()
        return tex
      })
    })
    return () => {
      cancelled = true
    }
  }, [url, tileRepeat])

  return texture
}

/**
 * Process a normal map texture with per-channel flips.
 * Flipping inverts that channel: value = 255 - value.
 */
function useFlippedNormalMap(
  source: THREE.Texture | null,
  flipR: boolean,
  flipG: boolean,
  flipB: boolean,
): THREE.Texture | null {
  const [flipped, setFlipped] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    // No flipping needed — pass through the source directly
    if (!source || (!flipR && !flipG && !flipB)) {
      setFlipped((prev) => {
        if (prev && prev !== source) prev.dispose()
        return source
      })
      return
    }

    const image = source.image as HTMLImageElement | HTMLCanvasElement
    if (!image) {
      setFlipped(source)
      return
    }

    const w = image.width || 512
    const h = image.height || 512
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(image, 0, 0)
    const imageData = ctx.getImageData(0, 0, w, h)
    const data = imageData.data
    for (let i = 0; i < data.length; i += 4) {
      if (flipR) data[i] = 255 - data[i]
      if (flipG) data[i + 1] = 255 - data[i + 1]
      if (flipB) data[i + 2] = 255 - data[i + 2]
    }
    ctx.putImageData(imageData, 0, 0)

    const tex = new THREE.CanvasTexture(canvas)
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping
    tex.repeat.copy(source.repeat)
    tex.needsUpdate = true

    setFlipped((prev) => {
      if (prev && prev !== source) prev.dispose()
      return tex
    })
  }, [source, flipR, flipG, flipB])

  return flipped
}

/**
 * Invert a grayscale texture: value = 255 - value.
 */
function useInvertedTexture(
  source: THREE.Texture | null,
  invert: boolean,
): THREE.Texture | null {
  const [result, setResult] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    if (!source || !invert) {
      setResult((prev) => {
        if (prev && prev !== source) prev.dispose()
        return source
      })
      return
    }

    const image = source.image as HTMLImageElement | HTMLCanvasElement
    if (!image) {
      setResult(source)
      return
    }

    const w = image.width || 512
    const h = image.height || 512
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(image, 0, 0)
    const imageData = ctx.getImageData(0, 0, w, h)
    const data = imageData.data
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i]
      data[i + 1] = 255 - data[i + 1]
      data[i + 2] = 255 - data[i + 2]
    }
    ctx.putImageData(imageData, 0, 0)

    const tex = new THREE.CanvasTexture(canvas)
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping
    tex.repeat.copy(source.repeat)
    tex.needsUpdate = true

    setResult((prev) => {
      if (prev && prev !== source) prev.dispose()
      return tex
    })
  }, [source, invert])

  return result
}

/**
 * Create a CanvasTexture from a paint canvas for real-time sphere painting.
 * Returns null when no canvas is provided, falling back to URL-loaded textures.
 */
function usePaintCanvasTexture(
  canvas: HTMLCanvasElement | null | undefined,
  tileRepeat: number,
): THREE.CanvasTexture | null {
  const [tex, setTex] = useState<THREE.CanvasTexture | null>(null)

  useEffect(() => {
    if (!canvas) {
      setTex(prev => { prev?.dispose(); return null })
      return
    }
    const t = new THREE.CanvasTexture(canvas)
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.repeat.set(tileRepeat, tileRepeat)
    setTex(prev => { prev?.dispose(); return t })
    return () => { t.dispose() }
  }, [canvas, tileRepeat])

  return tex
}

function useCustomGeometry(url: string | null): THREE.BufferGeometry | null {
  const [geo, setGeo] = useState<THREE.BufferGeometry | null>(null)
  useEffect(() => {
    if (!url) { setGeo(null); return }
    const isObj = url.endsWith('.obj') || url.includes('.obj')
    if (isObj) {
      new OBJLoader().load(url, (group) => {
        const mesh = group.children.find(c => (c as THREE.Mesh).isMesh) as THREE.Mesh | undefined
        if (mesh) setGeo(mesh.geometry)
      })
    } else {
      new GLTFLoader().load(url, (gltf) => {
        gltf.scene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            setGeo((child as THREE.Mesh).geometry)
          }
        })
      })
    }
  }, [url])
  return geo
}

interface MaterialSphereProps {
  textureUrl: string | null
  normalMapUrl: string | null
  heightMapUrl: string | null
  roughnessMapUrl: string | null
  emissiveMapUrl: string | null
  aoMapUrl: string | null
  metallicMapUrl: string | null
  translucencyMapUrl: string | null
  subsurfaceMapUrl: string | null
  roughness: number
  metalness: number
  specularIntensity: number
  normalScale: number
  displacementScale: number
  emissiveIntensity: number
  aoIntensity: number
  transmission: number
  thickness: number
  subsurfaceColor: string
  autoRotate: boolean
  normalFlipR: boolean
  normalFlipG: boolean
  normalFlipB: boolean
  tileRepeat: number
  ignoreMetallicMap: boolean
  ignoreRoughnessMap: boolean
  invertRoughness: boolean
  ior?: number
  motion?: MotionState
}

function TexturedSphere({
  textureUrl,
  normalMapUrl,
  heightMapUrl,
  roughnessMapUrl,
  emissiveMapUrl,
  aoMapUrl,
  metallicMapUrl,
  translucencyMapUrl,
  subsurfaceMapUrl,
  roughness,
  metalness,
  specularIntensity,
  normalScale,
  displacementScale,
  emissiveIntensity,
  aoIntensity,
  transmission,
  thickness,
  subsurfaceColor,
  autoRotate,
  normalFlipR,
  normalFlipG,
  normalFlipB,
  tileRepeat,
  ignoreMetallicMap = false,
  ignoreRoughnessMap = false,
  invertRoughness = false,
  ior = 1.5,
  motion,
  geometry = 'sphere',
  customMeshUrl = null,
  meshRef: externalMeshRef,
  paintCanvases,
  paintVersionRef,
}: MaterialSphereProps & {
  geometry?: string
  customMeshUrl?: string | null
  meshRef?: React.RefObject<THREE.Mesh | null>
  paintCanvases?: React.RefObject<Map<string, HTMLCanvasElement> | null>
  paintVersionRef?: React.RefObject<number>
}) {
  const internalMeshRef = useRef<THREE.Mesh>(null)
  const meshRef = externalMeshRef || internalMeshRef
  const customGeometry = useCustomGeometry(geometry === 'custom' ? customMeshUrl : null)

  // AO requires uv2 — copy uv to uv2 whenever geometry changes
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh?.geometry) return
    const geo = mesh.geometry
    const uv = geo.getAttribute('uv')
    if (uv) {
      geo.setAttribute('uv2', uv.clone())
      geo.attributes.uv2.needsUpdate = true
    }
  }, [geometry, customGeometry, tileRepeat])

  // Albedo always loaded via drei (single URL — stable hook call)
  const albedo = useTexture(textureUrl!)
  albedo.wrapS = albedo.wrapT = THREE.RepeatWrapping
  albedo.repeat.set(tileRepeat, tileRepeat)

  // Optional PBR maps loaded safely (always same number of hook calls)
  const rawNormalMap = useOptionalTexture(normalMapUrl, tileRepeat)
  const displacementMap = useOptionalTexture(heightMapUrl, tileRepeat)
  const rawRoughnessMap = useOptionalTexture(roughnessMapUrl, tileRepeat)
  const roughnessMap = useInvertedTexture(rawRoughnessMap, invertRoughness)
  const emissiveMap = useOptionalTexture(emissiveMapUrl, tileRepeat)
  const aoMap = useOptionalTexture(aoMapUrl, tileRepeat)
  const metallicMap = useOptionalTexture(metallicMapUrl, tileRepeat)
  const translucencyMap = useOptionalTexture(translucencyMapUrl, tileRepeat)
  const subsurfaceMap = useOptionalTexture(subsurfaceMapUrl, tileRepeat)

  // Apply channel flips to the normal map
  const normalMap = useFlippedNormalMap(rawNormalMap, normalFlipR, normalFlipG, normalFlipB)

  // Live paint canvas textures (null when not painting)
  const canvasMap = paintCanvases?.current
  const paintAlbedo = usePaintCanvasTexture(canvasMap?.get('basecolor'), tileRepeat)
  const paintNormal = usePaintCanvasTexture(canvasMap?.get('normal'), tileRepeat)
  const paintRoughness = usePaintCanvasTexture(canvasMap?.get('roughness'), tileRepeat)
  const paintMetallic = usePaintCanvasTexture(canvasMap?.get('metalness'), tileRepeat)
  const paintHeight = usePaintCanvasTexture(canvasMap?.get('height'), tileRepeat)
  const paintAo = usePaintCanvasTexture(canvasMap?.get('ao'), tileRepeat)
  const paintEmissive = usePaintCanvasTexture(canvasMap?.get('emissive'), tileRepeat)

  const lastPaintVersion = useRef(0)
  const materialRef = useRef<THREE.MeshPhysicalMaterial>(null)
  const timeRef = useRef(0)

  useFrame((_, delta) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y += delta * 0.3
    }
    // Live paint: mark canvas textures dirty when paint version changes
    if (paintVersionRef && paintVersionRef.current !== lastPaintVersion.current) {
      lastPaintVersion.current = paintVersionRef.current
      const textures = [paintAlbedo, paintNormal, paintRoughness, paintMetallic, paintHeight, paintAo, paintEmissive]
      for (const t of textures) {
        if (t) t.needsUpdate = true
      }
    }

    // Motion / animation — always write authoritative values each frame so
    // clearing a motion config reverts to the static UI value instead of
    // leaving the last animated value stuck on the material.
    timeRef.current += delta
    const t = timeRef.current
    const mat = materialRef.current
    const m = motion ?? {}

    const px = evaluateMotion(m.pan_x, t, 0)
    const py = evaluateMotion(m.pan_y, t, 0)
    const allTex = [
      finalAlbedo, finalNormalMap, finalRoughnessMap, finalMetallicMap,
      finalDisplacementMap, finalAoMap, finalEmissiveMap,
      translucencyMap, subsurfaceMap,
    ]
    for (const tex of allTex) {
      if (tex) tex.offset.set(px, py)
    }

    if (mat) {
      mat.displacementScale = evaluateMotion(m.displacement, t, displacementScale)
      mat.displacementBias = -mat.displacementScale * 0.5
      mat.transmission = evaluateMotion(m.transmission, t, transmission)
      mat.ior = evaluateMotion(m.ior, t, ior)
      mat.emissiveIntensity = evaluateMotion(m.emissive, t, emissiveIntensity)
      mat.roughness = evaluateMotion(m.roughness, t, roughness)
      mat.metalness = evaluateMotion(m.metalness, t, metalness)
    }
  })

  // Use paint canvas textures when available, fall back to URL-loaded
  const finalAlbedo = paintAlbedo || albedo
  const finalNormalMap = paintNormal || normalMap
  const finalRoughnessMap = paintRoughness || roughnessMap
  const finalMetallicMap = paintMetallic || metallicMap
  const finalDisplacementMap = paintHeight || displacementMap
  const finalAoMap = paintAo || aoMap
  const finalEmissiveMap = paintEmissive || emissiveMap

  // Force material shader recompile when the set of available maps changes
  const materialKey = `mat-${!!finalNormalMap}-${!!finalDisplacementMap}-${!!finalRoughnessMap}-${!!finalEmissiveMap}-${!!finalAoMap}-${!!finalMetallicMap}-${!!translucencyMap}-${!!subsurfaceMap}-${tileRepeat}-${ignoreMetallicMap}-${ignoreRoughnessMap}-${invertRoughness}-${metallicMapUrl?.substring(0, 30) ?? 'none'}-${!!paintAlbedo}`

  const normalScaleVec = useMemo(() => new THREE.Vector2(normalScale, normalScale), [normalScale])
  const attenuationColor = useMemo(() => new THREE.Color(subsurfaceColor), [subsurfaceColor])

  const mat = (
    <meshPhysicalMaterial
      ref={materialRef}
      key={materialKey}
      map={finalAlbedo}
      normalMap={finalNormalMap ?? undefined}
      normalScale={normalScaleVec}
      displacementMap={finalDisplacementMap ?? undefined}
      displacementScale={displacementScale}
      displacementBias={-displacementScale * 0.5}
      roughnessMap={ignoreRoughnessMap ? undefined : (finalRoughnessMap ?? undefined)}
      roughness={roughness}
      metalness={metalness}
      metalnessMap={ignoreMetallicMap ? undefined : (finalMetallicMap ?? undefined)}
      specularIntensity={specularIntensity}
      specularColor={new THREE.Color(1, 1, 1)}
      aoMap={finalAoMap ?? undefined}
      aoMapIntensity={aoIntensity}
      emissiveMap={finalEmissiveMap ?? undefined}
      emissive={new THREE.Color(1, 1, 1)}
      emissiveIntensity={emissiveIntensity}
      transmission={transmission}
      transmissionMap={translucencyMap ?? undefined}
      thickness={thickness}
      thicknessMap={subsurfaceMap ?? undefined}
      attenuationColor={attenuationColor}
      attenuationDistance={thickness > 0 ? 2.0 : 0}
      ior={ior}
    />
  )

  return (
    <mesh ref={meshRef}>
      {geometry === 'sphere' && <sphereGeometry args={[1.5, 512, 512]} />}
      {geometry === 'plane' && <planeGeometry args={[3, 3, 128, 128]} />}
      {geometry === 'cube' && <boxGeometry args={[2, 2, 2, 64, 64, 64]} />}
      {geometry === 'custom' && customGeometry && <primitive object={customGeometry} attach="geometry" />}
      {mat}
    </mesh>
  )
}

// Moved useCustomGeometry hook above TexturedSphere — defined before interface block

function PlaceholderMesh({ roughness, metalness, autoRotate, geometry }: {
  roughness: number; metalness: number; autoRotate: boolean; geometry: string
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((_, delta) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y += delta * 0.3
    }
  })

  return (
    <mesh ref={meshRef}>
      {geometry === 'plane' ? <planeGeometry args={[3, 3]} /> :
       geometry === 'cube' ? <boxGeometry args={[2, 2, 2]} /> :
       <sphereGeometry args={[1.5, 64, 64]} />}
      <meshStandardMaterial color="#444444" roughness={roughness} metalness={metalness} />
    </mesh>
  )
}

function MaterialSphere(props: MaterialSphereProps & { geometry: string; customMeshUrl: string | null; meshRef?: React.RefObject<THREE.Mesh | null>; paintCanvases?: React.RefObject<Map<string, HTMLCanvasElement> | null>; paintVersionRef?: React.RefObject<number> }) {
  if (!props.textureUrl) {
    return <PlaceholderMesh roughness={props.roughness} metalness={props.metalness} autoRotate={props.autoRotate} geometry={props.geometry} />
  }

  return (
    <Suspense fallback={<PlaceholderMesh roughness={props.roughness} metalness={props.metalness} autoRotate={props.autoRotate} geometry={props.geometry} />}>
      <TexturedSphere {...props} />
    </Suspense>
  )
}

// --- Paint handler: raycasts pointer events onto the sphere to get UV coordinates ---

function PaintHandler({
  meshRef,
  enabled,
  tileRepeat,
  onDown,
  onMove,
  onUp,
}: {
  meshRef: React.RefObject<THREE.Mesh | null>
  enabled: boolean
  tileRepeat: number
  onDown?: (x: number, y: number) => void
  onMove?: (x: number, y: number) => void
  onUp?: () => void
}) {
  const { gl, camera } = useThree()
  const raycasterRef = useRef(new THREE.Raycaster())
  const mouseRef = useRef(new THREE.Vector2())
  const paintingRef = useRef(false)

  const getPixelFromEvent = useCallback((e: PointerEvent): { x: number; y: number } | null => {
    const rect = gl.domElement.getBoundingClientRect()
    mouseRef.current.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycasterRef.current.setFromCamera(mouseRef.current, camera)
    const mesh = meshRef.current
    if (!mesh) return null
    const intersects = raycasterRef.current.intersectObject(mesh)
    if (intersects.length > 0 && intersects[0].uv) {
      const uv = intersects[0].uv
      const x = ((uv.x * tileRepeat) % 1) * 512
      const y = (1 - ((uv.y * tileRepeat) % 1)) * 512
      return { x, y }
    }
    return null
  }, [gl, camera, meshRef, tileRepeat])

  useEffect(() => {
    if (!enabled) return
    const el = gl.domElement

    const handleDown = (e: PointerEvent) => {
      const pos = getPixelFromEvent(e)
      if (pos) {
        paintingRef.current = true
        onDown?.(pos.x, pos.y)
      }
    }

    const handleMove = (e: PointerEvent) => {
      if (!paintingRef.current) return
      const pos = getPixelFromEvent(e)
      if (pos) onMove?.(pos.x, pos.y)
    }

    const handleUp = () => {
      if (paintingRef.current) {
        paintingRef.current = false
        onUp?.()
      }
    }

    el.addEventListener('pointerdown', handleDown)
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)

    return () => {
      el.removeEventListener('pointerdown', handleDown)
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
  }, [enabled, gl, getPixelFromEvent, onDown, onMove, onUp])

  return null
}

interface MaterialPreviewProps extends MaterialSphereProps {
  environment: string
  geometry: string
  customMeshUrl: string | null
  envIntensity: number
  keyLightIntensity: number
  fillLightIntensity: number
  rimLightIntensity: number
  specularIntensity: number
  paintMode?: boolean
  onPaintDown?: (x: number, y: number) => void
  onPaintMove?: (x: number, y: number) => void
  onPaintUp?: () => void
  paintCanvases?: React.RefObject<Map<string, HTMLCanvasElement> | null>
  paintVersionRef?: React.RefObject<number>
}

export default function MaterialPreview({
  environment,
  envIntensity,
  keyLightIntensity,
  fillLightIntensity,
  rimLightIntensity,
  paintMode = false,
  onPaintDown,
  onPaintMove,
  onPaintUp,
  paintCanvases,
  paintVersionRef,
  ...sphereProps
}: MaterialPreviewProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  return (
    <div
      className="preview-container"
      style={{
        width: '100%', height: '100%', background: '#1a1a1a', borderRadius: '8px',
        cursor: paintMode ? 'crosshair' : undefined,
      }}
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
        <ambientLight intensity={0.05} />
        <directionalLight position={[5, 5, 5]} intensity={keyLightIntensity} />
        <directionalLight position={[-3, 2, -2]} intensity={fillLightIntensity} />
        <directionalLight position={[0, -3, 4]} intensity={rimLightIntensity} />
        <MaterialSphere {...sphereProps} meshRef={meshRef} paintCanvases={paintCanvases} paintVersionRef={paintVersionRef} />
        <OrbitControls enablePan={false} enabled={!paintMode} />
        <PaintHandler
          meshRef={meshRef}
          enabled={paintMode}
          tileRepeat={sphereProps.tileRepeat}
          onDown={onPaintDown}
          onMove={onPaintMove}
          onUp={onPaintUp}
        />
        <Suspense fallback={null}>
          <Environment preset={environment as any} background={false} environmentIntensity={envIntensity} />
        </Suspense>
      </Canvas>
    </div>
  )
}
