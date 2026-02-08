import { useRef, useEffect, useState, useMemo, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, useTexture } from '@react-three/drei'
import * as THREE from 'three'

/**
 * Safely load an optional texture via THREE.TextureLoader.
 * Always calls the same hooks regardless of whether url is null,
 * avoiding the conditional-hook bug with drei's useTexture.
 */
function useOptionalTexture(url: string | null): THREE.Texture | null {
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
      tex.repeat.set(2, 2)
      setTexture((prev) => {
        prev?.dispose()
        return tex
      })
    })
    return () => {
      cancelled = true
    }
  }, [url])

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

interface MaterialSphereProps {
  textureUrl: string | null
  normalMapUrl: string | null
  heightMapUrl: string | null
  roughnessMapUrl: string | null
  emissiveMapUrl: string | null
  roughness: number
  metalness: number
  normalScale: number
  displacementScale: number
  emissiveIntensity: number
  autoRotate: boolean
  normalFlipR: boolean
  normalFlipG: boolean
  normalFlipB: boolean
}

function TexturedSphere({
  textureUrl,
  normalMapUrl,
  heightMapUrl,
  roughnessMapUrl,
  emissiveMapUrl,
  roughness,
  metalness,
  normalScale,
  displacementScale,
  emissiveIntensity,
  autoRotate,
  normalFlipR,
  normalFlipG,
  normalFlipB,
}: MaterialSphereProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  // Albedo always loaded via drei (single URL — stable hook call)
  const albedo = useTexture(textureUrl!)
  albedo.wrapS = albedo.wrapT = THREE.RepeatWrapping
  albedo.repeat.set(2, 2)

  // Optional PBR maps loaded safely (always same number of hook calls)
  const rawNormalMap = useOptionalTexture(normalMapUrl)
  const displacementMap = useOptionalTexture(heightMapUrl)
  const roughnessMap = useOptionalTexture(roughnessMapUrl)
  const emissiveMap = useOptionalTexture(emissiveMapUrl)

  // Apply channel flips to the normal map
  const normalMap = useFlippedNormalMap(rawNormalMap, normalFlipR, normalFlipG, normalFlipB)

  useFrame((_, delta) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y += delta * 0.3
    }
  })

  // Force material shader recompile when the set of available maps changes
  const materialKey = `mat-${!!normalMap}-${!!displacementMap}-${!!roughnessMap}-${!!emissiveMap}`

  const normalScaleVec = useMemo(() => new THREE.Vector2(normalScale, normalScale), [normalScale])

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.5, 128, 128]} />
      <meshPhysicalMaterial
        key={materialKey}
        map={albedo}
        normalMap={normalMap ?? undefined}
        normalScale={normalScaleVec}
        displacementMap={displacementMap ?? undefined}
        displacementScale={displacementScale}
        displacementBias={-displacementScale * 0.5}
        roughnessMap={roughnessMap ?? undefined}
        roughness={roughness}
        metalness={metalness}
        emissiveMap={emissiveMap ?? undefined}
        emissive={new THREE.Color(1, 1, 1)}
        emissiveIntensity={emissiveIntensity}
      />
    </mesh>
  )
}

function PlaceholderSphere({ roughness, metalness, autoRotate }: { roughness: number; metalness: number; autoRotate: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((_, delta) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y += delta * 0.3
    }
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.5, 64, 64]} />
      <meshStandardMaterial
        color="#444444"
        roughness={roughness}
        metalness={metalness}
      />
    </mesh>
  )
}

function MaterialSphere(props: MaterialSphereProps) {
  if (!props.textureUrl) {
    return <PlaceholderSphere roughness={props.roughness} metalness={props.metalness} autoRotate={props.autoRotate} />
  }

  return (
    <Suspense fallback={<PlaceholderSphere roughness={props.roughness} metalness={props.metalness} autoRotate={props.autoRotate} />}>
      <TexturedSphere {...props} />
    </Suspense>
  )
}

interface MaterialPreviewProps extends MaterialSphereProps {
  environment: string
}

export default function MaterialPreview({
  environment,
  ...sphereProps
}: MaterialPreviewProps) {
  return (
    <div className="preview-container" style={{ width: '100%', height: '100%', background: '#1a1a1a', borderRadius: '8px' }}>
      <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
        <ambientLight intensity={0.15} />
        <directionalLight position={[5, 5, 5]} intensity={1.8} />
        <directionalLight position={[-3, 2, -2]} intensity={0.4} />
        <directionalLight position={[0, -3, 4]} intensity={0.25} />
        <MaterialSphere {...sphereProps} />
        <OrbitControls enablePan={false} />
        <Suspense fallback={null}>
          <Environment preset={environment as any} background={false} />
        </Suspense>
      </Canvas>
    </div>
  )
}
