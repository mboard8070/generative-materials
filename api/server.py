#!/usr/bin/env python3
"""FastAPI backend for Text-to-Material generation (multi-map grid)."""

import os
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Optional
import threading
import time
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from peft import PeftModel
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# Allow importing from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from grid_utils import (
    split_grid, compose_grid,
    GRID_WIDTH, GRID_HEIGHT, GRID_CELL_SIZE,
    GRAYSCALE_MAPS, NEUTRAL_FILLS,
)

# Paths
OUTPUT_DIR = PROJECT_ROOT / "api" / "outputs"
LIBRARY_DIR = PROJECT_ROOT / "api" / "library"
LAYERS_FILE = PROJECT_ROOT / "api" / "layers.json"
MOTION_FILE = PROJECT_ROOT / "api" / "motion.json"
LORA_PATH = PROJECT_ROOT / "output" / "loras" / "pbr-materials-multimap-v1" / "final_lora"
MERGED_MODEL_PATH = PROJECT_ROOT / "output" / "merged_model_multimap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Surfaced API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated textures
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/library-files", StaticFiles(directory=str(LIBRARY_DIR)), name="library_files")

# Global pipelines and loading state
pipe = None
pipe_img2img = None
loading_state = {
    "status": "idle",  # idle, loading, ready, error
    "step": "",
    "progress": 0,
    "error": None,
    "start_time": None,
}

class GenerateRequest(BaseModel):
    prompt: str
    tiling_mode: str = "multiscale"
    seed: Optional[int] = None
    engine: str = "flux"  # "flux" or "patina"
    upscale_factor: int = 0  # 0 (none), 2, or 4 — PATINA only
    enable_prompt_expansion: bool = True  # PATINA only
    maps: Optional[list[str]] = None  # e.g. ["basecolor","normal","roughness"]

class ImageToPbrRequest(BaseModel):
    image_url: str
    maps: Optional[list[str]] = None
    seed: Optional[int] = None

class ExtractMaterialRequest(BaseModel):
    image_url: str
    label: str
    upscale_factor: int = 0
    maps: Optional[list[str]] = None
    seed: Optional[int] = None

class AdjustHeightRequest(BaseModel):
    texture_id: str
    contrast: float = 1.0  # multiplier around midpoint
    brightness: float = 0.0  # -1.0 to 1.0 shift
    invert: bool = False
    blur_radius: float = 0.0  # gaussian blur sigma

class EditRequest(BaseModel):
    image_url: str
    prompt: str
    strength: float = 0.7
    tiling_mode: str = "multiscale"
    seed: Optional[int] = None

class CompositePbrRequest(BaseModel):
    image_data: str  # data:image/png;base64,...

class ExportRequest(BaseModel):
    texture_url: str
    roughness: float = 0.5
    metalness: float = 0.0
    format: str = "unreal"

class SaveMaterialRequest(BaseModel):
    texture_id: str
    name: str
    prompt: str = ""
    engine: str = ""
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Layer stack models — mirrors ui/src/types/layers.ts
# ---------------------------------------------------------------------------

class ChannelSettings(BaseModel):
    enabled: bool = True


class LayerMask(BaseModel):
    enabled: bool = False
    noiseType: str = "perlin"  # 'perlin' | 'voronoi'
    scale: float = 4
    seed: int = 0
    invert: bool = False


class PbrMaps(BaseModel):
    basecolor: Optional[str] = None
    normal: Optional[str] = None
    roughness: Optional[str] = None
    metalness: Optional[str] = None
    height: Optional[str] = None
    translucency: Optional[str] = None
    subsurface: Optional[str] = None


class Layer(BaseModel):
    id: str
    name: str
    visible: bool = True
    opacity: float = 1.0
    blendMode: str = "normal"
    type: str = "material"  # 'material' | 'solid' | 'noise'

    materialPrompt: str = ""
    materialMaps: dict = {}
    generating: bool = False

    channels: dict = {}

    color: str = "#888888"

    noiseType: str = "perlin"
    noiseScale: float = 4
    noiseSeed: int = 0
    noiseColor1: str = "#000000"
    noiseColor2: str = "#ffffff"

    mask: dict = {}


class ReorderRequest(BaseModel):
    order: list[str]  # list of layer IDs in desired order


# ---------------------------------------------------------------------------
# Emissive map — still derived from albedo (threshold-based)
# ---------------------------------------------------------------------------

def generate_emissive_map(image: Image.Image) -> Image.Image:
    """Emissive map — only very bright albedo areas emit."""
    gray = np.array(image.convert('L')).astype(np.float32) / 255.0
    emissive = np.clip((gray - 0.85) / 0.15, 0.0, 1.0)
    return Image.fromarray((emissive * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# PBR map normalization
# ---------------------------------------------------------------------------
# Flux/PATINA tend to output dark roughness maps and noisy metallic maps.
# In Three.js meshPhysicalMaterial the roughness slider multiplies the map,
# so dark maps lock the surface into reflective no matter the slider. These
# helpers rebalance both so the slider behaves as users expect.

METAL_KEYWORDS = frozenset({
    "metal", "metals", "metallic", "metalness",
    "steel", "iron", "copper", "brass", "bronze", "gold", "silver",
    "chrome", "chromed", "aluminum", "aluminium", "tin", "lead",
    "nickel", "titanium", "platinum", "zinc", "pewter", "alloy",
    "rust", "rusted", "rusty", "corroded", "oxidized", "patina",
    "anodized", "galvanized", "plated", "gilded", "foil",
})


def _prompt_has_metal(prompt: str) -> bool:
    if not prompt:
        return False
    lowered = prompt.lower()
    tokens = set(lowered.replace("-", " ").replace("_", " ").split())
    return bool(tokens & METAL_KEYWORDS)


def normalize_roughness_map(img: Image.Image, min_mean: float = 0.55) -> Image.Image:
    """Lift roughness if its mean is below min_mean (too reflective).

    Uses a gamma that maps current_mean -> min_mean so bright areas stay
    relatively bright and dark areas get lifted more. If the map is already
    bright enough, return unchanged.
    """
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32) / 255.0
    current_mean = float(arr.mean())
    if current_mean >= min_mean:
        return gray
    if current_mean < 1e-3:
        return Image.new("L", gray.size, int(min_mean * 255))
    gamma = float(np.log(min_mean) / np.log(current_mean))
    arr = np.clip(np.power(arr, gamma), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def zero_metallic_map(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def normalize_pbr_maps(maps: dict, prompt: str) -> dict:
    """Fix roughness brightness and kill spurious metallic for non-metal prompts."""
    result = dict(maps)
    rough = result.get("roughness")
    if rough is not None:
        result["roughness"] = normalize_roughness_map(rough)
    metal = result.get("metallic")
    if metal is not None and not _prompt_has_metal(prompt):
        result["metallic"] = zero_metallic_map(metal.size)
    return result


def postprocess_pbr_files(texture_id: str, prompt: str) -> None:
    """Rewrite saved roughness/metallic PNGs on disk for prompt-based paths."""
    rough_path = OUTPUT_DIR / f"{texture_id}_roughness.png"
    if rough_path.exists():
        normalize_roughness_map(Image.open(rough_path)).save(rough_path)
    metal_path = OUTPUT_DIR / f"{texture_id}_metallic.png"
    if metal_path.exists() and not _prompt_has_metal(prompt):
        zero_metallic_map(Image.open(metal_path).size).save(metal_path)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _load_tiling_functions():
    """Lazily load tiling post-process functions."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from tile_postprocess import make_tileable, make_tileable_multipass, make_tileable_multiscale
    return make_tileable, make_tileable_multipass, make_tileable_multiscale


def apply_tiling_to_maps(maps: dict, tiling_mode: str) -> dict:
    """Apply tiling post-processing to each map independently.

    Grayscale (mode 'L') maps are temporarily converted to RGB for tiling,
    then converted back to L.
    """
    if tiling_mode == "none":
        return maps

    make_tileable, make_tileable_multipass, make_tileable_multiscale = _load_tiling_functions()

    tiled_maps = {}
    for name, img in maps.items():
        was_grayscale = img.mode == "L"
        if was_grayscale:
            img = img.convert("RGB")

        if tiling_mode == "basic":
            img = make_tileable(img)
        elif tiling_mode == "multipass":
            img = make_tileable_multipass(img)
        elif tiling_mode == "multiscale":
            img = make_tileable_multiscale(img)

        if was_grayscale:
            img = img.convert("L")

        tiled_maps[name] = img

    return tiled_maps


# ---------------------------------------------------------------------------
# Save split maps
# ---------------------------------------------------------------------------

def save_split_maps(maps: dict, texture_id: str) -> dict:
    """Save each PBR map as an individual PNG and return URL dict.

    Also generates emissive map from the albedo.
    """
    urls = {}

    # Save albedo
    albedo = maps["albedo"]
    albedo_path = OUTPUT_DIR / f"{texture_id}.png"
    albedo.save(albedo_path)
    urls["texture_url"] = f"/outputs/{texture_id}.png"

    # Save AI-generated PBR maps
    for name in ("normal", "roughness", "height", "metallic", "ao"):
        img = maps.get(name)
        if img is not None:
            path = OUTPUT_DIR / f"{texture_id}_{name}.png"
            img.save(path)
            urls[f"{name}_map_url"] = f"/outputs/{texture_id}_{name}.png"

    # Generate and save emissive (still threshold-based from albedo)
    emissive = generate_emissive_map(albedo)
    emissive_path = OUTPUT_DIR / f"{texture_id}_emissive.png"
    emissive.save(emissive_path)
    urls["emissive_map_url"] = f"/outputs/{texture_id}_emissive.png"

    return urls


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def update_status(status: str, step: str = "", progress: int = 0, error: str = None):
    """Update loading status."""
    loading_state["status"] = status
    loading_state["step"] = step
    loading_state["progress"] = progress
    if error:
        loading_state["error"] = error
    print(f"[{status}] {step} ({progress}%)")

def load_pipeline():
    """Load FLUX pipeline with merged LoRA weights."""
    global pipe
    if pipe is not None:
        return pipe

    loading_state["start_time"] = time.time()

    try:
        merged_transformer_path = MERGED_MODEL_PATH / "transformer"

        if merged_transformer_path.exists():
            update_status("loading", "Loading pre-merged model...", 10)
            from diffusers import FluxTransformer2DModel

            transformer = FluxTransformer2DModel.from_pretrained(
                str(merged_transformer_path),
                torch_dtype=torch.bfloat16,
            )
            update_status("loading", "Loading pipeline components...", 50)

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            update_status("loading", "Loading base FLUX model...", 10)
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
            )

            if LORA_PATH.exists():
                update_status("loading", "Applying LoRA adapter...", 40)
                pipe.transformer = PeftModel.from_pretrained(
                    pipe.transformer,
                    str(LORA_PATH),
                )

                update_status("loading", "Merging LoRA into weights (one-time)...", 60)
                pipe.transformer = pipe.transformer.merge_and_unload()

                update_status("loading", "Saving merged model for faster startup...", 70)
                MERGED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                pipe.transformer.save_pretrained(str(merged_transformer_path))
                update_status("loading", "Merged model saved!", 75)

        update_status("loading", "Moving to GPU...", 80)
        pipe.to("cuda")

        elapsed = time.time() - loading_state["start_time"]
        update_status("ready", f"Pipeline ready! (loaded in {elapsed:.1f}s)", 100)

        return pipe

    except Exception as e:
        update_status("error", str(e), 0, str(e))
        raise

def load_img2img_pipeline():
    """Load FLUX img2img pipeline by reusing components from txt2img pipeline."""
    global pipe_img2img, pipe
    if pipe_img2img is not None:
        return pipe_img2img

    if pipe is None:
        load_pipeline()

    update_status("loading", "Creating img2img pipeline...", 90)
    pipe_img2img = FluxImg2ImgPipeline(
        transformer=pipe.transformer,
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
    )
    update_status("ready", "Img2Img pipeline ready!", 100)
    return pipe_img2img


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Pre-load the pipeline on startup."""
    def preload():
        try:
            load_pipeline()
        except Exception as e:
            print(f"Preload failed: {e}")
    threading.Thread(target=preload, daemon=True).start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/status")
async def status():
    """Check pipeline loading status."""
    patina_ready = bool(os.environ.get("FAL_KEY"))
    return {
        "pipeline_loaded": pipe is not None and loading_state["status"] == "ready",
        "img2img_loaded": pipe_img2img is not None,
        "ready": loading_state["status"] == "ready",
        "patina_ready": patina_ready,
        "loading": loading_state["status"] == "loading",
        "step": loading_state["step"],
        "progress": loading_state["progress"],
        "error": loading_state["error"],
        "elapsed": time.time() - loading_state["start_time"] if loading_state["start_time"] else 0,
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate a material texture from prompt."""
    if request.engine == "patina":
        return await generate_patina(request)

    if loading_state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline loading: {loading_state['step']}")

    pipeline = load_pipeline()

    texture_id = str(uuid.uuid4())[:8]

    seed = request.seed or torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    # Prepend multi-map trigger word
    full_prompt = f"pbr_multimap {request.prompt}"
    print(f"Generating (Flux): {full_prompt}")

    grid_image = pipeline(
        prompt=full_prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        generator=generator,
    ).images[0]

    # Split grid into individual maps
    maps = split_grid(grid_image)

    # Rebalance roughness / kill spurious metallic for non-metal prompts
    maps = normalize_pbr_maps(maps, request.prompt)

    # Apply tiling to each map independently
    maps = apply_tiling_to_maps(maps, request.tiling_mode)

    # Save all maps and get URLs
    urls = save_split_maps(maps, texture_id)

    return {
        **urls,
        "seed": seed,
        "texture_id": texture_id,
        "engine": "flux",
    }


# ---------------------------------------------------------------------------
# PATINA helpers
# ---------------------------------------------------------------------------

PATINA_MAP_KEYS = {
    "basecolor": ("texture_url", "{tid}.png"),
    "normal": ("normal_map_url", "{tid}_normal.png"),
    "roughness": ("roughness_map_url", "{tid}_roughness.png"),
    "metalness": ("metallic_map_url", "{tid}_metallic.png"),
    "height": ("height_map_url", "{tid}_height.png"),
}


def _require_fal():
    import fal_client
    import httpx as hx
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=500, detail="FAL_KEY not set")
    return fal_client, hx


def _download_patina_maps(result: dict, texture_id: str, hx) -> dict:
    """Download PATINA result images and return URL dict."""
    urls = {}
    for img in result.get("images", []):
        map_type = img.get("map_type")
        if not map_type or map_type not in PATINA_MAP_KEYS:
            continue
        url_key, fname_tpl = PATINA_MAP_KEYS[map_type]
        filename = fname_tpl.format(tid=texture_id)
        dest = OUTPUT_DIR / filename
        with hx.stream("GET", img["url"], follow_redirects=True, timeout=60.0) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(8192):
                    f.write(chunk)
        urls[url_key] = f"/outputs/{filename}"

    # Generate emissive from basecolor
    albedo_path = OUTPUT_DIR / f"{texture_id}.png"
    if albedo_path.exists():
        albedo = Image.open(albedo_path)
        emissive = generate_emissive_map(albedo)
        emissive_path = OUTPUT_DIR / f"{texture_id}_emissive.png"
        emissive.save(emissive_path)
        urls["emissive_map_url"] = f"/outputs/{texture_id}_emissive.png"

    return urls


async def generate_patina(request: GenerateRequest):
    """Generate a material using fal.ai PATINA /material endpoint."""
    fal_client, hx = _require_fal()
    texture_id = str(uuid.uuid4())[:8]
    print(f"Generating (PATINA): {request.prompt}")

    args = {
        "prompt": request.prompt,
        "output_format": "png",
        "enable_prompt_expansion": request.enable_prompt_expansion,
    }
    if request.seed is not None:
        args["seed"] = request.seed
    if request.upscale_factor in (2, 4):
        args["upscale_factor"] = request.upscale_factor
    if request.maps:
        args["maps"] = request.maps

    try:
        result = fal_client.subscribe("fal-ai/patina/material", arguments=args, with_logs=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PATINA API error: {e}")

    urls = _download_patina_maps(result, texture_id, hx)
    postprocess_pbr_files(texture_id, request.prompt)
    return {
        **urls,
        "seed": result.get("seed", 0),
        "texture_id": texture_id,
        "engine": "patina",
    }


# ---------------------------------------------------------------------------
# Image-to-PBR (PATINA base endpoint)
# ---------------------------------------------------------------------------

@app.post("/image-to-pbr")
async def image_to_pbr(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    maps: str = Form(None),
    seed: int = Form(None),
):
    """Upload an image (or pass a URL) and get PBR maps from PATINA."""
    fal_client, hx = _require_fal()
    texture_id = str(uuid.uuid4())[:8]

    # Handle file upload — save locally and upload to fal
    if image and image.filename:
        upload_path = OUTPUT_DIR / f"upload_{texture_id}.png"
        content = await image.read()
        upload_path.write_bytes(content)
        fal_url = fal_client.upload_file(str(upload_path))
    elif image_url:
        fal_url = image_url
    else:
        raise HTTPException(status_code=400, detail="Provide image file or image_url")

    print(f"Image-to-PBR (PATINA): {fal_url}")

    args = {"image_url": fal_url, "output_format": "png"}
    if seed is not None:
        args["seed"] = seed
    if maps:
        args["maps"] = [m.strip() for m in maps.split(",")]

    try:
        result = fal_client.subscribe("fal-ai/patina", arguments=args, with_logs=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PATINA API error: {e}")

    urls = _download_patina_maps(result, texture_id, hx)
    rough_path = OUTPUT_DIR / f"{texture_id}_roughness.png"
    if rough_path.exists():
        normalize_roughness_map(Image.open(rough_path)).save(rough_path)
    return {
        **urls,
        "seed": result.get("seed", 0),
        "texture_id": texture_id,
        "engine": "patina",
    }


# ---------------------------------------------------------------------------
# Material extraction (PATINA /material/extract)
# ---------------------------------------------------------------------------

@app.post("/extract-material")
async def extract_material(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    label: str = Form(...),
    upscale_factor: int = Form(0),
    maps: str = Form(None),
    seed: int = Form(None),
):
    """Extract a specific material from a scene photo."""
    fal_client, hx = _require_fal()
    texture_id = str(uuid.uuid4())[:8]

    if image and image.filename:
        upload_path = OUTPUT_DIR / f"upload_{texture_id}.png"
        content = await image.read()
        upload_path.write_bytes(content)
        fal_url = fal_client.upload_file(str(upload_path))
    elif image_url:
        fal_url = image_url
    else:
        raise HTTPException(status_code=400, detail="Provide image file or image_url")

    print(f"Extract material (PATINA): '{label}' from {fal_url}")

    args = {
        "image_url": fal_url,
        "prompt": label,
        "output_format": "png",
    }
    if seed is not None:
        args["seed"] = seed
    if upscale_factor in (2, 4):
        args["upscale_factor"] = upscale_factor
    if maps:
        args["maps"] = [m.strip() for m in maps.split(",")]

    try:
        result = fal_client.subscribe("fal-ai/patina/material/extract", arguments=args, with_logs=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PATINA API error: {e}")

    urls = _download_patina_maps(result, texture_id, hx)
    postprocess_pbr_files(texture_id, label)
    return {
        **urls,
        "seed": result.get("seed", 0),
        "texture_id": texture_id,
        "engine": "patina",
    }


# ---------------------------------------------------------------------------
# Height map adjustment
# ---------------------------------------------------------------------------

@app.post("/adjust-height")
async def adjust_height(request: AdjustHeightRequest):
    """Post-process the height map: contrast, brightness, invert, blur."""
    height_path = OUTPUT_DIR / f"{request.texture_id}_height.png"
    if not height_path.exists():
        raise HTTPException(status_code=404, detail="Height map not found")

    img = Image.open(height_path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0

    # Invert
    if request.invert:
        arr = 1.0 - arr

    # Contrast (expand around 0.5 midpoint)
    if request.contrast != 1.0:
        arr = (arr - 0.5) * request.contrast + 0.5

    # Brightness shift
    if request.brightness != 0.0:
        arr = arr + request.brightness

    # Clamp
    arr = np.clip(arr, 0.0, 1.0)

    # Gaussian blur
    if request.blur_radius > 0:
        arr = gaussian_filter(arr, sigma=request.blur_radius)

    adjusted = Image.fromarray((arr * 255).astype(np.uint8))
    out_filename = f"{request.texture_id}_height_adjusted.png"
    out_path = OUTPUT_DIR / out_filename
    adjusted.save(out_path)

    return {"height_map_url": f"/outputs/{out_filename}"}


# ---------------------------------------------------------------------------
# Download all maps as ZIP
# ---------------------------------------------------------------------------

@app.get("/download-all/{texture_id}")
async def download_all(texture_id: str):
    """Download all PBR maps for a texture as a ZIP."""
    import shutil

    # Find all files for this texture_id
    files = list(OUTPUT_DIR.glob(f"{texture_id}*.png"))
    if not files:
        raise HTTPException(status_code=404, detail="No maps found")

    zip_path = OUTPUT_DIR / f"{texture_id}_all.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.write(f, f.name)

    return FileResponse(zip_path, media_type="application/zip", filename=f"{texture_id}_maps.zip")


# ---------------------------------------------------------------------------
# Material Library
# ---------------------------------------------------------------------------

@app.post("/library/save")
async def save_material(request: SaveMaterialRequest):
    """Save a generated material to the library."""
    import json
    import shutil

    mat_id = str(uuid.uuid4())[:8]
    mat_dir = LIBRARY_DIR / mat_id
    mat_dir.mkdir(parents=True, exist_ok=True)

    # Copy all map files for this texture_id
    map_names = {
        "basecolor": f"{request.texture_id}.png",
        "normal": f"{request.texture_id}_normal.png",
        "roughness": f"{request.texture_id}_roughness.png",
        "metallic": f"{request.texture_id}_metallic.png",
        "height": f"{request.texture_id}_height.png",
        "emissive": f"{request.texture_id}_emissive.png",
    }

    saved_maps = {}
    for map_type, filename in map_names.items():
        src = OUTPUT_DIR / filename
        if src.exists():
            dest = mat_dir / f"{map_type}.png"
            shutil.copy2(src, dest)
            saved_maps[map_type] = f"{map_type}.png"

    if not saved_maps:
        mat_dir.rmdir()
        raise HTTPException(status_code=404, detail="No maps found for this texture_id")

    # Write metadata
    metadata = {
        "id": mat_id,
        "name": request.name,
        "prompt": request.prompt,
        "engine": request.engine,
        "seed": request.seed,
        "maps": saved_maps,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (mat_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata


@app.get("/library")
async def list_library():
    """List all saved materials."""
    import json

    materials = []
    for mat_dir in sorted(LIBRARY_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not mat_dir.is_dir():
            continue
        meta_path = mat_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # Add thumbnail URL
            if (mat_dir / "basecolor.png").exists():
                meta["thumbnail_url"] = f"/library-files/{mat_dir.name}/basecolor.png"
            materials.append(meta)

    return materials


@app.get("/library/{material_id}")
async def load_material(material_id: str):
    """Load a saved material — returns map URLs."""
    import json

    mat_dir = LIBRARY_DIR / material_id
    meta_path = mat_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Material not found")

    meta = json.loads(meta_path.read_text())

    # Copy maps back to outputs with a new texture_id so the app can use them
    texture_id = str(uuid.uuid4())[:8]
    urls = {}
    map_to_url = {
        "basecolor": ("texture_url", f"{texture_id}.png"),
        "normal": ("normal_map_url", f"{texture_id}_normal.png"),
        "roughness": ("roughness_map_url", f"{texture_id}_roughness.png"),
        "metallic": ("metallic_map_url", f"{texture_id}_metallic.png"),
        "height": ("height_map_url", f"{texture_id}_height.png"),
        "emissive": ("emissive_map_url", f"{texture_id}_emissive.png"),
    }

    import shutil
    for map_type, (url_key, filename) in map_to_url.items():
        src = mat_dir / f"{map_type}.png"
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / filename)
            urls[url_key] = f"/outputs/{filename}"

    return {
        **urls,
        "texture_id": texture_id,
        "seed": meta.get("seed"),
        "engine": meta.get("engine", ""),
        "name": meta.get("name", ""),
        "prompt": meta.get("prompt", ""),
    }


@app.delete("/library/{material_id}")
async def delete_material(material_id: str):
    """Delete a saved material."""
    import shutil
    mat_dir = LIBRARY_DIR / material_id
    if not mat_dir.exists():
        raise HTTPException(status_code=404, detail="Material not found")
    shutil.rmtree(mat_dir)
    return {"ok": True}


@app.post("/edit")
async def edit(request: EditRequest):
    """Edit an existing texture using img2img on the full grid."""
    if loading_state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline loading: {loading_state['step']}")

    pipeline = load_img2img_pipeline()

    # Load the source albedo
    if request.image_url.startswith('/outputs/'):
        image_path = OUTPUT_DIR / request.image_url.split('/')[-1]
        source_albedo = Image.open(image_path).convert("RGB")
    else:
        response = requests.get(request.image_url)
        source_albedo = Image.open(BytesIO(response.content)).convert("RGB")

    source_albedo = source_albedo.resize((GRID_CELL_SIZE, GRID_CELL_SIZE), Image.LANCZOS)

    # Try to load existing PBR maps for this texture
    texture_id_old = request.image_url.split('/')[-1].replace('.png', '')
    existing_maps = {"albedo": source_albedo}
    for name in ("normal", "roughness", "height", "metallic", "ao"):
        map_path = OUTPUT_DIR / f"{texture_id_old}_{name}.png"
        if map_path.exists():
            img = Image.open(map_path)
            if name in GRAYSCALE_MAPS:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((GRID_CELL_SIZE, GRID_CELL_SIZE), Image.LANCZOS)
            existing_maps[name] = img

    # Build input grid from existing maps
    input_grid = compose_grid(existing_maps)

    texture_id = str(uuid.uuid4())[:8]

    seed = request.seed or torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    full_prompt = f"pbr_multimap seamless tileable PBR texture, {request.prompt}, game texture, photorealistic material"
    print(f"Editing with prompt: {full_prompt}")

    grid_image = pipeline(
        prompt=full_prompt,
        image=input_grid,
        strength=request.strength,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    maps = split_grid(grid_image)
    maps = apply_tiling_to_maps(maps, request.tiling_mode)
    urls = save_split_maps(maps, texture_id)

    return {
        **urls,
        "seed": seed,
        "texture_id": texture_id,
    }

@app.post("/composite-pbr")
async def composite_pbr(request: CompositePbrRequest):
    """Generate PBR maps from a composited layer stack image via img2img."""
    import base64 as b64module

    if loading_state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline loading: {loading_state['step']}")

    pipeline = load_img2img_pipeline()

    # Strip data URL prefix
    data = request.image_data
    if "," in data:
        data = data.split(",", 1)[1]

    image_bytes = b64module.b64decode(data)
    albedo = Image.open(BytesIO(image_bytes)).convert("RGB")
    albedo = albedo.resize((GRID_CELL_SIZE, GRID_CELL_SIZE), Image.LANCZOS)

    # Build grid with the composited albedo and neutral maps
    input_grid = compose_grid({"albedo": albedo})

    texture_id = str(uuid.uuid4())[:8]

    seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    # Run img2img at low strength to generate coherent maps while preserving albedo
    grid_image = pipeline(
        prompt="pbr_multimap seamless tileable PBR texture, game texture, photorealistic material, top-down view, even lighting, highly detailed",
        image=input_grid,
        strength=0.5,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    maps = split_grid(grid_image)
    urls = save_split_maps(maps, texture_id)

    return {
        **urls,
        "texture_id": texture_id,
    }

@app.post("/export")
async def export(request: ExportRequest):
    """Export material for Unreal Engine, including all PBR maps."""
    texture_file = request.texture_url.split("/")[-1]
    texture_path = OUTPUT_DIR / texture_file

    if not texture_path.exists():
        raise HTTPException(status_code=404, detail="Texture not found")

    # Derive texture_id from filename
    texture_id = texture_file.replace('.png', '')

    export_id = str(uuid.uuid4())[:8]
    export_dir = OUTPUT_DIR / f"export_{export_id}"
    export_dir.mkdir(exist_ok=True)

    import shutil
    import json

    # Copy all available PBR map files
    map_files = {
        "T_Material_BaseColor.png": texture_path,
    }
    pbr_map_names = {
        "normal": "T_Material_Normal.png",
        "roughness": "T_Material_Roughness.png",
        "height": "T_Material_Height.png",
        "metallic": "T_Material_Metallic.png",
        "ao": "T_Material_AO.png",
        "emissive": "T_Material_Emissive.png",
    }

    for map_name, export_name in pbr_map_names.items():
        map_path = OUTPUT_DIR / f"{texture_id}_{map_name}.png"
        if map_path.exists():
            map_files[export_name] = map_path

    for export_name, src_path in map_files.items():
        shutil.copy(src_path, export_dir / export_name)

    # Generate Unreal Python script
    unreal_script = f'''# Unreal Engine Material Setup Script
# Run this in Unreal's Python console

import unreal

# Create material
material_name = "M_GeneratedMaterial_{export_id}"
material_path = f"/Game/Materials/{{material_name}}"

# Import textures
texture_path = "/Game/Textures/T_Material_BaseColor"

# Create material instance
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
material_factory = unreal.MaterialFactoryNew()

material = asset_tools.create_asset(
    material_name,
    "/Game/Materials",
    unreal.Material,
    material_factory
)

# Set material properties
material.set_editor_property("roughness", {request.roughness})
material.set_editor_property("metallic", {request.metalness})

print(f"Material created: {{material_path}}")
'''

    (export_dir / "setup_material.py").write_text(unreal_script)

    # Material info JSON
    textures_info = {"baseColor": "T_Material_BaseColor.png"}
    for map_name, export_name in pbr_map_names.items():
        if (export_dir / export_name).exists():
            textures_info[map_name] = export_name

    material_info = {
        "name": f"GeneratedMaterial_{export_id}",
        "roughness": request.roughness,
        "metalness": request.metalness,
        "textures": textures_info,
    }
    (export_dir / "material_info.json").write_text(json.dumps(material_info, indent=2))

    # Create ZIP
    zip_path = OUTPUT_DIR / f"material_export_{export_id}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in export_dir.iterdir():
            zf.write(file, file.name)

    shutil.rmtree(export_dir)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"material_export_{export_id}.zip"
    )

# ---------------------------------------------------------------------------
# Layer Stack — server-side state with JSON persistence
# ---------------------------------------------------------------------------

import json as _json

_layer_lock = threading.Lock()
_layer_stack: list[dict] = []


def _default_channels() -> dict:
    return {
        "basecolor": {"enabled": True},
        "normal": {"enabled": True},
        "roughness": {"enabled": True},
        "metalness": {"enabled": True},
        "height": {"enabled": True},
        "translucency": {"enabled": False},
        "subsurface": {"enabled": False},
    }


def _default_mask() -> dict:
    return {
        "enabled": False,
        "noiseType": "perlin",
        "scale": 4,
        "seed": 0,
        "invert": False,
    }


def _layer_with_defaults(layer: dict) -> dict:
    """Fill in missing fields on a layer dict with sensible defaults."""
    out = dict(layer)
    out.setdefault("id", f"layer-{uuid.uuid4().hex[:8]}")
    out.setdefault("name", "Layer")
    out.setdefault("visible", True)
    out.setdefault("opacity", 1.0)
    out.setdefault("blendMode", "normal")
    out.setdefault("type", "material")
    out.setdefault("materialPrompt", "")
    out.setdefault("materialMaps", {})
    out.setdefault("generating", False)
    if not out.get("channels"):
        out["channels"] = _default_channels()
    out.setdefault("color", "#888888")
    out.setdefault("noiseType", "perlin")
    out.setdefault("noiseScale", 4)
    out.setdefault("noiseSeed", 0)
    out.setdefault("noiseColor1", "#000000")
    out.setdefault("noiseColor2", "#ffffff")
    if not out.get("mask"):
        out["mask"] = _default_mask()
    return out


def _save_layers() -> None:
    LAYERS_FILE.write_text(_json.dumps(_layer_stack, indent=2))


def _load_layers() -> None:
    global _layer_stack
    if LAYERS_FILE.exists():
        try:
            _layer_stack = _json.loads(LAYERS_FILE.read_text())
            if not isinstance(_layer_stack, list):
                _layer_stack = []
        except Exception as e:
            print(f"Failed to load layers.json: {e}")
            _layer_stack = []
    else:
        _layer_stack = []


_load_layers()


@app.get("/layers")
async def list_layers():
    """Return the current layer stack (bottom first, top last)."""
    with _layer_lock:
        return list(_layer_stack)


@app.post("/layers")
async def add_layer(layer: Optional[dict] = None):
    """Append a new layer to the stack. If body is empty, creates a default layer."""
    with _layer_lock:
        new_layer = _layer_with_defaults(layer or {})
        # Ensure name uniqueness by default
        if layer is None or not layer.get("name"):
            new_layer["name"] = f"Layer {len(_layer_stack) + 1}"
        if not new_layer.get("id") or any(l["id"] == new_layer["id"] for l in _layer_stack):
            new_layer["id"] = f"layer-{uuid.uuid4().hex[:8]}"
        _layer_stack.append(new_layer)
        _save_layers()
        return new_layer


@app.patch("/layers/{layer_id}")
async def update_layer(layer_id: str, updates: dict):
    """Partial update on a layer. Returns the updated layer."""
    with _layer_lock:
        for i, l in enumerate(_layer_stack):
            if l["id"] == layer_id:
                _layer_stack[i] = {**l, **updates, "id": layer_id}
                _save_layers()
                return _layer_stack[i]
    raise HTTPException(status_code=404, detail="Layer not found")


@app.delete("/layers/{layer_id}")
async def delete_layer(layer_id: str):
    """Remove a layer from the stack."""
    with _layer_lock:
        before = len(_layer_stack)
        _layer_stack[:] = [l for l in _layer_stack if l["id"] != layer_id]
        if len(_layer_stack) == before:
            raise HTTPException(status_code=404, detail="Layer not found")
        _save_layers()
        return {"ok": True}


@app.post("/layers/{layer_id}/duplicate")
async def duplicate_layer(layer_id: str):
    """Duplicate a layer, inserting the copy directly above the source."""
    with _layer_lock:
        for i, l in enumerate(_layer_stack):
            if l["id"] == layer_id:
                copy = dict(l)
                copy["id"] = f"layer-{uuid.uuid4().hex[:8]}"
                copy["name"] = f"{l.get('name', 'Layer')} copy"
                _layer_stack.insert(i + 1, copy)
                _save_layers()
                return copy
    raise HTTPException(status_code=404, detail="Layer not found")


@app.post("/layers/reorder")
async def reorder_layers(request: ReorderRequest):
    """Reorder the stack. `order` must be a permutation of existing layer IDs."""
    with _layer_lock:
        current_ids = {l["id"] for l in _layer_stack}
        if set(request.order) != current_ids:
            raise HTTPException(
                status_code=400,
                detail="order must contain exactly the current layer IDs",
            )
        by_id = {l["id"]: l for l in _layer_stack}
        _layer_stack[:] = [by_id[i] for i in request.order]
        _save_layers()
        return list(_layer_stack)


@app.delete("/layers")
async def clear_layers():
    """Remove all layers."""
    with _layer_lock:
        _layer_stack.clear()
        _save_layers()
        return {"ok": True}


# ---------------------------------------------------------------------------
# Motion / animation — server-side scalar animation state
# ---------------------------------------------------------------------------

_motion_lock = threading.Lock()
_motion_state: dict = {}

_ANIMATED_PROPS = {
    "pan_x", "pan_y", "displacement", "transmission",
    "ior", "emissive", "roughness", "metalness",
}
_ANIM_MODES = {"static", "sin", "cos", "linear"}


def _save_motion() -> None:
    MOTION_FILE.write_text(_json.dumps(_motion_state, indent=2))


def _load_motion() -> None:
    global _motion_state
    if MOTION_FILE.exists():
        try:
            loaded = _json.loads(MOTION_FILE.read_text())
            _motion_state = loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            print(f"Failed to load motion.json: {e}")
            _motion_state = {}
    else:
        _motion_state = {}


def _validate_motion_cfg(cfg: dict) -> dict:
    """Validate and fill in defaults for a motion config entry."""
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="Motion config must be an object")
    mode = cfg.get("mode", "sin")
    if mode not in _ANIM_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'")
    return {
        "mode": mode,
        "base": float(cfg.get("base", 0.0)),
        "amp": float(cfg.get("amp", 0.0)),
        "freq": float(cfg.get("freq", 0.0)),
        "phase": float(cfg.get("phase", 0.0)),
    }


_load_motion()


@app.get("/motion")
async def get_motion():
    """Return the current motion state for all animated properties."""
    with _motion_lock:
        return dict(_motion_state)


@app.patch("/motion")
async def patch_motion(updates: dict):
    """Partial update: set or clear motion for one or more properties.

    Body: { "<prop>": <config> | null, ... } — null clears that property.
    Valid props: pan_x, pan_y, displacement, transmission, ior, emissive,
    roughness, metalness.
    """
    with _motion_lock:
        for prop, cfg in updates.items():
            if prop not in _ANIMATED_PROPS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown animated property '{prop}'. Valid: {sorted(_ANIMATED_PROPS)}",
                )
            if cfg is None:
                _motion_state.pop(prop, None)
            else:
                _motion_state[prop] = _validate_motion_cfg(cfg)
        _save_motion()
        return dict(_motion_state)


@app.delete("/motion")
async def clear_motion():
    """Remove all motion configs."""
    with _motion_lock:
        _motion_state.clear()
        _save_motion()
        return {"ok": True}


# Serve frontend — must be after all API routes
UI_DIR = PROJECT_ROOT / "ui" / "dist"
if UI_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIR / "assets")), name="ui_assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React frontend for any non-API route."""
        file_path = UI_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(UI_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
