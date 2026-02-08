#!/usr/bin/env python3
"""FastAPI backend for Text-to-Material generation."""

import os
import uuid
import zipfile
from pathlib import Path
from typing import Optional
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from peft import PeftModel
from PIL import Image
import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
import requests
from io import BytesIO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "api" / "outputs"
LORA_PATH = PROJECT_ROOT / "output" / "loras" / "pbr-materials-v1" / "final_lora"
MERGED_MODEL_PATH = PROJECT_ROOT / "output" / "merged_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Text-to-Material API")

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

def _enhance_detail(gray: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Unsharp mask to recover fine detail before computing PBR maps."""
    blurred = gaussian_filter(gray, sigma=3.0)
    return np.clip(gray + (gray - blurred) * amount, 0.0, 1.0)

def _contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """Stretch values to use the full [0,1] range."""
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo > 0.01:
        return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return gray

def generate_normal_map(image: Image.Image, strength: float = 2.5) -> Image.Image:
    """Multi-scale normal map — combines fine and coarse surface detail.

    Uses detail-enhanced grayscale at two scales so both micro-texture
    bumps (pores, grain) and macro-structure (bricks, cracks) show up.
    """
    gray = np.array(image.convert('L')).astype(np.float32) / 255.0
    gray = _enhance_detail(gray, 0.8)  # sharpen before gradient extraction

    # Fine detail: raw gradients capture pores/scratches
    gx_fine = sobel(gray, axis=1) * strength * 1.5
    gy_fine = sobel(gray, axis=0) * strength * 1.5

    # Coarse structure: blurred gradients capture large-scale shape
    coarse = gaussian_filter(gray, sigma=3.0)
    gx_coarse = sobel(coarse, axis=1) * strength * 0.6
    gy_coarse = sobel(coarse, axis=0) * strength * 0.6

    gx = gx_fine + gx_coarse
    gy = -(gy_fine + gy_coarse)  # OpenGL convention (Y flipped)

    gz = np.ones_like(gx)
    length = np.sqrt(gx**2 + gy**2 + gz**2)
    nx = gx / length
    ny = gy / length
    nz = gz / length

    normal_map = np.stack([
        ((nx + 1.0) * 0.5 * 255).astype(np.uint8),
        ((ny + 1.0) * 0.5 * 255).astype(np.uint8),
        ((nz + 1.0) * 0.5 * 255).astype(np.uint8),
    ], axis=-1)

    return Image.fromarray(normal_map)

def generate_height_map(image: Image.Image) -> Image.Image:
    """Height map with contrast-stretched luminance and preserved detail."""
    gray = np.array(image.convert('L')).astype(np.float32) / 255.0

    # Minimal blur — just enough to suppress single-pixel noise
    gray = gaussian_filter(gray, sigma=0.3)

    # Contrast stretch so displacement uses the full range
    gray = _contrast_stretch(gray)

    return Image.fromarray((gray * 255).astype(np.uint8))

def generate_roughness_map(image: Image.Image) -> Image.Image:
    """Roughness map from local detail variance.

    Textured / high-frequency areas → rough (matte).
    Smooth / uniform areas → slightly less rough.
    Base roughness is high because most real-world materials are matte.
    """
    gray = np.array(image.convert('L')).astype(np.float32) / 255.0
    gray = _enhance_detail(gray, 0.5)

    # Small filter preserves fine detail
    local_mean = uniform_filter(gray, size=5)
    local_var = uniform_filter((gray - local_mean) ** 2, size=5)
    detail = np.sqrt(np.clip(local_var, 0, None))

    # Base 0.55, detail adds up to 0.45 → range [0.55, 1.0]
    # Most surfaces should read as matte; only the smoothest areas go below 0.6
    roughness = np.clip(0.55 + detail * 4.0, 0.35, 1.0)
    return Image.fromarray((roughness * 255).astype(np.uint8))

def generate_emissive_map(image: Image.Image) -> Image.Image:
    """Emissive map — only very bright albedo areas emit."""
    gray = np.array(image.convert('L')).astype(np.float32) / 255.0
    emissive = np.clip((gray - 0.85) / 0.15, 0.0, 1.0)
    return Image.fromarray((emissive * 255).astype(np.uint8))

def generate_all_pbr_maps(image: Image.Image, texture_id: str) -> dict:
    """Generate and save all PBR channel maps. Returns URL dict."""
    maps = {
        "normal": generate_normal_map(image),
        "height": generate_height_map(image),
        "roughness": generate_roughness_map(image),
        "emissive": generate_emissive_map(image),
    }
    urls = {}
    for name, img in maps.items():
        path = OUTPUT_DIR / f"{texture_id}_{name}.png"
        img.save(path)
        urls[f"{name}_map_url"] = f"/outputs/{texture_id}_{name}.png"
    return urls

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
        # Check if we have a pre-merged model
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
            # First time: load base + LoRA, merge, and save
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
                
                # Save merged model for faster future loads
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
    
    # Make sure base pipeline is loaded first
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

@app.on_event("startup")
async def startup():
    """Pre-load the pipeline on startup."""
    def preload():
        try:
            load_pipeline()
        except Exception as e:
            print(f"Preload failed: {e}")
    threading.Thread(target=preload, daemon=True).start()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/status")
async def status():
    """Check pipeline loading status."""
    return {
        "pipeline_loaded": pipe is not None and loading_state["status"] == "ready",
        "img2img_loaded": pipe_img2img is not None,
        "ready": loading_state["status"] == "ready",
        "loading": loading_state["status"] == "loading",
        "step": loading_state["step"],
        "progress": loading_state["progress"],
        "error": loading_state["error"],
        "elapsed": time.time() - loading_state["start_time"] if loading_state["start_time"] else 0,
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate a material texture from prompt."""
    if loading_state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline loading: {loading_state['step']}")
    
    pipe = load_pipeline()
    
    # Generate unique ID for this texture
    texture_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{texture_id}.png"
    
    # Set seed
    seed = request.seed or torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"Generating: {request.prompt}")
    
    image = pipe(
        prompt=request.prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    
    # Apply tiling post-processing if requested
    if request.tiling_mode != "none":
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from tile_postprocess import make_tileable, make_tileable_multipass, make_tileable_multiscale
        
        if request.tiling_mode == "basic":
            image = make_tileable(image)
        elif request.tiling_mode == "multipass":
            image = make_tileable_multipass(image)
        elif request.tiling_mode == "multiscale":
            image = make_tileable_multiscale(image)
    
    image.save(output_path)

    # Generate all PBR channel maps from the post-tiled albedo
    pbr_urls = generate_all_pbr_maps(image, texture_id)

    return {
        "texture_url": f"/outputs/{texture_id}.png",
        **pbr_urls,
        "seed": seed,
        "texture_id": texture_id
    }

@app.post("/edit")
async def edit(request: EditRequest):
    """Edit an existing texture using img2img."""
    if loading_state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline loading: {loading_state['step']}")
    
    pipe = load_img2img_pipeline()
    
    # Load the source image
    if request.image_url.startswith('/outputs/'):
        # Local file
        image_path = OUTPUT_DIR / request.image_url.split('/')[-1]
        source_image = Image.open(image_path).convert("RGB")
    else:
        # Remote URL
        response = requests.get(request.image_url)
        source_image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Resize to 512x512 if needed
    source_image = source_image.resize((512, 512), Image.LANCZOS)
    
    # Generate unique ID
    texture_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{texture_id}.png"
    
    # Set seed
    seed = request.seed or torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"Editing with prompt: {request.prompt}")
    
    image = pipe(
        prompt=f"seamless tileable PBR texture, {request.prompt}, game texture, photorealistic material",
        image=source_image,
        strength=request.strength,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]
    
    # Apply tiling post-processing if requested
    if request.tiling_mode != "none":
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from tile_postprocess import make_tileable, make_tileable_multipass, make_tileable_multiscale
        
        if request.tiling_mode == "basic":
            image = make_tileable(image)
        elif request.tiling_mode == "multipass":
            image = make_tileable_multipass(image)
        elif request.tiling_mode == "multiscale":
            image = make_tileable_multiscale(image)
    
    image.save(output_path)

    # Generate all PBR channel maps from the post-tiled albedo
    pbr_urls = generate_all_pbr_maps(image, texture_id)

    return {
        "texture_url": f"/outputs/{texture_id}.png",
        **pbr_urls,
        "seed": seed,
        "texture_id": texture_id
    }

@app.post("/composite-pbr")
async def composite_pbr(request: CompositePbrRequest):
    """Generate PBR maps from a composited layer stack image."""
    import base64 as b64module

    # Strip data URL prefix
    data = request.image_data
    if "," in data:
        data = data.split(",", 1)[1]

    # Decode base64 → PIL Image
    image_bytes = b64module.b64decode(data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Generate unique ID and save albedo
    texture_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{texture_id}.png"
    image.save(output_path)

    # Generate all PBR channel maps
    pbr_urls = generate_all_pbr_maps(image, texture_id)

    return {
        "texture_url": f"/outputs/{texture_id}.png",
        **pbr_urls,
        "texture_id": texture_id,
    }

@app.post("/export")
async def export(request: ExportRequest):
    """Export material for Unreal Engine."""
    # Get texture filename from URL
    texture_file = request.texture_url.split("/")[-1]
    texture_path = OUTPUT_DIR / texture_file
    
    if not texture_path.exists():
        raise HTTPException(status_code=404, detail="Texture not found")
    
    # Create export package
    export_id = str(uuid.uuid4())[:8]
    export_dir = OUTPUT_DIR / f"export_{export_id}"
    export_dir.mkdir(exist_ok=True)
    
    # Copy texture
    import shutil
    shutil.copy(texture_path, export_dir / "T_Material_BaseColor.png")
    
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
    
    # Create material info JSON
    import json
    material_info = {
        "name": f"GeneratedMaterial_{export_id}",
        "roughness": request.roughness,
        "metalness": request.metalness,
        "textures": {
            "baseColor": "T_Material_BaseColor.png"
        }
    }
    (export_dir / "material_info.json").write_text(json.dumps(material_info, indent=2))
    
    # Create README
    readme = f"""# Generated Material Export

## Contents
- T_Material_BaseColor.png - Diffuse/Albedo texture
- setup_material.py - Unreal Engine Python setup script
- material_info.json - Material properties

## Usage in Unreal Engine
1. Import textures to /Game/Textures/
2. Run setup_material.py in Unreal's Python console
3. Or manually create material with these settings:
   - Roughness: {request.roughness}
   - Metallic: {request.metalness}

Generated by Text-to-Material
"""
    (export_dir / "README.md").write_text(readme)
    
    # Create ZIP
    zip_path = OUTPUT_DIR / f"material_export_{export_id}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in export_dir.iterdir():
            zf.write(file, file.name)
    
    # Cleanup export dir
    shutil.rmtree(export_dir)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"material_export_{export_id}.zip"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
