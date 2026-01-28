#!/usr/bin/env python3
"""
Text-to-Material Generator

Generate PBR material maps from text descriptions using Flux.1-dev.
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline

# Use local Flux.1-dev
MODEL_ID = "black-forest-labs/FLUX.1-dev"


def make_seamless(image: Image.Image) -> Image.Image:
    """
    Make an image tile seamlessly using mirror blending.
    Simple approach - can improve later with latent-space tiling.
    """
    import numpy as np
    
    img = np.array(image)
    h, w = img.shape[:2]
    blend_size = w // 8
    
    result = img.copy()
    
    # Horizontal blend
    for i in range(blend_size):
        alpha = i / blend_size
        result[:, i] = ((1 - alpha) * img[:, w - blend_size + i] + alpha * img[:, i]).astype(np.uint8)
        result[:, w - 1 - i] = ((1 - alpha) * img[:, blend_size - 1 - i] + alpha * img[:, w - 1 - i]).astype(np.uint8)
    
    # Vertical blend
    for i in range(blend_size):
        alpha = i / blend_size
        result[i, :] = ((1 - alpha) * result[h - blend_size + i, :] + alpha * result[i, :]).astype(np.uint8)
        result[h - 1 - i, :] = ((1 - alpha) * result[blend_size - 1 - i, :] + alpha * result[h - 1 - i, :]).astype(np.uint8)
    
    return Image.fromarray(result)


def generate_material(
    prompt: str,
    output_dir: Path,
    resolution: int = 512,
    seamless: bool = True,
    seed: int = None,
    steps: int = 28,
):
    """
    Generate a PBR material from a text prompt using Flux.1-dev.
    
    Phase 1: Generates albedo/diffuse map
    Phase 2: Will generate full PBR set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Flux.1-dev on {device}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()  # Saves VRAM
    
    # Enhance prompt for material generation
    material_prompt = f"seamless tileable PBR texture of {prompt}, game texture, photorealistic material, top-down flat view, even lighting, no shadows, highly detailed surface, 4k quality"
    
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
    
    print(f"Generating: {prompt}")
    print(f"Resolution: {resolution}x{resolution}, Steps: {steps}")
    
    image = pipe(
        prompt=material_prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]
    
    if seamless:
        print("Making seamless...")
        image = make_seamless(image)
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean filename from prompt
    clean_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
    clean_name = clean_name.replace(" ", "_")[:50]
    
    albedo_path = output_dir / f"{clean_name}_albedo.png"
    image.save(albedo_path)
    print(f"Saved: {albedo_path}")
    
    # Create tiled preview
    tiled = Image.new('RGB', (resolution * 2, resolution * 2))
    for x in range(2):
        for y in range(2):
            tiled.paste(image, (x * resolution, y * resolution))
    
    tiled_path = output_dir / f"{clean_name}_tiled.png"
    tiled.save(tiled_path)
    print(f"Saved tiled preview: {tiled_path}")
    
    return {
        "albedo": albedo_path,
        "tiled": tiled_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PBR materials from text using Flux.1")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Material description")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resolution", "-r", type=int, default=512, choices=[512, 768, 1024], help="Output resolution")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps (default: 28)")
    parser.add_argument("--no-seamless", action="store_true", help="Disable seamless tiling")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    results = generate_material(
        prompt=args.prompt,
        output_dir=args.output,
        resolution=args.resolution,
        seamless=not args.no_seamless,
        seed=args.seed,
        steps=args.steps,
    )
    
    print("\nGenerated material maps:")
    for map_type, path in results.items():
        print(f"  {map_type}: {path}")


if __name__ == "__main__":
    main()
