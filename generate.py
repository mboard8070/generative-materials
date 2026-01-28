#!/usr/bin/env python3
"""
Text-to-Material Generator

Generate PBR material maps from text descriptions.
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Default model - can swap for fine-tuned version later
MODEL_ID = "stabilityai/stable-diffusion-2-1"

def make_seamless(image: Image.Image) -> Image.Image:
    """
    Make an image tile seamlessly using mirror blending.
    Simple approach - can improve later.
    """
    import numpy as np
    
    img = np.array(image)
    h, w = img.shape[:2]
    blend_size = w // 8
    
    # Create seamless version by blending edges
    # This is a basic approach - Phase 2 will do this in latent space
    result = img.copy()
    
    # Horizontal blend
    for i in range(blend_size):
        alpha = i / blend_size
        result[:, i] = (1 - alpha) * img[:, w - blend_size + i] + alpha * img[:, i]
        result[:, w - 1 - i] = (1 - alpha) * img[:, blend_size - 1 - i] + alpha * img[:, w - 1 - i]
    
    return Image.fromarray(result.astype(np.uint8))


def generate_material(
    prompt: str,
    output_dir: Path,
    resolution: int = 512,
    seamless: bool = True,
    seed: int = None,
):
    """
    Generate a PBR material from a text prompt.
    
    Phase 1: Just generates albedo map
    Phase 2: Will generate full PBR set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading model on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Enhance prompt for material generation
    material_prompt = f"seamless tileable texture of {prompt}, PBR material, game texture, 4k, highly detailed, top down view, flat lighting"
    negative_prompt = "text, watermark, signature, frame, border, 3d render, perspective, objects, items"
    
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    print(f"Generating: {prompt}")
    image = pipe(
        prompt=material_prompt,
        negative_prompt=negative_prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=30,
        guidance_scale=7.5,
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
    
    # Phase 2: Generate other maps here
    # normal_map = generate_normal(image)
    # roughness_map = generate_roughness(prompt, image)
    # etc.
    
    return {
        "albedo": albedo_path,
        # "normal": normal_path,
        # "roughness": roughness_path,
        # "metallic": metallic_path,
        # "height": height_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PBR materials from text")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Material description")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resolution", "-r", type=int, default=512, choices=[512, 1024, 2048], help="Output resolution")
    parser.add_argument("--no-seamless", action="store_true", help="Disable seamless tiling")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    results = generate_material(
        prompt=args.prompt,
        output_dir=args.output,
        resolution=args.resolution,
        seamless=not args.no_seamless,
        seed=args.seed,
    )
    
    print("\nGenerated material maps:")
    for map_type, path in results.items():
        print(f"  {map_type}: {path}")


if __name__ == "__main__":
    main()
