#!/usr/bin/env python3
"""
Text-to-Material Generator

Generate PBR material maps from text descriptions using Flux.1-dev.
Supports both direct prompts and surface property sliders.
"""

import os
import sys

# Block Apex fused norms - they don't support bf16
class _ApexBlocker:
    def find_module(self, name, path=None):
        if name.startswith("apex"):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"Blocked {name}")

sys.meta_path.insert(0, _ApexBlocker())

import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline

from prompt_builder import build_prompt, get_preset, list_presets

MODEL_ID = "black-forest-labs/FLUX.1-dev"


def make_seamless(image: Image.Image) -> Image.Image:
    """Make an image tile seamlessly using offset + cross-blend method."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    img = np.array(image, dtype=np.float32)
    h, w = img.shape[:2]
    
    # Step 1: Offset image by half in both directions (wraps edges to center)
    offset_img = np.roll(np.roll(img, h // 2, axis=0), w // 2, axis=1)
    
    # Step 2: Create cross-shaped blend mask (blend the center seams)
    blend_width = w // 4  # Width of blend zone
    
    # Create smooth gradient masks
    y_mask = np.ones((h, w), dtype=np.float32)
    x_mask = np.ones((h, w), dtype=np.float32)
    
    # Vertical blend zone (horizontal seam)
    for i in range(blend_width):
        alpha = i / blend_width
        # Smooth cosine interpolation
        alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
        y_mask[h // 2 - blend_width // 2 + i, :] = alpha
        y_mask[h // 2 + blend_width // 2 - i - 1, :] = alpha
    y_mask[h // 2 - blend_width // 2:h // 2 + blend_width // 2, :] = np.minimum(
        y_mask[h // 2 - blend_width // 2:h // 2 + blend_width // 2, :],
        np.linspace(0, 1, blend_width)[:, np.newaxis]
    )
    
    # Horizontal blend zone (vertical seam)  
    for i in range(blend_width):
        alpha = i / blend_width
        alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
        x_mask[:, w // 2 - blend_width // 2 + i] = alpha
        x_mask[:, w // 2 + blend_width // 2 - i - 1] = alpha
    x_mask[:, w // 2 - blend_width // 2:w // 2 + blend_width // 2] = np.minimum(
        x_mask[:, w // 2 - blend_width // 2:w // 2 + blend_width // 2],
        np.linspace(0, 1, blend_width)[np.newaxis, :]
    )
    
    # Combine masks
    mask = np.minimum(x_mask, y_mask)
    mask = gaussian_filter(mask, sigma=blend_width // 8)  # Smooth edges
    
    # Expand mask to 3 channels
    mask_3d = mask[:, :, np.newaxis]
    
    # Step 3: Blend original (offset) with itself using mask
    # The trick: blend offset image with original at the center cross
    result = offset_img * mask_3d + np.roll(np.roll(offset_img, h // 2, axis=0), w // 2, axis=1) * (1 - mask_3d)
    
    # Step 4: Offset back
    result = np.roll(np.roll(result, h // 2, axis=0), w // 2, axis=1)
    
    return Image.fromarray(result.astype(np.uint8))


def generate_material(
    prompt: str = None,
    base_material: str = None,
    preset: str = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
    age: float = 0.0,
    color: str = "",
    output_dir: Path = Path("outputs"),
    resolution: int = 512,
    seamless: bool = True,
    seed: int = None,
    steps: int = 28,
):
    """
    Generate a PBR material using Flux.1-dev.
    
    Can use either:
    - Direct prompt string
    - Preset name
    - Base material + property sliders
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build prompt
    if prompt:
        # Direct prompt provided
        final_prompt = prompt
    elif preset:
        # Use preset
        preset_values = get_preset(preset)
        final_prompt = build_prompt(
            base_material=preset.replace("_", " "),
            roughness=preset_values.get("roughness", roughness),
            metallic=preset_values.get("metallic", metallic),
            age=preset_values.get("age", age),
            color=color,
        )
    elif base_material:
        # Use sliders
        final_prompt = build_prompt(
            base_material=base_material,
            roughness=roughness,
            metallic=metallic,
            age=age,
            color=color,
        )
    else:
        raise ValueError("Must provide --prompt, --preset, or --material")
    
    print(f"Loading Flux.1-dev on {device}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
    
    print(f"\nPrompt: {final_prompt}\n")
    print(f"Resolution: {resolution}x{resolution}, Steps: {steps}")
    
    image = pipe(
        prompt=final_prompt,
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
    
    # Generate filename
    if base_material:
        clean_name = base_material.replace(" ", "_")[:30]
    elif preset:
        clean_name = preset
    else:
        clean_name = "material"
    
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
        "prompt": final_prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate PBR materials from text using Flux.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct prompt
  python generate.py -p "rusty weathered steel with scratches"
  
  # Using preset
  python generate.py --preset rusted_steel
  
  # Using sliders
  python generate.py -m "steel" --roughness 0.6 --age 0.8 --color "dark gray"
  
  # List presets
  python generate.py --list-presets
        """
    )
    
    # Input modes
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--prompt", "-p", type=str, help="Direct prompt string")
    input_group.add_argument("--preset", type=str, help="Material preset name")
    input_group.add_argument("--material", "-m", type=str, help="Base material type")
    
    # Surface properties (for slider mode)
    parser.add_argument("--roughness", type=float, default=0.5, help="Roughness 0-1 (default: 0.5)")
    parser.add_argument("--metallic", type=float, default=0.0, help="Metallic 0-1 (default: 0.0)")
    parser.add_argument("--age", type=float, default=0.0, help="Age/wear 0-1 (default: 0.0)")
    parser.add_argument("--color", type=str, default="", help="Color description")
    
    # Generation settings
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resolution", "-r", type=int, default=512, choices=[512, 768, 1024])
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--no-seamless", action="store_true", help="Disable seamless tiling")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    
    # Utility
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\n=== Available Material Presets ===\n")
        for preset in list_presets():
            values = get_preset(preset)
            print(f"  {preset:20} R:{values.get('roughness', 0.5):.1f}  M:{values.get('metallic', 0):.1f}  A:{values.get('age', 0):.1f}")
        return
    
    if not (args.prompt or args.preset or args.material):
        parser.error("Must provide --prompt, --preset, or --material (or --list-presets)")
    
    results = generate_material(
        prompt=args.prompt,
        base_material=args.material,
        preset=args.preset,
        roughness=args.roughness,
        metallic=args.metallic,
        age=args.age,
        color=args.color,
        output_dir=Path(args.output),
        resolution=args.resolution,
        seamless=not args.no_seamless,
        seed=args.seed,
        steps=args.steps,
    )
    
    print("\n=== Generated ===")
    print(f"Prompt: {results['prompt']}")
    print(f"Albedo: {results['albedo']}")
    print(f"Tiled:  {results['tiled']}")


if __name__ == "__main__":
    main()
