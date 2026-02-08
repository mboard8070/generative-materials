#!/usr/bin/env python3
"""Quick generation script to test the trained LoRA."""

import argparse
from pathlib import Path
import torch
from diffusers import FluxPipeline
from peft import PeftModel

def generate(prompt: str, output_path: str, lora_path: str, seed: int = 42):
    print(f"Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Loading LoRA from {lora_path}...")
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        lora_path,
    )
    
    pipe.to("cuda")
    
    print(f"Generating: {prompt}")
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    
    image.save(output_path)
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Generation prompt")
    parser.add_argument("-o", "--output", type=str, default="output.png")
    parser.add_argument("-l", "--lora", type=str, 
                        default="output/loras/pbr-materials-v1/final_lora")
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()
    
    generate(args.prompt, args.output, args.lora, args.seed)
