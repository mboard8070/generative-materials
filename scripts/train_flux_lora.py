#!/usr/bin/env python3
"""
FLUX.1-dev LoRA Training Script for PBR Material Textures

This script trains a LoRA adapter on FLUX.1-dev to generate seamless PBR textures.
Optimized for DGX Spark with 128GB unified memory.
"""

import os
import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import safetensors.torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_DIR = PROJECT_ROOT / "output" / "loras"

# Default to HuggingFace model (will be cached locally after first download)
DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-dev"


class StyleDataset(Dataset):
    """Dataset for loading training images with captions."""

    def __init__(
        self,
        image_dir: str,
        resolution: int = 1024,
        caption_ext: str = ".txt"
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.caption_ext = caption_ext

        # Find all images with corresponding caption files
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths = []

        for ext in extensions:
            for img_path in self.image_dir.glob(f"*{ext}"):
                caption_path = img_path.with_suffix(caption_ext)
                if caption_path.exists():
                    self.image_paths.append((img_path, caption_path))

        print(f"Found {len(self.image_paths)} image-caption pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, caption_path = self.image_paths[idx]

        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Load caption
        caption = caption_path.read_text().strip()

        return {
            "image": image,
            "caption": caption,
            "path": str(img_path)
        }


def collate_fn(examples):
    """Collate function for DataLoader."""
    images = [ex["image"] for ex in examples]
    captions = [ex["caption"] for ex in examples]
    paths = [ex["path"] for ex in examples]
    return {"images": images, "captions": captions, "paths": paths}


def generate_position_ids_flux(latent_height, latent_width, device, dtype):
    """Generate position IDs for FLUX transformer based on latent dimensions.

    Position IDs account for 2x2 latent packing.
    Returns 2D tensor (no batch dimension) as expected by newer diffusers.
    """
    packed_h, packed_w = latent_height // 2, latent_width // 2
    img_ids = torch.zeros(packed_h, packed_w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_w, device=device, dtype=dtype)[None, :]
    img_ids = rearrange(img_ids, "h w c -> (h w) c")
    return img_ids


def train_flux_lora(
    image_dir: str,
    output_dir: str,
    model_id: str = DEFAULT_FLUX_MODEL,
    resolution: int = 1024,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    max_train_steps: int = 1000,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 250,
    seed: int = 42,
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    caption_ext: str = ".txt",
):
    """Train a LoRA adapter on FLUX.1-dev."""

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=str(output_path),
        logging_dir=str(output_path / "logs")
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=accelerator_config,
    )

    if accelerator.is_main_process:
        print(f"Training FLUX.1-dev LoRA")
        print(f"  Model: {model_id}")
        print(f"  Images: {image_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Resolution: {resolution}")
        print(f"  LoRA Rank: {lora_rank}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Max Steps: {max_train_steps}")

    # Set seed
    torch.manual_seed(seed)

    # Load dataset
    dataset = StyleDataset(
        image_dir=image_dir,
        resolution=resolution,
        caption_ext=caption_ext
    )

    if len(dataset) == 0:
        print(f"ERROR: No image-caption pairs found in {image_dir}")
        print("Make sure your images have corresponding .txt caption files.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load FLUX pipeline
    print(f"Loading FLUX.1-dev model from {model_id}...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
    )

    # Get the transformer (main model to train)
    transformer = pipe.transformer

    # Enable gradient checkpointing
    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Setup LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Setup optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            print("bitsandbytes not available, using standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Setup scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=max_train_steps,
    )

    # Prepare with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # Move other components to device
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)
    pipe.vae.to(accelerator.device)

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps),
        desc="Training",
        disable=not accelerator.is_main_process
    )

    transformer.train()

    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Encode images to latents
                images = batch["images"]
                captions = batch["captions"]

                # Process images through VAE
                pixel_values = torch.stack([
                    pipe.image_processor.preprocess(img)[0]
                    for img in images
                ]).to(accelerator.device, dtype=pipe.vae.dtype)

                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

                # Encode text (ignore text_ids, we generate our own)
                prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=captions,
                    prompt_2=captions,
                    device=accelerator.device,
                )

                # Sample noise
                noise = torch.randn_like(latents)

                # Sample timesteps for flow matching
                # Use sigmoid sampling for better distribution
                num_timesteps = 1000
                t_raw = torch.sigmoid(torch.randn((latents.shape[0],), device=accelerator.device))
                timesteps = ((1 - t_raw) * num_timesteps).to(latents.dtype)

                # Normalized timestep for interpolation [0, 1]
                t_01 = (timesteps / num_timesteps).view(-1, 1, 1, 1).to(latents.dtype)

                # Flow matching interpolation: zt = (1 - t) * x0 + t * noise
                noisy_latents = (1.0 - t_01) * latents + t_01 * noise

                # Pack latents using einops (FLUX expects packed format)
                # Shape: [B, C, H, W] -> [B, (H/2 * W/2), C*4]
                batch_size = latents.shape[0]
                height, width = latents.shape[2], latents.shape[3]
                packed_noisy_latents = rearrange(
                    noisy_latents,
                    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=2, pw=2
                )

                # Generate position IDs for images (2D tensor, no batch dim)
                img_ids = generate_position_ids_flux(
                    height, width,
                    accelerator.device, prompt_embeds.dtype
                )

                # Generate text IDs (2D tensor: [seq_len, 3], no batch dim)
                txt_ids = torch.zeros(
                    prompt_embeds.shape[1], 3,
                    device=accelerator.device, dtype=prompt_embeds.dtype
                )

                # Guidance for FLUX
                guidance = torch.full(
                    (batch_size,), 3.5,
                    device=accelerator.device,
                    dtype=latents.dtype
                )

                # Forward pass through transformer
                # CRITICAL: timestep must be normalized to [0, 1]
                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / num_timesteps,  # Normalize to [0, 1]
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # Unpack prediction using einops
                noise_pred = rearrange(
                    model_pred,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=height // 2,
                    w=width // 2,
                    ph=2, pw=2,
                    c=latents.shape[1]
                )

                # Compute flow matching loss
                # Target is the velocity: noise - latents
                target = (noise - latents).detach()
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                # Gradient clipping for stability (especially in bf16)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

                    # Save checkpoint
                    if global_step % save_steps == 0:
                        save_path = output_path / f"checkpoint-{global_step}"
                        save_path.mkdir(exist_ok=True)

                        unwrapped = accelerator.unwrap_model(transformer)
                        unwrapped.save_pretrained(save_path)
                        print(f"Saved checkpoint to {save_path}")

            if global_step >= max_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        final_path = output_path / "final_lora"
        final_path.mkdir(exist_ok=True)

        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_path)
        print(f"Training complete! LoRA saved to {final_path}")

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train FLUX.1-dev LoRA for art style")

    parser.add_argument(
        "--image_dir", "-i",
        type=str,
        default=str(IMAGES_DIR),
        help="Directory containing training images with captions"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for trained LoRA"
    )
    parser.add_argument(
        "--model_id", "-m",
        type=str,
        default=DEFAULT_FLUX_MODEL,
        help="FLUX model to use (HuggingFace model ID)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=512,
        help="Training resolution"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation", "-g",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps", "-s",
        type=int,
        default=1500,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    train_flux_lora(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_steps,
        lora_rank=args.lora_rank,
        save_steps=args.save_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
