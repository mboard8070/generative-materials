#!/usr/bin/env python3
"""
Text-to-Material Web UI

Gradio interface for generating PBR materials.
"""

import gradio as gr
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "stabilityai/stable-diffusion-2-1"
pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
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
    return pipe


def generate(prompt: str, resolution: int, seed: int, seamless: bool):
    """Generate material from prompt."""
    pipe = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    material_prompt = f"seamless tileable texture of {prompt}, PBR material, game texture, 4k, highly detailed, top down view, flat lighting"
    negative_prompt = "text, watermark, signature, frame, border, 3d render, perspective, objects, items"
    
    generator = torch.Generator(device=device)
    if seed > 0:
        generator.manual_seed(seed)
    
    image = pipe(
        prompt=material_prompt,
        negative_prompt=negative_prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    
    # Simple seamless (Phase 2 will improve this)
    if seamless:
        import numpy as np
        img = np.array(image)
        h, w = img.shape[:2]
        blend_size = w // 8
        result = img.copy()
        for i in range(blend_size):
            alpha = i / blend_size
            result[:, i] = ((1 - alpha) * img[:, w - blend_size + i] + alpha * img[:, i]).astype(np.uint8)
        image = Image.fromarray(result)
    
    # Create tiled preview
    tiled = Image.new('RGB', (resolution * 2, resolution * 2))
    for x in range(2):
        for y in range(2):
            tiled.paste(image, (x * resolution, y * resolution))
    
    return image, tiled


# Build UI
with gr.Blocks(title="Text-to-Material", theme=gr.themes.Soft(primary_hue="violet")) as demo:
    gr.Markdown("""
    # 🎨 Text-to-Material
    Generate PBR textures from text descriptions.
    
    *Phase 1: Albedo map generation. More maps coming soon!*
    """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Material Description",
                placeholder="rusty weathered metal with scratches",
                lines=2,
            )
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=[512, 1024],
                    value=512,
                    label="Resolution",
                )
                seed = gr.Number(
                    value=-1,
                    label="Seed (-1 = random)",
                    precision=0,
                )
            seamless = gr.Checkbox(value=True, label="Seamless tiling")
            generate_btn = gr.Button("Generate Material", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Texture", type="pil")
            tiled_preview = gr.Image(label="Tiled Preview (2x2)", type="pil")
    
    gr.Markdown("""
    ### Tips
    - Be descriptive: "worn leather with stitching" > "leather"
    - Include surface details: scratches, rust, wear, patterns
    - Specify style if needed: photorealistic, stylized, hand-painted
    
    ### Coming Soon
    - Normal map generation
    - Roughness/Metallic maps  
    - Height/Displacement maps
    - Export to Unity/Unreal
    """)
    
    generate_btn.click(
        fn=generate,
        inputs=[prompt, resolution, seed, seamless],
        outputs=[output_image, tiled_preview],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
