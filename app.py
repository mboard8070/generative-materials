#!/usr/bin/env python3
"""
Text-to-Material Web UI

Gradio interface for generating PBR materials with Flux.1-dev.
"""

import gradio as gr
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline
import numpy as np

MODEL_ID = "black-forest-labs/FLUX.1-dev"
pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    print("Loading Flux.1-dev...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("Model loaded!")
    return pipe


def make_seamless(image: Image.Image) -> Image.Image:
    """Make texture tile seamlessly."""
    img = np.array(image)
    h, w = img.shape[:2]
    blend_size = w // 8
    result = img.copy()
    
    for i in range(blend_size):
        alpha = i / blend_size
        result[:, i] = ((1 - alpha) * img[:, w - blend_size + i] + alpha * img[:, i]).astype(np.uint8)
        result[:, w - 1 - i] = ((1 - alpha) * img[:, blend_size - 1 - i] + alpha * img[:, w - 1 - i]).astype(np.uint8)
    
    for i in range(blend_size):
        alpha = i / blend_size
        result[i, :] = ((1 - alpha) * result[h - blend_size + i, :] + alpha * result[i, :]).astype(np.uint8)
        result[h - 1 - i, :] = ((1 - alpha) * result[blend_size - 1 - i, :] + alpha * result[h - 1 - i, :]).astype(np.uint8)
    
    return Image.fromarray(result)


def generate(prompt: str, resolution: int, steps: int, seed: int, seamless: bool):
    """Generate material from prompt using Flux.1-dev."""
    pipe = load_model()
    
    material_prompt = f"seamless tileable PBR texture of {prompt}, game texture, photorealistic material, top-down flat view, even lighting, no shadows, highly detailed surface, 4k quality"
    
    generator = torch.Generator(device="cpu")
    if seed > 0:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
    
    image = pipe(
        prompt=material_prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]
    
    if seamless:
        image = make_seamless(image)
    
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
    Generate PBR textures from text descriptions using **Flux.1-dev**.
    
    *Phase 1: Albedo map generation. Normal/Roughness/Metallic maps coming soon!*
    """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Material Description",
                placeholder="rusty weathered metal with deep scratches and corrosion",
                lines=2,
            )
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=512,
                    label="Resolution",
                )
                steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=28,
                    step=1,
                    label="Steps",
                )
            with gr.Row():
                seed = gr.Number(
                    value=-1,
                    label="Seed (-1 = random)",
                    precision=0,
                )
                seamless = gr.Checkbox(value=True, label="Seamless tiling")
            generate_btn = gr.Button("🎨 Generate Material", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Texture", type="pil")
            tiled_preview = gr.Image(label="Tiled Preview (2x2)", type="pil")
    
    gr.Markdown("""
    ### 💡 Tips for Better Results
    - **Be specific**: "worn brown leather with visible grain and subtle scratches" > "leather"
    - **Include wear details**: scratches, rust, patina, weathering, stains
    - **Specify color**: helps with consistency
    - **Material properties**: shiny, matte, rough, smooth, polished
    
    ### 📋 Example Prompts
    - `brushed stainless steel with circular scratches`
    - `old brick wall with crumbling mortar and moss`
    - `dark walnut wood grain with subtle knots`
    - `rough concrete with cracks and water stains`
    - `hammered copper with green patina`
    - `woven carbon fiber with glossy resin`
    
    ### 🚧 Coming Soon
    - Normal map generation (from albedo or direct)
    - Roughness & Metallic maps
    - Height/Displacement maps
    - One-click export to Unity/Unreal
    """)
    
    generate_btn.click(
        fn=generate,
        inputs=[prompt, resolution, steps, seed, seamless],
        outputs=[output_image, tiled_preview],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
