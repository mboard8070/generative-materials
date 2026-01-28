#!/usr/bin/env python3
"""
Text-to-Material Web UI

Gradio interface for generating PBR materials with Flux.1-dev.
Features prompt builder with surface quality sliders.
"""

import gradio as gr
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline
import numpy as np

from prompt_builder import (
    build_prompt, 
    get_preset, 
    list_presets,
    SURFACE_VOCAB,
    MATERIAL_PRESETS,
)

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


def update_from_preset(preset_name):
    """Update sliders when preset is selected."""
    if not preset_name or preset_name == "Custom":
        return [gr.update()] * 6
    
    values = get_preset(preset_name)
    return [
        gr.update(value=values.get("roughness", 0.5)),
        gr.update(value=values.get("metallic", 0.0)),
        gr.update(value=values.get("age", 0.0)),
        gr.update(value=values.get("moisture", 0.25)),
        gr.update(value=values.get("temperature", 0.5)),
        gr.update(value=values.get("cleanliness", 0.25)),
    ]


def build_prompt_preview(
    base_material, color, roughness, metallic, age, 
    moisture, temperature, cleanliness, extra_details
):
    """Live preview of generated prompt."""
    return build_prompt(
        base_material=base_material,
        roughness=roughness,
        metallic=metallic,
        age=age,
        moisture=moisture,
        temperature=temperature,
        cleanliness=cleanliness,
        color=color,
        extra_details=extra_details,
    )


def generate(
    base_material, color, roughness, metallic, age,
    moisture, temperature, cleanliness, extra_details,
    resolution, steps, seed, seamless
):
    """Generate material from prompt builder settings."""
    pipe = load_model()
    
    # Build prompt from sliders
    prompt = build_prompt(
        base_material=base_material,
        roughness=roughness,
        metallic=metallic,
        age=age,
        moisture=moisture,
        temperature=temperature,
        cleanliness=cleanliness,
        color=color,
        extra_details=extra_details,
    )
    
    generator = torch.Generator(device="cpu")
    if seed > 0:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
    
    image = pipe(
        prompt=prompt,
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
    
    return image, tiled, prompt


def get_roughness_label(value):
    """Get descriptive label for roughness value."""
    if value < 0.2: return "Mirror/Glossy"
    if value < 0.4: return "Satin/Semi-gloss"
    if value < 0.6: return "Matte"
    if value < 0.8: return "Rough"
    return "Very Rough"


# Build UI
with gr.Blocks(title="Text-to-Material", theme=gr.themes.Soft(primary_hue="violet")) as demo:
    gr.Markdown("""
    # 🎨 Text-to-Material
    Generate PBR textures using surface property sliders + **Flux.1-dev**
    """)
    
    with gr.Row():
        # Left column - Controls
        with gr.Column(scale=1):
            gr.Markdown("### 📦 Material Base")
            
            preset_dropdown = gr.Dropdown(
                choices=["Custom"] + list_presets(),
                value="Custom",
                label="Preset",
            )
            
            with gr.Row():
                base_material = gr.Textbox(
                    label="Material Type",
                    placeholder="steel, wood, concrete...",
                    value="steel",
                )
                color = gr.Textbox(
                    label="Color",
                    placeholder="dark gray, rusty orange...",
                    value="",
                )
            
            gr.Markdown("### 🎚️ Surface Properties")
            
            roughness = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.05,
                label="Roughness (0=mirror, 1=rough)",
            )
            metallic = gr.Slider(
                minimum=0, maximum=1, value=0.0, step=0.05,
                label="Metallic (0=dielectric, 1=metal)",
            )
            age = gr.Slider(
                minimum=0, maximum=1, value=0.0, step=0.05,
                label="Age/Wear (0=new, 1=weathered)",
            )
            
            with gr.Accordion("🔧 Advanced Properties", open=False):
                moisture = gr.Slider(
                    minimum=0, maximum=1, value=0.25, step=0.05,
                    label="Moisture (0=dry, 1=wet)",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.05,
                    label="Temperature (0=frozen, 1=hot)",
                )
                cleanliness = gr.Slider(
                    minimum=0, maximum=1, value=0.25, step=0.05,
                    label="Dirt (0=clean, 1=filthy)",
                )
                extra_details = gr.Textbox(
                    label="Extra Details",
                    placeholder="scratches, dents, patterns...",
                    value="",
                )
            
            gr.Markdown("### ⚙️ Generation Settings")
            
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=512,
                    label="Resolution",
                )
                steps = gr.Slider(
                    minimum=10, maximum=50, value=28, step=1,
                    label="Steps",
                )
            
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                seamless = gr.Checkbox(value=True, label="Seamless")
            
            generate_btn = gr.Button("🎨 Generate Material", variant="primary", size="lg")
        
        # Right column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Material")
            output_image = gr.Image(label="Texture", type="pil")
            tiled_preview = gr.Image(label="Tiled Preview (2x2)", type="pil")
            
            gr.Markdown("### 📝 Generated Prompt")
            prompt_preview = gr.Textbox(
                label="Prompt (auto-generated)",
                lines=3,
                interactive=False,
            )
    
    # Preset updates sliders
    preset_dropdown.change(
        fn=update_from_preset,
        inputs=[preset_dropdown],
        outputs=[roughness, metallic, age, moisture, temperature, cleanliness],
    )
    
    # Live prompt preview
    prompt_inputs = [
        base_material, color, roughness, metallic, age,
        moisture, temperature, cleanliness, extra_details
    ]
    
    for inp in prompt_inputs:
        inp.change(
            fn=build_prompt_preview,
            inputs=prompt_inputs,
            outputs=[prompt_preview],
        )
    
    # Generate
    generate_btn.click(
        fn=generate,
        inputs=prompt_inputs + [resolution, steps, seed, seamless],
        outputs=[output_image, tiled_preview, prompt_preview],
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Presets** give you starting points — tweak from there
    - **Roughness** is the most important slider for PBR feel
    - **Age** adds wear, scratches, patina automatically
    - Combine **Moisture + Dirt** for realistic outdoor surfaces
    
    ### 🚧 Coming Soon
    - Normal map extraction from generated albedo
    - Roughness/Metallic map inference
    - AO generation
    - Direct export to UE5/Unity
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
