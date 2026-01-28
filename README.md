# 🎨 Text-to-Material

Generate PBR materials from text descriptions using AI.

> "rusty weathered metal with scratches" → Full PBR material set

## Features (Planned)

- **Text-to-PBR** — Generate complete material sets:
  - Albedo/Diffuse
  - Normal map
  - Roughness
  - Metallic
  - Height/Displacement
  - AO (Ambient Occlusion)

- **Seamless tiling** — Output tileable textures ready for game engines
- **Resolution control** — 512, 1K, 2K, 4K output options
- **Style presets** — Photorealistic, stylized, hand-painted
- **Export formats** — PNG, EXR, Unreal/Unity ready

## Tech Stack

- **Diffusers** — Stable Diffusion backbone
- **PyTorch** — Model training/inference
- **ControlNet** — For multi-map generation
- **Gradio** — Web UI for testing

## Approach

### Phase 1: Single Map Generation
- Fine-tune SD on material textures
- Ensure seamless tiling via latent space tricks
- Output albedo maps from text prompts

### Phase 2: Multi-Map Pipeline
- Train/fine-tune for normal, roughness, metallic maps
- Either: multi-head output OR sequential generation
- Investigate ControlNet for map-to-map consistency

### Phase 3: Material Graph (Stretch)
- Generate Substance-style node graphs
- Or: HLSL/GLSL shader code generation

## Datasets

Potential training data sources:
- [Poly Haven](https://polyhaven.com/textures) — CC0 PBR materials
- [AmbientCG](https://ambientcg.com/) — CC0 materials
- [3D Textures](https://3dtextures.me/) — Free PBR textures
- [Quixel Megascans](https://quixel.com/megascans) — High quality (licensing TBD)

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the prototype
python generate.py --prompt "worn leather with stitching"
```

## Project Structure

```
text-to-material/
├── data/
│   ├── raw/              # Downloaded material datasets
│   └── processed/        # Prepared training data
├── models/               # Saved model checkpoints
├── outputs/              # Generated materials
├── scripts/
│   ├── download_data.py  # Dataset downloaders
│   ├── prepare_data.py   # Data preprocessing
│   └── train.py          # Training script
├── generate.py           # Inference script
├── app.py                # Gradio web UI
└── requirements.txt
```

## References

- [MatFuse: Controllable Material Generation](https://arxiv.org/abs/2308.11408)
- [Text2Tex: Text-driven Texture Synthesis](https://arxiv.org/abs/2303.11396)
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800)

## License

MIT
