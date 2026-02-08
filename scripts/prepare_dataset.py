#!/usr/bin/env python3
"""
Prepare downloaded materials for LoRA training.

Creates captioned dataset in the format expected by ai-toolkit / diffusers.
"""

import json
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/training")

# Caption templates based on material properties
CAPTION_TEMPLATE = "seamless tileable PBR texture of {description}, game texture, photorealistic material, top-down view, even lighting, highly detailed"


def clean_name(name: str) -> str:
    """Convert asset ID to readable description."""
    # Remove common prefixes/suffixes
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"_\d+$", "", name)
    
    # Replace underscores and clean up
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    
    return name.lower()


def infer_properties(name: str, tags: list = None) -> dict:
    """Infer material properties from name and tags."""
    name_lower = name.lower()
    tags_lower = [t.lower() for t in (tags or [])]
    all_text = name_lower + " " + " ".join(tags_lower)
    
    properties = []
    
    # Roughness indicators
    if any(w in all_text for w in ["polished", "glossy", "shiny", "mirror"]):
        properties.append("polished")
    elif any(w in all_text for w in ["rough", "raw", "coarse"]):
        properties.append("rough")
    elif any(w in all_text for w in ["matte", "flat"]):
        properties.append("matte")
    
    # Age indicators
    if any(w in all_text for w in ["old", "aged", "weathered", "worn", "ancient"]):
        properties.append("weathered")
    elif any(w in all_text for w in ["rusted", "rust", "corroded"]):
        properties.append("rusted")
    elif any(w in all_text for w in ["new", "clean", "pristine"]):
        properties.append("clean")
    
    # Material type hints
    if any(w in all_text for w in ["metal", "steel", "iron", "copper", "brass", "aluminum"]):
        properties.append("metallic")
    if any(w in all_text for w in ["wood", "timber", "plank", "oak", "pine"]):
        properties.append("wooden")
    if any(w in all_text for w in ["stone", "rock", "granite", "marble"]):
        properties.append("stone")
    if any(w in all_text for w in ["concrete", "cement"]):
        properties.append("concrete")
    if any(w in all_text for w in ["brick", "tile"]):
        properties.append("brick")
    if any(w in all_text for w in ["fabric", "cloth", "textile", "leather"]):
        properties.append("fabric")
    
    return properties


def generate_caption(name: str, tags: list = None) -> str:
    """Generate a training caption for a material."""
    clean = clean_name(name)
    properties = infer_properties(name, tags)
    
    # Build description
    if properties:
        description = f"{' '.join(properties)} {clean}"
    else:
        description = clean
    
    return CAPTION_TEMPLATE.format(description=description)


def process_material(material_path: Path, output_dir: Path) -> dict:
    """Process a single material directory."""
    
    # Load metadata if exists
    meta_path = material_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"name": material_path.name, "tags": []}
    
    # Find diffuse/albedo map
    diffuse_path = None
    for name in ["diffuse.png", "albedo.png", "color.png"]:
        p = material_path / name
        if p.exists():
            diffuse_path = p
            break
    
    if not diffuse_path:
        return {"status": "skip", "reason": "no diffuse map"}
    
    # Generate caption
    caption = generate_caption(
        meta.get("name", material_path.name),
        meta.get("tags", [])
    )
    
    # Create output
    output_name = material_path.name
    img_output = output_dir / f"{output_name}.png"
    
    # Copy/convert image
    try:
        img = Image.open(diffuse_path)
        
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize if too large (training typically uses 512 or 1024)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Ensure square
        if img.size[0] != img.size[1]:
            size = min(img.size)
            left = (img.size[0] - size) // 2
            top = (img.size[1] - size) // 2
            img = img.crop((left, top, left + size, top + size))
        
        img.save(img_output, "PNG")
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    
    # Save caption
    caption_output = output_dir / f"{output_name}.txt"
    caption_output.write_text(caption)
    
    return {
        "status": "ok",
        "image": str(img_output),
        "caption": caption,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare materials for LoRA training")
    parser.add_argument("--sources", nargs="+", default=["polyhaven", "ambientcg"],
                        help="Data sources to process")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all material directories
    material_dirs = []
    for source in args.sources:
        source_dir = RAW_DIR / source
        if source_dir.exists():
            material_dirs.extend(list(source_dir.iterdir()))
    
    print(f"Found {len(material_dirs)} materials to process")
    print(f"Output: {OUTPUT_DIR.absolute()}\n")
    
    # Process materials
    results = {"ok": 0, "skip": 0, "error": 0}
    
    for mat_dir in tqdm(material_dirs, desc="Processing"):
        if not mat_dir.is_dir():
            continue
        
        result = process_material(mat_dir, OUTPUT_DIR)
        results[result["status"]] += 1
    
    print(f"\n=== Dataset Prepared ===")
    print(f"  Processed: {results['ok']}")
    print(f"  Skipped:   {results['skip']}")
    print(f"  Errors:    {results['error']}")
    print(f"\nTraining data saved to: {OUTPUT_DIR.absolute()}")
    print(f"Total images: {len(list(OUTPUT_DIR.glob('*.png')))}")
    
    # Create dataset info
    info = {
        "num_images": len(list(OUTPUT_DIR.glob("*.png"))),
        "sources": args.sources,
        "format": "image + caption txt",
    }
    
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
