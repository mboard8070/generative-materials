#!/usr/bin/env python3
"""
Download PBR materials from Poly Haven for training data.
https://polyhaven.com/textures

All assets are CC0 (public domain).
"""

import requests
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

API_BASE = "https://api.polyhaven.com/assets"
DOWNLOAD_BASE = "https://dl.polyhaven.org/file/ph-assets/Textures"

OUTPUT_DIR = Path("data/raw/polyhaven")


def get_texture_list():
    """Get list of all textures from Poly Haven."""
    print("Fetching texture list...")
    response = requests.get(f"{API_BASE}?t=textures")
    response.raise_for_status()
    return response.json()


def download_texture(name: str, resolution: str = "1k"):
    """Download a single texture with all its maps."""
    output_path = OUTPUT_DIR / name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map types we want
    map_types = ["diff", "nor_gl", "rough", "metal", "disp"]
    
    for map_type in map_types:
        url = f"{DOWNLOAD_BASE}/{resolution}/{name}/{name}_{map_type}_{resolution}.png"
        file_path = output_path / f"{map_type}.png"
        
        if file_path.exists():
            continue
            
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                file_path.write_bytes(response.content)
        except Exception as e:
            pass  # Some textures don't have all map types


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    textures = get_texture_list()
    print(f"Found {len(textures)} textures")
    
    # Download with progress bar
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(download_texture, textures.keys()),
            total=len(textures),
            desc="Downloading"
        ))
    
    print(f"\nDownloaded to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
