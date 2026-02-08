#!/usr/bin/env python3
"""
Download PBR materials from Poly Haven for training data.
https://polyhaven.com/textures

All assets are CC0 (public domain).
"""

import requests
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

API_BASE = "https://api.polyhaven.com"
OUTPUT_DIR = Path("data/raw/polyhaven")

# Map types we need for PBR (API uses capitalized keys)
MAP_TYPES = {
    "diffuse": ["Diffuse"],
    "normal": ["nor_gl", "Normal"],
    "roughness": ["Rough", "Roughness"],
    "metallic": ["Metal", "Metallic"],
    "displacement": ["Displacement", "Height"],
    "ao": ["AO"],
}


def get_texture_list():
    """Get list of all textures from Poly Haven."""
    print("Fetching texture list from Poly Haven...")
    response = requests.get(f"{API_BASE}/assets?t=textures")
    response.raise_for_status()
    return response.json()


def get_texture_info(texture_id: str):
    """Get detailed info for a texture."""
    response = requests.get(f"{API_BASE}/files/{texture_id}")
    if response.status_code == 200:
        return response.json()
    return None


def download_file(url: str, path: Path, retries: int = 3):
    """Download a file with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def download_texture(texture_id: str, resolution: str = "1k"):
    """Download a texture with all its maps."""
    output_path = OUTPUT_DIR / texture_id
    
    # Check if already downloaded
    if (output_path / "diffuse.png").exists():
        return {"id": texture_id, "status": "skipped", "maps": []}
    
    # Get file info
    info = get_texture_info(texture_id)
    if not info:
        return {"id": texture_id, "status": "error", "maps": []}
    
    output_path.mkdir(parents=True, exist_ok=True)
    downloaded_maps = []
    
    # Try to find each map type
    # API structure: info[MapType][resolution][format][url]
    for map_name, possible_keys in MAP_TYPES.items():
        for key in possible_keys:
            if key not in info:
                continue
            # Look in different resolutions
            for res_key in [resolution, "1k", "2k"]:
                try:
                    if res_key in info[key]:
                        # Prefer jpg (smaller), fall back to png
                        for fmt in ["jpg", "png"]:
                            if fmt in info[key][res_key]:
                                url = info[key][res_key][fmt]["url"]
                                file_path = output_path / f"{map_name}.png"
                                if download_file(url, file_path):
                                    downloaded_maps.append(map_name)
                                break
                        if map_name in downloaded_maps:
                            break
                except (KeyError, TypeError):
                    continue
            if map_name in downloaded_maps:
                break
    
    # Save metadata
    meta = {
        "id": texture_id,
        "name": texture_id.replace("_", " ").title(),
        "maps": downloaded_maps,
        "resolution": resolution,
        "source": "polyhaven",
        "license": "CC0",
    }
    
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return {"id": texture_id, "status": "ok", "maps": downloaded_maps}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get texture list
    textures = get_texture_list()
    texture_ids = list(textures.keys())
    
    print(f"Found {len(texture_ids)} textures on Poly Haven")
    print(f"Downloading to {OUTPUT_DIR.absolute()}\n")
    
    # Download with progress bar
    results = {"ok": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_texture, tid): tid for tid in texture_ids}
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results[result["status"]] += 1
                pbar.set_postfix(results)
                pbar.update(1)
    
    print(f"\n=== Download Complete ===")
    print(f"  Downloaded: {results['ok']}")
    print(f"  Skipped:    {results['skipped']}")
    print(f"  Errors:     {results['error']}")
    print(f"\nData saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
