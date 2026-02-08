#!/usr/bin/env python3
"""
Download PBR materials from AmbientCG for training data.
https://ambientcg.com/

All assets are CC0 (public domain).
"""

import requests
import zipfile
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re

API_BASE = "https://ambientcg.com/api/v2"
OUTPUT_DIR = Path("data/raw/ambientcg")


def get_material_list(limit: int = None):
    """Get list of all materials from AmbientCG."""
    print("Fetching material list from AmbientCG...")
    
    materials = []
    offset = 0
    per_page = 100
    
    while True:
        params = {
            "type": "Material",
            "include": "downloadData",
            "limit": per_page,
            "offset": offset,
        }
        
        response = requests.get(f"{API_BASE}/full_json", params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("foundAssets"):
            break
            
        materials.extend(data["foundAssets"])
        offset += per_page
        
        if limit and len(materials) >= limit:
            materials = materials[:limit]
            break
            
        if offset >= data.get("numberOfResults", 0):
            break
    
    return materials


def download_material(material: dict, resolution: str = "1K-PNG"):
    """Download a material with all its maps."""
    asset_id = material.get("assetId", "")
    output_path = OUTPUT_DIR / asset_id
    
    # Check if already downloaded
    if (output_path / "diffuse.png").exists():
        return {"id": asset_id, "status": "skipped"}
    
    # Find the download URL for requested resolution
    downloads = material.get("downloadFolders", {}).get("default", {}).get("downloadFiletypeCategories", {})
    
    zip_url = None
    for category in ["zip", "png"]:
        if category in downloads:
            for download in downloads[category].get("downloads", []):
                if resolution in download.get("attribute", ""):
                    zip_url = download.get("fullDownloadPath")
                    break
        if zip_url:
            break
    
    if not zip_url:
        # Try any PNG resolution
        for category in ["zip", "png"]:
            if category in downloads:
                for download in downloads[category].get("downloads", []):
                    if "PNG" in download.get("attribute", ""):
                        zip_url = download.get("fullDownloadPath")
                        break
            if zip_url:
                break
    
    if not zip_url:
        return {"id": asset_id, "status": "error", "reason": "no download URL"}
    
    try:
        # Download zip
        response = requests.get(zip_url, timeout=120)
        if response.status_code != 200:
            return {"id": asset_id, "status": "error", "reason": f"HTTP {response.status_code}"}
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract maps from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            downloaded_maps = []
            
            for name in zf.namelist():
                name_lower = name.lower()
                
                # Map filename patterns to our standard names
                if "color" in name_lower or "diffuse" in name_lower or "albedo" in name_lower:
                    out_name = "diffuse.png"
                elif "normal" in name_lower and "gl" in name_lower:
                    out_name = "normal.png"
                elif "normal" in name_lower:
                    out_name = "normal.png"
                elif "roughness" in name_lower:
                    out_name = "roughness.png"
                elif "metallic" in name_lower or "metalness" in name_lower:
                    out_name = "metallic.png"
                elif "displacement" in name_lower or "height" in name_lower:
                    out_name = "displacement.png"
                elif "ao" in name_lower or "ambientocclusion" in name_lower:
                    out_name = "ao.png"
                else:
                    continue
                
                # Extract file
                data = zf.read(name)
                (output_path / out_name).write_bytes(data)
                downloaded_maps.append(out_name.replace(".png", ""))
        
        # Save metadata
        meta = {
            "id": asset_id,
            "name": material.get("displayName", asset_id),
            "tags": material.get("tags", []),
            "maps": downloaded_maps,
            "source": "ambientcg",
            "license": "CC0",
        }
        
        with open(output_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        return {"id": asset_id, "status": "ok", "maps": downloaded_maps}
        
    except Exception as e:
        return {"id": asset_id, "status": "error", "reason": str(e)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PBR materials from AmbientCG")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of materials")
    parser.add_argument("--resolution", type=str, default="1K-PNG", help="Resolution to download")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get material list
    materials = get_material_list(limit=args.limit)
    
    print(f"Found {len(materials)} materials on AmbientCG")
    print(f"Downloading to {OUTPUT_DIR.absolute()}\n")
    
    # Download with progress bar
    results = {"ok": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_material, mat, args.resolution): mat for mat in materials}
        
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
