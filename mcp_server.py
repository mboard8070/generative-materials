#!/usr/bin/env python3
"""Surfaced MCP Server.

Exposes Surfaced's PBR material generation capabilities as MCP tools.
Proxies all calls to the Surfaced FastAPI backend.

Usage:
    # stdio transport (for Claude Code / Claude Desktop)
    python mcp_server.py

    # HTTP transport (for remote access over Tailscale)
    python mcp_server.py --http --port 8002
"""

import argparse
import httpx
from mcp.server.fastmcp import FastMCP

SURFACED_URL = "http://localhost:8001"

mcp = FastMCP(
    "Surfaced",
    instructions="AI-powered PBR material generation. Generate, extract, paint, and manage physically-based rendering materials.",
)

client = httpx.Client(base_url=SURFACED_URL, timeout=120.0)


def _post(path: str, json: dict) -> dict:
    """POST to Surfaced API and return JSON response."""
    r = client.post(path, json=json)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict | list:
    """GET from Surfaced API and return JSON response."""
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> dict:
    """DELETE from Surfaced API."""
    r = client.delete(path)
    r.raise_for_status()
    return r.json()


def _format_maps(data: dict) -> str:
    """Format a generation result into readable text with URLs."""
    lines = []
    if data.get("texture_id"):
        lines.append(f"Texture ID: {data['texture_id']}")
    if data.get("seed"):
        lines.append(f"Seed: {data['seed']}")
    if data.get("engine"):
        lines.append(f"Engine: {data['engine']}")

    lines.append("\nMaps:")
    map_keys = [
        ("texture_url", "Base Color"),
        ("normal_map_url", "Normal"),
        ("roughness_map_url", "Roughness"),
        ("metallic_map_url", "Metalness"),
        ("height_map_url", "Height"),
        ("emissive_map_url", "Emissive"),
        ("ao_map_url", "AO"),
    ]
    for key, label in map_keys:
        if data.get(key):
            lines.append(f"  {label}: {SURFACED_URL}{data[key]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_material(
    prompt: str,
    engine: str = "patina",
    upscale_factor: int = 0,
    enable_prompt_expansion: bool = True,
    seed: int | None = None,
) -> str:
    """Generate a PBR material from a text description.

    Creates a complete set of PBR texture maps (base color, normal, roughness,
    metalness, height) from a text prompt. Use descriptive material names like
    'weathered copper with green patina' or 'polished marble with gold veins'.

    Args:
        prompt: Material description (e.g. 'rusted steel', 'oak wood planks')
        engine: 'patina' (fast cloud, ~15s, ~$0.08) or 'flux' (local, ~45s, free)
        upscale_factor: 0 (1024px), 2 (2048px), or 4 (4096px) — patina only
        enable_prompt_expansion: Auto-enhance terse prompts — patina only
        seed: Random seed for reproducibility
    """
    body: dict = {
        "prompt": prompt,
        "engine": engine,
        "tiling_mode": "none",
        "upscale_factor": upscale_factor,
        "enable_prompt_expansion": enable_prompt_expansion,
    }
    if seed is not None:
        body["seed"] = seed

    result = _post("/generate", body)
    return _format_maps(result)


@mcp.tool()
def image_to_pbr(image_url: str, seed: int | None = None) -> str:
    """Generate PBR maps from an existing image.

    Upload a photo of a surface (brick wall, wood floor, fabric, etc.) and
    get back a complete set of PBR material maps derived from it.

    Args:
        image_url: URL of the source image
        seed: Random seed for reproducibility
    """
    import urllib.parse

    form_data = {"image_url": image_url}
    if seed is not None:
        form_data["seed"] = str(seed)

    r = client.post("/image-to-pbr", data=form_data)
    r.raise_for_status()
    return _format_maps(r.json())


@mcp.tool()
def extract_material(
    image_url: str,
    label: str,
    upscale_factor: int = 0,
    seed: int | None = None,
) -> str:
    """Extract a specific material from a scene photo.

    Give it a photo of a room or scene and name the material you want
    (e.g. 'brick wall', 'hardwood floor'). It identifies that surface,
    flattens it, makes it tileable, and returns PBR maps.

    Args:
        image_url: URL of the scene photo
        label: Name of the material to extract (e.g. 'concrete floor')
        upscale_factor: 0, 2, or 4
        seed: Random seed
    """
    form_data = {"image_url": image_url, "label": label, "upscale_factor": str(upscale_factor)}
    if seed is not None:
        form_data["seed"] = str(seed)

    r = client.post("/extract-material", data=form_data)
    r.raise_for_status()
    return _format_maps(r.json())


@mcp.tool()
def save_material(
    texture_id: str,
    name: str,
    prompt: str = "",
    engine: str = "",
    seed: int | None = None,
) -> str:
    """Save a generated material to the persistent library.

    Call this after generate_material to keep the material for later use.
    Materials in the library persist across server restarts.

    Args:
        texture_id: The texture_id from a generation result
        name: Human-readable name for the material
        prompt: The prompt used to generate it
        engine: Which engine was used
        seed: The seed used
    """
    result = _post("/library/save", {
        "texture_id": texture_id,
        "name": name,
        "prompt": prompt,
        "engine": engine,
        "seed": seed,
    })
    return f"Saved '{result['name']}' (id: {result['id']}) with {len(result.get('maps', {}))} maps.\nCreated: {result.get('created')}"


@mcp.tool()
def list_library() -> str:
    """List all saved materials in the library.

    Returns names, prompts, engines, and thumbnails for all saved materials.
    """
    items = _get("/library")
    if not items:
        return "Library is empty."

    lines = [f"Library: {len(items)} materials\n"]
    for item in items:
        lines.append(f"  {item['id']}  {item['name']}")
        if item.get("prompt"):
            lines.append(f"         prompt: {item['prompt']}")
        if item.get("engine"):
            lines.append(f"         engine: {item['engine']}  seed: {item.get('seed')}")
        lines.append(f"         created: {item.get('created')}")
        if item.get("thumbnail_url"):
            lines.append(f"         thumbnail: {SURFACED_URL}{item['thumbnail_url']}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def load_material(material_id: str) -> str:
    """Load a saved material from the library by its ID.

    Returns the map URLs so the material can be used or further edited.

    Args:
        material_id: The ID from list_library
    """
    result = _get(f"/library/{material_id}")
    name = result.get("name", material_id)
    prompt = result.get("prompt", "")
    header = f"Loaded '{name}'"
    if prompt:
        header += f" (prompt: {prompt})"
    return header + "\n" + _format_maps(result)


@mcp.tool()
def delete_material(material_id: str) -> str:
    """Delete a material from the library.

    Args:
        material_id: The ID of the material to delete
    """
    _delete(f"/library/{material_id}")
    return f"Deleted material {material_id}"


@mcp.tool()
def adjust_height(
    texture_id: str,
    contrast: float = 1.0,
    brightness: float = 0.0,
    invert: bool = False,
    blur_radius: float = 0.0,
) -> str:
    """Adjust the height/displacement map of a generated material.

    Post-process the height map to control displacement depth, invert
    raised/recessed areas, blur for smoother displacement, or adjust contrast.

    Args:
        texture_id: The texture_id from a generation result
        contrast: Height contrast multiplier (1.0 = unchanged, 2.0 = double)
        brightness: Shift the height values (-0.5 to 0.5)
        invert: Flip raised and recessed areas
        blur_radius: Gaussian blur sigma (0 = sharp, 10 = very smooth)
    """
    result = _post("/adjust-height", {
        "texture_id": texture_id,
        "contrast": contrast,
        "brightness": brightness,
        "invert": invert,
        "blur_radius": blur_radius,
    })
    return f"Adjusted height map: {SURFACED_URL}{result['height_map_url']}"


@mcp.tool()
def surfaced_status() -> str:
    """Check the status of the Surfaced backend.

    Returns whether the PATINA (cloud) and Flux (local) engines are ready.
    """
    result = _get("/status")
    lines = [
        f"PATINA ready: {result.get('patina_ready', False)}",
        f"Flux ready: {result.get('ready', False)}",
    ]
    if result.get("loading"):
        lines.append(f"Loading: {result.get('step')} ({result.get('progress')}%)")
    return "\n".join(lines)


@mcp.tool()
def generate_and_save(
    prompt: str,
    name: str = "",
    engine: str = "patina",
    upscale_factor: int = 0,
) -> str:
    """Generate a material and immediately save it to the library.

    Convenience tool that combines generate_material + save_material.

    Args:
        prompt: Material description
        name: Library name (defaults to prompt)
        engine: 'patina' or 'flux'
        upscale_factor: 0, 2, or 4
    """
    gen_result = _post("/generate", {
        "prompt": prompt,
        "engine": engine,
        "tiling_mode": "none",
        "upscale_factor": upscale_factor,
        "enable_prompt_expansion": True,
    })

    save_name = name or prompt[:50]
    save_result = _post("/library/save", {
        "texture_id": gen_result["texture_id"],
        "name": save_name,
        "prompt": prompt,
        "engine": engine,
        "seed": gen_result.get("seed"),
    })

    return (
        f"Generated and saved '{save_result['name']}' (id: {save_result['id']})\n"
        + _format_maps(gen_result)
    )


@mcp.tool()
def batch_generate(
    prompts: list[str],
    engine: str = "patina",
    save_to_library: bool = True,
) -> str:
    """Generate multiple materials in sequence.

    Takes a list of material descriptions and generates each one.
    Optionally saves all to the library.

    Args:
        prompts: List of material descriptions
        engine: 'patina' or 'flux'
        save_to_library: Whether to save each to the library
    """
    results = []
    for i, prompt in enumerate(prompts):
        gen_result = _post("/generate", {
            "prompt": prompt,
            "engine": engine,
            "tiling_mode": "none",
            "enable_prompt_expansion": True,
        })

        status = f"[{i+1}/{len(prompts)}] {prompt}"
        if save_to_library:
            save_result = _post("/library/save", {
                "texture_id": gen_result["texture_id"],
                "name": prompt[:50],
                "prompt": prompt,
                "engine": engine,
                "seed": gen_result.get("seed"),
            })
            status += f" → saved as {save_result['id']}"
        else:
            status += f" → {gen_result['texture_id']}"

        results.append(status)

    return f"Generated {len(results)} materials:\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surfaced MCP Server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server instead of stdio")
    parser.add_argument("--port", type=int, default=8002, help="HTTP port (default: 8002)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default: 0.0.0.0)")
    args = parser.parse_args()

    if args.http:
        # Reconfigure with HTTP host/port
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
