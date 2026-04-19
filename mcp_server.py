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


def _patch(path: str, json: dict) -> dict:
    """PATCH the Surfaced API and return JSON response."""
    r = client.patch(path, json=json)
    r.raise_for_status()
    return r.json()


def _format_layer(l: dict) -> str:
    """One-line summary of a layer."""
    vis = "visible" if l.get("visible", True) else "hidden"
    parts = [
        f"  {l['id']}  {l.get('name', '')}",
        f"type={l.get('type', 'material')}",
        f"blend={l.get('blendMode', 'normal')}",
        f"opacity={l.get('opacity', 1.0):.2f}",
        vis,
    ]
    if l.get("materialPrompt"):
        parts.append(f"prompt='{l['materialPrompt']}'")
    return "  ".join(parts)


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
# Layer stack
# ---------------------------------------------------------------------------

@mcp.tool()
def list_layers() -> str:
    """List the current layer stack (bottom first, top last).

    Each layer has: id, name, type (material/solid/noise), blendMode, opacity,
    visibility, and optionally a material prompt/maps.
    """
    items = _get("/layers")
    if not items:
        return "Layer stack is empty."
    lines = [f"Layer stack: {len(items)} layers (bottom → top)"]
    for l in items:
        lines.append(_format_layer(l))
    return "\n".join(lines)


@mcp.tool()
def add_layer(
    name: str = "",
    layer_type: str = "material",
    blend_mode: str = "normal",
    opacity: float = 1.0,
    visible: bool = True,
    material_prompt: str = "",
    color: str = "#888888",
    noise_type: str = "perlin",
    noise_scale: float = 4.0,
) -> str:
    """Append a new layer to the top of the stack.

    Args:
        name: Layer name (auto-generated if empty)
        layer_type: 'material' (PBR material), 'solid' (flat color), or 'noise' (procedural)
        blend_mode: normal, multiply, screen, overlay, soft-light, add, subtract, divide,
                    darken, lighten, color-dodge, color-burn, difference
        opacity: 0.0 to 1.0
        visible: whether the layer is rendered
        material_prompt: for material layers, the prompt describing the material
        color: hex color for solid layers (e.g. '#ff0000')
        noise_type: 'perlin' or 'voronoi' for noise layers
        noise_scale: noise tiling scale (1 = large, 10 = fine)
    """
    body: dict = {
        "type": layer_type,
        "blendMode": blend_mode,
        "opacity": opacity,
        "visible": visible,
        "materialPrompt": material_prompt,
        "color": color,
        "noiseType": noise_type,
        "noiseScale": noise_scale,
    }
    if name:
        body["name"] = name
    result = _post("/layers", body)
    return f"Added layer:\n{_format_layer(result)}"


@mcp.tool()
def update_layer(
    layer_id: str,
    name: str | None = None,
    blend_mode: str | None = None,
    opacity: float | None = None,
    visible: bool | None = None,
    material_prompt: str | None = None,
    color: str | None = None,
) -> str:
    """Modify a layer's properties. Only provided fields are updated.

    Args:
        layer_id: The layer ID (from list_layers)
        name: New layer name
        blend_mode: New blend mode
        opacity: 0.0 to 1.0
        visible: visibility toggle
        material_prompt: New material prompt (will not auto-regenerate)
        color: New hex color for solid layers
    """
    updates: dict = {}
    if name is not None:
        updates["name"] = name
    if blend_mode is not None:
        updates["blendMode"] = blend_mode
    if opacity is not None:
        updates["opacity"] = opacity
    if visible is not None:
        updates["visible"] = visible
    if material_prompt is not None:
        updates["materialPrompt"] = material_prompt
    if color is not None:
        updates["color"] = color
    if not updates:
        return "No updates provided."
    result = _patch(f"/layers/{layer_id}", updates)
    return f"Updated layer:\n{_format_layer(result)}"


@mcp.tool()
def delete_layer(layer_id: str) -> str:
    """Remove a layer from the stack.

    Args:
        layer_id: The ID of the layer to remove
    """
    _delete(f"/layers/{layer_id}")
    return f"Deleted layer {layer_id}"


@mcp.tool()
def duplicate_layer(layer_id: str) -> str:
    """Duplicate a layer. The copy is inserted directly above the source.

    Args:
        layer_id: The ID of the layer to duplicate
    """
    result = _post(f"/layers/{layer_id}/duplicate", {})
    return f"Duplicated layer:\n{_format_layer(result)}"


@mcp.tool()
def move_layer(layer_id: str, direction: str) -> str:
    """Move a layer up or down by one position.

    Args:
        layer_id: The ID of the layer to move
        direction: 'up' (toward top of stack) or 'down' (toward bottom)
    """
    if direction not in ("up", "down"):
        return "direction must be 'up' or 'down'"
    items = _get("/layers")
    ids = [l["id"] for l in items]
    try:
        idx = ids.index(layer_id)
    except ValueError:
        return f"Layer {layer_id} not found"
    target = idx + 1 if direction == "up" else idx - 1
    if target < 0 or target >= len(ids):
        return f"Cannot move {direction}: already at the {'top' if direction == 'up' else 'bottom'}."
    ids[idx], ids[target] = ids[target], ids[idx]
    _post("/layers/reorder", {"order": ids})
    return f"Moved layer {layer_id} {direction}."


@mcp.tool()
def reorder_layers(layer_ids: list[str]) -> str:
    """Set the full layer stack order (bottom first, top last).

    Args:
        layer_ids: All existing layer IDs in the desired new order
    """
    result = _post("/layers/reorder", {"order": layer_ids})
    lines = [f"Reordered stack ({len(result)} layers, bottom → top):"]
    for l in result:
        lines.append(_format_layer(l))
    return "\n".join(lines)


@mcp.tool()
def clear_layers() -> str:
    """Remove all layers from the stack."""
    _delete("/layers")
    return "Cleared all layers."


# ---------------------------------------------------------------------------
# Additional generation / export tools
# ---------------------------------------------------------------------------

@mcp.tool()
def edit_material(
    image_url: str,
    prompt: str,
    strength: float = 0.7,
    tiling_mode: str = "multiscale",
    seed: int | None = None,
) -> str:
    """Edit an existing texture using img2img.

    Applies a prompted change to an existing texture while preserving overall
    structure. Lower strength = subtle changes; higher = more dramatic.

    Args:
        image_url: URL of the existing albedo (e.g. /outputs/abc.png or full URL)
        prompt: Describe the edit (e.g. 'add rust spots', 'make it more blue')
        strength: 0.1 (subtle) to 1.0 (dramatic)
        tiling_mode: 'none', 'basic', 'multipass', 'multiscale'
        seed: Random seed
    """
    body: dict = {
        "image_url": image_url,
        "prompt": prompt,
        "strength": strength,
        "tiling_mode": tiling_mode,
    }
    if seed is not None:
        body["seed"] = seed
    result = _post("/edit", body)
    return _format_maps(result)


@mcp.tool()
def composite_pbr(image_data_url: str) -> str:
    """Generate PBR maps from a composited layer stack image via img2img.

    Takes a base64-encoded PNG (data URL) of a flattened albedo and generates
    coherent PBR maps for it at low strength.

    Args:
        image_data_url: data URL string, e.g. 'data:image/png;base64,iVBOR...'
    """
    result = _post("/composite-pbr", {"image_data": image_data_url})
    return _format_maps(result)


@mcp.tool()
def export_unreal(
    texture_url: str,
    roughness: float = 0.5,
    metalness: float = 0.0,
    save_to: str = "",
) -> str:
    """Export the current material as a ZIP for Unreal Engine.

    Builds a zip containing all PBR maps renamed with UE conventions, plus a
    Python script to wire them up in UE5.

    Args:
        texture_url: URL of the base-color texture (e.g. /outputs/abc.png)
        roughness: Default roughness value
        metalness: Default metalness value
        save_to: Local path to save the zip. If empty, saves to current dir.
    """
    import os
    r = client.post(
        "/export",
        json={"texture_url": texture_url, "roughness": roughness, "metalness": metalness, "format": "unreal"},
    )
    r.raise_for_status()
    filename = "material_export.zip"
    disp = r.headers.get("content-disposition", "")
    if "filename=" in disp:
        filename = disp.split("filename=", 1)[1].strip('"; ')
    out_path = os.path.abspath(os.path.join(save_to or os.getcwd(), filename))
    with open(out_path, "wb") as f:
        f.write(r.content)
    return f"Exported Unreal material to: {out_path} ({len(r.content)} bytes)"


@mcp.tool()
def download_all(texture_id: str, save_to: str = "") -> str:
    """Download all PBR maps for a texture as a ZIP.

    Args:
        texture_id: The texture_id from a generation result
        save_to: Local path to save the zip. If empty, saves to current dir.
    """
    import os
    r = client.get(f"/download-all/{texture_id}")
    r.raise_for_status()
    filename = f"{texture_id}_maps.zip"
    out_path = os.path.abspath(os.path.join(save_to or os.getcwd(), filename))
    with open(out_path, "wb") as f:
        f.write(r.content)
    return f"Downloaded all maps to: {out_path} ({len(r.content)} bytes)"


# ---------------------------------------------------------------------------
# Motion / animation
# ---------------------------------------------------------------------------

_ANIMATED_PROPS = (
    "pan_x", "pan_y", "displacement", "transmission",
    "ior", "emissive", "roughness", "metalness",
)


@mcp.tool()
def list_motion() -> str:
    """List current motion configs for all animated material properties.

    Animated props: pan_x, pan_y (UV scroll), displacement (depth), transmission
    (refraction strength), ior (refraction angle), emissive, roughness, metalness.
    Each config has: mode (static|sin|cos|linear), base, amp, freq, phase.
    """
    state = _get("/motion")
    if not state:
        return "No motion configured. All properties are static."
    lines = ["Motion state:"]
    for prop in _ANIMATED_PROPS:
        cfg = state.get(prop)
        if cfg:
            lines.append(
                f"  {prop}: mode={cfg['mode']} base={cfg['base']:.3f} "
                f"amp={cfg['amp']:.3f} freq={cfg['freq']:.3f} phase={cfg['phase']:.2f}"
            )
    return "\n".join(lines)


@mcp.tool()
def set_motion(
    prop: str,
    mode: str = "sin",
    base: float = 0.0,
    amp: float = 0.1,
    freq: float = 0.5,
    phase: float = 0.0,
) -> str:
    """Animate a material property over time.

    Value at time t is computed as:
      static: base  (effectively disabled)
      sin:    base + amp * sin(2π * freq * t + phase)
      cos:    base + amp * cos(2π * freq * t + phase)
      linear: base + freq * t           (use negative freq to reverse)

    Pan properties (pan_x, pan_y) scroll the UV offset on all texture maps.
    'linear' is typical for scrolling; 'sin'/'cos' give oscillation.

    Args:
        prop: One of: pan_x, pan_y, displacement, transmission, ior, emissive,
              roughness, metalness
        mode: static | sin | cos | linear
        base: Center/offset value
        amp: Amplitude (sin/cos only)
        freq: Cycles per second (sin/cos) or rate per second (linear)
        phase: Phase offset in radians (sin/cos)
    """
    if prop not in _ANIMATED_PROPS:
        return f"Unknown prop '{prop}'. Valid: {', '.join(_ANIMATED_PROPS)}"
    if mode not in ("static", "sin", "cos", "linear"):
        return f"Unknown mode '{mode}'. Valid: static, sin, cos, linear"
    cfg = {"mode": mode, "base": base, "amp": amp, "freq": freq, "phase": phase}
    _patch("/motion", {prop: cfg})
    return (
        f"Set motion on {prop}: mode={mode} base={base:.3f} amp={amp:.3f} "
        f"freq={freq:.3f} phase={phase:.2f}"
    )


@mcp.tool()
def clear_motion(prop: str) -> str:
    """Stop animating a single property (reverts it to the UI slider value).

    Args:
        prop: The property to clear
    """
    if prop not in _ANIMATED_PROPS:
        return f"Unknown prop '{prop}'. Valid: {', '.join(_ANIMATED_PROPS)}"
    _patch("/motion", {prop: None})
    return f"Cleared motion on {prop}."


@mcp.tool()
def clear_all_motion() -> str:
    """Remove all motion configs. Every property returns to its UI slider value."""
    _delete("/motion")
    return "All motion cleared."


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
