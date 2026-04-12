#!/usr/bin/env python3
"""
Multi-Map Grid Utilities

Shared constants and functions for composing/splitting 3x2 PBR map grids.

Grid Layout:
  [ Albedo (RGB)  | Normal (RGB)     | Roughness (gray) ]   row 0
  [ Height (gray) | Metallic (gray)  | AO (gray)        ]   row 1

Each cell: 512x512, Total: 1536x1024
Grayscale maps are stored as RGB (R=G=B) in the grid and converted back to
mode 'L' when split.
"""

from PIL import Image

# Grid geometry
GRID_CELL_SIZE = 512
GRID_COLS = 3
GRID_ROWS = 2
GRID_WIDTH = GRID_CELL_SIZE * GRID_COLS   # 1536
GRID_HEIGHT = GRID_CELL_SIZE * GRID_ROWS  # 1024

# Map name -> (col, row) position in the grid
GRID_LAYOUT = {
    "albedo":    (0, 0),
    "normal":    (1, 0),
    "roughness": (2, 0),
    "height":    (0, 1),
    "metallic":  (1, 1),
    "ao":        (2, 1),
}

# Maps that are inherently single-channel (grayscale)
GRAYSCALE_MAPS = {"roughness", "height", "metallic", "ao"}

# Neutral fill values for missing maps
# Normal: flat facing-camera normal (128, 128, 255)
# Roughness: mid roughness 128
# Height: mid height 128
# Metallic: non-metallic 0
# AO: fully lit 255
NEUTRAL_FILLS = {
    "albedo":    (128, 128, 128),
    "normal":    (128, 128, 255),
    "roughness": 128,
    "height":    128,
    "metallic":  0,
    "ao":        255,
}


def split_grid(grid_image: Image.Image) -> dict:
    """Crop a 1536x1024 grid image into 6 individual PBR maps.

    Returns a dict mapping map name -> PIL Image.
    Grayscale maps are returned in mode 'L'.
    """
    if grid_image.size != (GRID_WIDTH, GRID_HEIGHT):
        grid_image = grid_image.resize((GRID_WIDTH, GRID_HEIGHT), Image.LANCZOS)

    maps = {}
    for name, (col, row) in GRID_LAYOUT.items():
        x = col * GRID_CELL_SIZE
        y = row * GRID_CELL_SIZE
        cell = grid_image.crop((x, y, x + GRID_CELL_SIZE, y + GRID_CELL_SIZE))

        if name in GRAYSCALE_MAPS:
            cell = cell.convert("L")
        else:
            cell = cell.convert("RGB")

        maps[name] = cell

    return maps


def compose_grid(maps: dict) -> Image.Image:
    """Assemble individual PBR maps into a 1536x1024 grid image.

    Args:
        maps: dict mapping map name -> PIL Image (any mode).
              Missing maps are filled with neutral values.

    Returns:
        RGB PIL Image of size 1536x1024.
    """
    grid = Image.new("RGB", (GRID_WIDTH, GRID_HEIGHT))

    for name, (col, row) in GRID_LAYOUT.items():
        x = col * GRID_CELL_SIZE
        y = row * GRID_CELL_SIZE

        if name in maps and maps[name] is not None:
            cell = maps[name]
            # Resize to cell size if needed
            if cell.size != (GRID_CELL_SIZE, GRID_CELL_SIZE):
                cell = cell.resize((GRID_CELL_SIZE, GRID_CELL_SIZE), Image.LANCZOS)
            # Convert grayscale to RGB for grid storage
            if cell.mode == "L":
                cell = cell.convert("RGB")
            elif cell.mode != "RGB":
                cell = cell.convert("RGB")
        else:
            # Fill with neutral value
            fill = NEUTRAL_FILLS[name]
            if isinstance(fill, tuple):
                cell = Image.new("RGB", (GRID_CELL_SIZE, GRID_CELL_SIZE), fill)
            else:
                cell = Image.new("L", (GRID_CELL_SIZE, GRID_CELL_SIZE), fill)
                cell = cell.convert("RGB")

        grid.paste(cell, (x, y))

    return grid
