#!/usr/bin/env python3
"""
Seamless Tiling Post-Processor

Uses noise-masked blending to make any texture tile seamlessly.
Supports Voronoi and Perlin noise masks.
"""

import numpy as np
from PIL import Image, ImageFilter
import argparse
from pathlib import Path


def generate_perlin_noise_2d(shape, scale=4, octaves=4):
    """Generate 2D Perlin-like noise using multiple octaves of smooth noise."""
    def smooth_noise(shape, scale):
        noise = np.random.rand(shape[0] // scale + 2, shape[1] // scale + 2)
        # Bilinear interpolation
        x = np.linspace(0, noise.shape[0] - 1, shape[0])
        y = np.linspace(0, noise.shape[1] - 1, shape[1])
        x_idx = x.astype(int)
        y_idx = y.astype(int)
        x_frac = x - x_idx
        y_frac = y - y_idx
        
        # Clamp indices
        x_idx = np.clip(x_idx, 0, noise.shape[0] - 2)
        y_idx = np.clip(y_idx, 0, noise.shape[1] - 2)
        
        # Bilinear interpolation
        result = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                xi, yi = x_idx[i], y_idx[j]
                xf, yf = x_frac[i], y_frac[j]
                result[i, j] = (
                    noise[xi, yi] * (1 - xf) * (1 - yf) +
                    noise[xi + 1, yi] * xf * (1 - yf) +
                    noise[xi, yi + 1] * (1 - xf) * yf +
                    noise[xi + 1, yi + 1] * xf * yf
                )
        return result
    
    result = np.zeros(shape)
    amplitude = 1.0
    total_amplitude = 0.0
    
    for i in range(octaves):
        current_scale = max(1, scale * (2 ** i))
        result += smooth_noise(shape, current_scale) * amplitude
        total_amplitude += amplitude
        amplitude *= 0.5
    
    return result / total_amplitude


def generate_voronoi_noise(shape, num_points=50, seed=None):
    """Generate Voronoi-based noise pattern."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points (tile them for seamless edges)
    points = np.random.rand(num_points, 2)
    
    # Create tiled points for seamless wrapping
    tiled_points = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tiled_points.append(points + np.array([dx, dy]))
    tiled_points = np.vstack(tiled_points)
    
    # Scale to image coordinates
    scaled_points = tiled_points * np.array([shape[0], shape[1]])
    
    # Create distance field
    y_coords, x_coords = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)
    
    # Find distance to nearest point for each pixel
    distances = np.zeros(shape[0] * shape[1])
    for i, coord in enumerate(coords):
        dists = np.sqrt(np.sum((scaled_points - coord) ** 2, axis=1))
        distances[i] = np.min(dists)
    
    distances = distances.reshape(shape)
    
    # Normalize to 0-1
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    return distances


def generate_voronoi_cells(shape, num_points=30, seed=None):
    """Generate Voronoi cell-based pattern (smoother blending zones). Vectorized."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points
    points = np.random.rand(num_points, 2)
    
    # Tile for seamless
    tiled_points = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tiled_points.append(points + np.array([dx, dy]))
    tiled_points = np.vstack(tiled_points)
    
    scaled_points = tiled_points * np.array([shape[0], shape[1]])
    
    # Vectorized distance computation
    y_coords, x_coords = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.stack([y_coords, x_coords], axis=-1)  # (H, W, 2)
    
    # Compute distances to all points: (H, W, num_tiled_points)
    # Use broadcasting: coords is (H, W, 1, 2), scaled_points is (1, 1, N, 2)
    diff = coords[:, :, np.newaxis, :] - scaled_points[np.newaxis, np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # (H, W, N)
    
    # Find two smallest distances per pixel
    sorted_dists = np.partition(dists, 1, axis=-1)[:, :, :2]
    d1 = sorted_dists[:, :, 0]
    d2 = sorted_dists[:, :, 1]
    
    # Edge distance normalized
    result = d1 / (d1 + d2 + 1e-6)
    
    return result


def make_tileable(image, noise_type='voronoi', blend_width=0.3, noise_params=None):
    """
    Make an image tileable using noise-masked blending.
    
    Args:
        image: PIL Image or numpy array
        noise_type: 'voronoi', 'perlin', or 'hybrid'
        blend_width: Width of blend zone as fraction of image size
        noise_params: Dict of parameters for noise generation
    
    Returns:
        Tileable PIL Image
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        img_array = image.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    
    h, w = img_array.shape[:2]
    has_alpha = img_array.shape[2] == 4 if len(img_array.shape) > 2 else False
    
    # Create offset version (shift by half)
    offset_array = np.roll(np.roll(img_array, h // 2, axis=0), w // 2, axis=1)
    
    # Generate noise mask
    noise_params = noise_params or {}
    
    if noise_type == 'voronoi':
        mask = generate_voronoi_cells((h, w), 
                                       num_points=noise_params.get('num_points', 30),
                                       seed=noise_params.get('seed', 42))
    elif noise_type == 'perlin':
        mask = generate_perlin_noise_2d((h, w),
                                         scale=noise_params.get('scale', 8),
                                         octaves=noise_params.get('octaves', 4))
    elif noise_type == 'hybrid':
        voronoi = generate_voronoi_cells((h, w), num_points=25, seed=42)
        perlin = generate_perlin_noise_2d((h, w), scale=6, octaves=3)
        mask = voronoi * 0.6 + perlin * 0.4
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Apply some blur for smoother transitions
    blur_radius = int(max(h, w) * 0.02)
    if blur_radius > 0:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask = np.array(mask_img).astype(np.float32) / 255.0
    
    # Normalize mask to 0-1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    
    # Expand mask for RGB(A) channels
    if len(img_array.shape) > 2:
        mask = mask[:, :, np.newaxis]
    
    # Blend original and offset using mask
    result = img_array * mask + offset_array * (1 - mask)
    
    # Convert back to uint8
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    return Image.fromarray(result)


def make_tileable_multipass(image, passes=3, noise_params=None):
    """
    Multi-pass tiling with varied offsets and noise patterns.
    Creates more organic, less obviously repeating results.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        img_array = image.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    
    h, w = img_array.shape[:2]
    noise_params = noise_params or {}
    base_seed = noise_params.get('seed', 42)
    
    result = img_array.copy()
    
    # Multiple passes with different offsets and noise patterns
    offsets = [
        (h // 2, w // 2),      # Center offset
        (h // 3, w * 2 // 3),  # Asymmetric offset 1
        (h * 2 // 3, w // 3),  # Asymmetric offset 2
    ]
    
    for i, (off_y, off_x) in enumerate(offsets[:passes]):
        # Create offset version
        offset_array = np.roll(np.roll(result, off_y, axis=0), off_x, axis=1)
        
        # Generate unique noise mask for this pass
        voronoi = generate_voronoi_cells((h, w), 
                                          num_points=40 + i * 10,
                                          seed=base_seed + i * 100)
        perlin = generate_perlin_noise_2d((h, w), scale=4 + i * 2, octaves=4)
        
        # Mix noise types with varying ratios per pass
        mix_ratio = 0.5 + (i * 0.1)
        mask = voronoi * mix_ratio + perlin * (1 - mix_ratio)
        
        # Blur for smooth transitions
        blur_radius = int(max(h, w) * (0.015 + i * 0.005))
        if blur_radius > 0:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = np.array(mask_img).astype(np.float32) / 255.0
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        
        # Expand for RGB
        if len(result.shape) > 2:
            mask = mask[:, :, np.newaxis]
        
        # Blend with decreasing influence per pass
        blend_strength = 0.6 - (i * 0.15)
        blended = result * mask + offset_array * (1 - mask)
        result = result * (1 - blend_strength) + blended * blend_strength
    
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result)


def make_tileable_multiscale(image, scale=2, noise_params=None):
    """
    Multi-scale tiling: tile at larger scale, blend, extract final tile.
    This breaks repetition at the tile boundary level.
    
    Args:
        image: Input image
        scale: Work at NxN tile scale (2 = 2x2, 3 = 3x3)
        noise_params: Noise generation parameters
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        img_array = image.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    
    h, w = img_array.shape[:2]
    noise_params = noise_params or {}
    base_seed = noise_params.get('seed', 42)
    
    # Create scaled-up tiled version
    big_h, big_w = h * scale, w * scale
    big_array = np.tile(img_array, (scale, scale, 1) if len(img_array.shape) > 2 else (scale, scale))
    
    # Apply multiple noise layers at different frequencies
    noise_layers = []
    
    # Layer 1: Large-scale Voronoi (spans multiple tiles)
    v1 = generate_voronoi_cells((big_h, big_w), num_points=20, seed=base_seed)
    noise_layers.append(('voronoi_large', v1, 0.3))
    
    # Layer 2: Medium Perlin
    p1 = generate_perlin_noise_2d((big_h, big_w), scale=4, octaves=4)
    noise_layers.append(('perlin_med', p1, 0.25))
    
    # Layer 3: Small-scale Voronoi (tile-scale features)
    v2 = generate_voronoi_cells((big_h, big_w), num_points=80, seed=base_seed + 50)
    noise_layers.append(('voronoi_small', v2, 0.25))
    
    # Layer 4: Fine Perlin
    p2 = generate_perlin_noise_2d((big_h, big_w), scale=16, octaves=3)
    noise_layers.append(('perlin_fine', p2, 0.2))
    
    # Combine noise layers
    combined_noise = np.zeros((big_h, big_w))
    for name, noise, weight in noise_layers:
        # Normalize each layer
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        combined_noise += noise * weight
    
    # Normalize combined
    combined_noise = (combined_noise - combined_noise.min()) / (combined_noise.max() - combined_noise.min() + 1e-6)
    
    # Blur for smooth transitions
    blur_radius = int(max(big_h, big_w) * 0.01)
    if blur_radius > 0:
        mask_img = Image.fromarray((combined_noise * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        combined_noise = np.array(mask_img).astype(np.float32) / 255.0
    
    # Create multiple offset versions and blend
    result = big_array.copy()
    
    offsets = [
        (big_h // 2, big_w // 2),
        (big_h // 3, big_w * 2 // 3),
        (big_h * 2 // 3, big_w // 4),
        (big_h // 4, big_w // 3),
    ]
    
    for i, (off_y, off_x) in enumerate(offsets):
        offset_array = np.roll(np.roll(result, off_y, axis=0), off_x, axis=1)
        
        # Rotate/shift the noise for each pass
        mask = np.roll(np.roll(combined_noise, off_y // 2, axis=0), off_x // 2, axis=1)
        
        # Add some per-pass variation
        perlin_var = generate_perlin_noise_2d((big_h, big_w), scale=8 + i * 4, octaves=2)
        perlin_var = (perlin_var - perlin_var.min()) / (perlin_var.max() - perlin_var.min() + 1e-6)
        mask = mask * 0.7 + perlin_var * 0.3
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        
        if len(result.shape) > 2:
            mask = mask[:, :, np.newaxis]
        
        blend_strength = 0.5 - (i * 0.1)
        blended = result * mask + offset_array * (1 - mask)
        result = result * (1 - blend_strength) + blended * blend_strength
    
    # Extract center tile (avoids edge artifacts)
    start_y = h // 2
    start_x = w // 2
    final = result[start_y:start_y + h, start_x:start_x + w]
    
    final = (np.clip(final, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(final)


def verify_tiling(image, grid_size=2):
    """Create a tiled preview to verify seamlessness."""
    w, h = image.size
    tiled = Image.new(image.mode, (w * grid_size, h * grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            tiled.paste(image, (i * w, j * h))
    
    return tiled


def main():
    parser = argparse.ArgumentParser(description='Make textures seamlessly tileable')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('-o', '--output', type=str, help='Output path (default: input_tileable.png)')
    parser.add_argument('-t', '--type', choices=['voronoi', 'perlin', 'hybrid', 'multipass', 'multiscale'], 
                        default='voronoi', help='Noise type (multiscale = work at 2x scale with multiple noise layers)')
    parser.add_argument('-p', '--preview', action='store_true', 
                        help='Also save a 2x2 tiled preview')
    parser.add_argument('--points', type=int, default=30, 
                        help='Number of Voronoi points (for voronoi/hybrid)')
    parser.add_argument('--scale', type=int, default=8, 
                        help='Perlin noise scale (for perlin/hybrid)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1
    
    # Load image
    print(f"Loading {input_path}...")
    image = Image.open(input_path).convert('RGB')
    
    # Process
    print(f"Making tileable using {args.type} noise mask...")
    noise_params = {
        'num_points': args.points,
        'scale': args.scale,
        'seed': args.seed
    }
    
    if args.type == 'multipass':
        tileable = make_tileable_multipass(image, passes=3, noise_params=noise_params)
    elif args.type == 'multiscale':
        tileable = make_tileable_multiscale(image, scale=2, noise_params=noise_params)
    else:
        tileable = make_tileable(image, noise_type=args.type, noise_params=noise_params)
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_tileable{input_path.suffix}"
    
    tileable.save(output_path)
    print(f"Saved tileable texture: {output_path}")
    
    # Preview
    if args.preview:
        preview = verify_tiling(tileable, grid_size=2)
        preview_path = output_path.parent / f"{output_path.stem}_preview{output_path.suffix}"
        preview.save(preview_path)
        print(f"Saved tiling preview: {preview_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
