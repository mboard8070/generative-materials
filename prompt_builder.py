#!/usr/bin/env python3
"""
PBR Material Prompt Builder

Surface vocabulary and prompt engineering for material generation.
"""

from typing import Dict, List, Tuple
import random

# Surface quality vocabulary mapped to 0-1 ranges
SURFACE_VOCAB = {
    "roughness": {
        0.0: ["polished", "mirror-like", "glossy", "smooth", "highly reflective"],
        0.25: ["satin", "semi-gloss", "brushed", "soft sheen", "lightly polished"],
        0.5: ["matte", "eggshell", "low luster", "subtle texture"],
        0.75: ["rough", "textured", "coarse", "grainy", "bumpy"],
        1.0: ["very rough", "gritty", "porous", "raw", "unfinished", "heavily textured"],
    },
    "metallic": {
        0.0: ["plastic", "wood", "fabric", "ceramic", "stone", "rubber", "organic"],
        0.25: ["lacquered", "painted", "coated", "sealed"],
        0.5: ["metallic flake", "pearlescent", "car paint", "iridescent"],
        0.75: ["brushed metal", "anodized", "plated"],
        1.0: ["pure metal", "steel", "chrome", "iron", "copper", "gold", "aluminum", "brass"],
    },
    "age": {
        0.0: ["brand new", "pristine", "clean", "factory fresh", "mint condition"],
        0.25: ["like new", "minimal wear", "well maintained"],
        0.5: ["used", "worn", "some wear", "light scratches", "minor scuffs"],
        0.75: ["old", "weathered", "aged", "patina", "visible wear"],
        1.0: ["ancient", "heavily weathered", "rusted", "corroded", "distressed", "decayed"],
    },
    "moisture": {
        0.0: ["bone dry", "arid", "dusty", "parched"],
        0.25: ["dry", "normal"],
        0.5: ["slightly damp", "humid"],
        0.75: ["damp", "moist", "dewy"],
        1.0: ["wet", "slick", "water droplets", "rain-soaked", "dripping"],
    },
    "temperature": {
        0.0: ["frozen", "icy", "frost-covered", "freezing cold"],
        0.25: ["cold", "cool"],
        0.5: ["room temperature", "neutral"],
        0.75: ["warm", "heated"],
        1.0: ["hot", "glowing hot", "heat distortion", "scorched"],
    },
    "cleanliness": {
        0.0: ["spotless", "immaculate", "sterile", "perfectly clean"],
        0.25: ["clean", "tidy"],
        0.5: ["slightly dirty", "light dust", "minor smudges"],
        0.75: ["dirty", "dusty", "grimy", "stained"],
        1.0: ["filthy", "heavily soiled", "mud-caked", "grease-covered", "soot-covered"],
    },
}

# Base material types with their typical properties
MATERIAL_PRESETS = {
    "steel": {"roughness": 0.3, "metallic": 1.0, "age": 0.0},
    "brushed_steel": {"roughness": 0.4, "metallic": 1.0, "age": 0.1},
    "rusted_steel": {"roughness": 0.8, "metallic": 0.7, "age": 1.0},
    "chrome": {"roughness": 0.05, "metallic": 1.0, "age": 0.0},
    "copper": {"roughness": 0.3, "metallic": 1.0, "age": 0.2},
    "aged_copper": {"roughness": 0.5, "metallic": 0.8, "age": 0.8},
    "gold": {"roughness": 0.2, "metallic": 1.0, "age": 0.0},
    "aluminum": {"roughness": 0.35, "metallic": 1.0, "age": 0.1},
    "iron": {"roughness": 0.5, "metallic": 1.0, "age": 0.3},
    "plastic_glossy": {"roughness": 0.1, "metallic": 0.0, "age": 0.0},
    "plastic_matte": {"roughness": 0.6, "metallic": 0.0, "age": 0.0},
    "rubber": {"roughness": 0.7, "metallic": 0.0, "age": 0.2},
    "wood_polished": {"roughness": 0.25, "metallic": 0.0, "age": 0.1},
    "wood_raw": {"roughness": 0.7, "metallic": 0.0, "age": 0.3},
    "wood_weathered": {"roughness": 0.8, "metallic": 0.0, "age": 0.9},
    "concrete_new": {"roughness": 0.6, "metallic": 0.0, "age": 0.1},
    "concrete_old": {"roughness": 0.75, "metallic": 0.0, "age": 0.7},
    "brick": {"roughness": 0.8, "metallic": 0.0, "age": 0.4},
    "stone_polished": {"roughness": 0.15, "metallic": 0.0, "age": 0.0},
    "stone_rough": {"roughness": 0.85, "metallic": 0.0, "age": 0.5},
    "marble": {"roughness": 0.1, "metallic": 0.0, "age": 0.0},
    "granite": {"roughness": 0.3, "metallic": 0.0, "age": 0.1},
    "ceramic_glossy": {"roughness": 0.05, "metallic": 0.0, "age": 0.0},
    "ceramic_matte": {"roughness": 0.5, "metallic": 0.0, "age": 0.0},
    "glass": {"roughness": 0.0, "metallic": 0.0, "age": 0.0},
    "frosted_glass": {"roughness": 0.4, "metallic": 0.0, "age": 0.0},
    "leather": {"roughness": 0.5, "metallic": 0.0, "age": 0.2},
    "leather_worn": {"roughness": 0.6, "metallic": 0.0, "age": 0.7},
    "fabric_silk": {"roughness": 0.2, "metallic": 0.0, "age": 0.0},
    "fabric_cotton": {"roughness": 0.7, "metallic": 0.0, "age": 0.1},
    "fabric_denim": {"roughness": 0.75, "metallic": 0.0, "age": 0.3},
    "carbon_fiber": {"roughness": 0.3, "metallic": 0.3, "age": 0.0},
    "asphalt": {"roughness": 0.9, "metallic": 0.0, "age": 0.5},
    "sand": {"roughness": 1.0, "metallic": 0.0, "age": 0.0},
    "mud": {"roughness": 0.8, "metallic": 0.0, "age": 0.0, "moisture": 0.8},
    "snow": {"roughness": 0.6, "metallic": 0.0, "temperature": 0.0},
    "ice": {"roughness": 0.1, "metallic": 0.0, "temperature": 0.0},
}


def interpolate_vocab(vocab_dict: Dict[float, List[str]], value: float) -> List[str]:
    """
    Interpolate between vocabulary levels based on a 0-1 value.
    Returns blended word list from adjacent levels.
    """
    levels = sorted(vocab_dict.keys())
    
    # Find surrounding levels
    lower_level = 0.0
    upper_level = 1.0
    
    for level in levels:
        if level <= value:
            lower_level = level
        if level >= value and upper_level == 1.0:
            upper_level = level
            break
    
    if lower_level == upper_level:
        return vocab_dict[lower_level]
    
    # Blend based on proximity
    blend = (value - lower_level) / (upper_level - lower_level)
    
    lower_words = vocab_dict[lower_level]
    upper_words = vocab_dict[upper_level]
    
    # If closer to upper, prefer upper words
    if blend > 0.5:
        return upper_words[:3] + lower_words[:1]
    else:
        return lower_words[:3] + upper_words[:1]


def build_prompt(
    base_material: str = "",
    roughness: float = 0.5,
    metallic: float = 0.0,
    age: float = 0.0,
    moisture: float = 0.25,
    temperature: float = 0.5,
    cleanliness: float = 0.25,
    color: str = "",
    extra_details: str = "",
) -> str:
    """
    Build a material prompt from surface properties.
    
    Args:
        base_material: Base material type (e.g., "steel", "wood", "concrete")
        roughness: 0.0 (mirror) to 1.0 (very rough)
        metallic: 0.0 (dielectric) to 1.0 (pure metal)
        age: 0.0 (new) to 1.0 (ancient/weathered)
        moisture: 0.0 (dry) to 1.0 (wet)
        temperature: 0.0 (frozen) to 1.0 (hot)
        cleanliness: 0.0 (clean) to 1.0 (dirty)
        color: Optional color description
        extra_details: Additional details to append
        
    Returns:
        Generated prompt string
    """
    parts = []
    
    # Base material
    if base_material:
        parts.append(base_material)
    
    # Color
    if color:
        parts.append(color)
    
    # Surface qualities - only include if notably different from neutral
    
    # Roughness (always include - critical for PBR)
    roughness_words = interpolate_vocab(SURFACE_VOCAB["roughness"], roughness)
    parts.append(random.choice(roughness_words[:2]))
    
    # Metallic (include if notably metallic or explicitly non-metallic)
    if metallic > 0.4:
        metallic_words = interpolate_vocab(SURFACE_VOCAB["metallic"], metallic)
        parts.append(random.choice(metallic_words[:2]))
    
    # Age (include if not new)
    if age > 0.2:
        age_words = interpolate_vocab(SURFACE_VOCAB["age"], age)
        parts.append(random.choice(age_words[:2]))
    
    # Moisture (include if notably wet or dry)
    if moisture < 0.2 or moisture > 0.6:
        moisture_words = interpolate_vocab(SURFACE_VOCAB["moisture"], moisture)
        parts.append(random.choice(moisture_words[:2]))
    
    # Temperature (include if extreme)
    if temperature < 0.2 or temperature > 0.8:
        temp_words = interpolate_vocab(SURFACE_VOCAB["temperature"], temperature)
        parts.append(random.choice(temp_words[:2]))
    
    # Cleanliness (include if notably dirty)
    if cleanliness > 0.4:
        clean_words = interpolate_vocab(SURFACE_VOCAB["cleanliness"], cleanliness)
        parts.append(random.choice(clean_words[:2]))
    
    # Extra details
    if extra_details:
        parts.append(extra_details)
    
    # Build the prompt
    material_desc = " ".join(parts)
    
    prompt = f"seamless tileable PBR texture of {material_desc}, game texture, photorealistic material, top-down flat view, even lighting, no shadows, highly detailed surface, 4k quality"
    
    return prompt


def get_preset(preset_name: str) -> Dict[str, float]:
    """Get a material preset's default values."""
    return MATERIAL_PRESETS.get(preset_name, {
        "roughness": 0.5,
        "metallic": 0.0,
        "age": 0.0,
    })


def list_presets() -> List[str]:
    """List available material presets."""
    return list(MATERIAL_PRESETS.keys())


# Quick test
if __name__ == "__main__":
    print("=== Prompt Builder Test ===\n")
    
    # Test with sliders
    prompt = build_prompt(
        base_material="steel",
        roughness=0.4,
        metallic=1.0,
        age=0.6,
        color="dark gray",
    )
    print(f"Steel (worn):\n{prompt}\n")
    
    prompt = build_prompt(
        base_material="wood planks",
        roughness=0.7,
        metallic=0.0,
        age=0.8,
        moisture=0.3,
    )
    print(f"Wood (weathered):\n{prompt}\n")
    
    prompt = build_prompt(
        base_material="concrete",
        roughness=0.75,
        metallic=0.0,
        age=0.5,
        cleanliness=0.7,
    )
    print(f"Concrete (dirty):\n{prompt}\n")
    
    # Test presets
    print("=== Available Presets ===")
    for preset in list_presets()[:10]:
        values = get_preset(preset)
        print(f"  {preset}: {values}")
