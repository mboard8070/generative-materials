// Material presets and prompt suggestions

export interface MaterialPreset {
  category: string
  materials: {
    name: string
    prompt: string
    roughness: number
    metalness: number
  }[]
}

export const materialPresets: MaterialPreset[] = [
  {
    category: "Metals",
    materials: [
      { name: "Brushed Steel", prompt: "brushed stainless steel, industrial finish", roughness: 0.3, metalness: 1.0 },
      { name: "Rusty Iron", prompt: "weathered rusty iron, oxidized metal with orange patina", roughness: 0.8, metalness: 0.7 },
      { name: "Copper Patina", prompt: "aged copper with green verdigris patina", roughness: 0.6, metalness: 0.8 },
      { name: "Gold Leaf", prompt: "hammered gold leaf, luxury metallic finish", roughness: 0.2, metalness: 1.0 },
      { name: "Bronze", prompt: "antique bronze, warm brown metal finish", roughness: 0.4, metalness: 0.9 },
      { name: "Aluminum", prompt: "anodized aluminum, matte silver industrial", roughness: 0.5, metalness: 0.9 },
    ]
  },
  {
    category: "Wood",
    materials: [
      { name: "Oak Planks", prompt: "natural oak wood planks, warm grain pattern", roughness: 0.7, metalness: 0.0 },
      { name: "Walnut", prompt: "dark walnut wood, rich brown grain", roughness: 0.6, metalness: 0.0 },
      { name: "Weathered Barn", prompt: "weathered barn wood, grey aged planks with nail holes", roughness: 0.9, metalness: 0.0 },
      { name: "Cherry Wood", prompt: "polished cherry wood, reddish brown smooth finish", roughness: 0.4, metalness: 0.0 },
      { name: "Pine", prompt: "knotty pine wood, light colored with visible knots", roughness: 0.7, metalness: 0.0 },
      { name: "Bamboo", prompt: "natural bamboo strips, light tan woven pattern", roughness: 0.5, metalness: 0.0 },
    ]
  },
  {
    category: "Stone",
    materials: [
      { name: "Marble", prompt: "white carrara marble, elegant veined stone", roughness: 0.3, metalness: 0.0 },
      { name: "Granite", prompt: "speckled grey granite, polished stone surface", roughness: 0.4, metalness: 0.0 },
      { name: "Slate", prompt: "dark grey slate, layered natural stone", roughness: 0.8, metalness: 0.0 },
      { name: "Sandstone", prompt: "tan sandstone, rough natural desert rock", roughness: 0.9, metalness: 0.0 },
      { name: "Cobblestone", prompt: "old cobblestone pavers, rounded worn stones", roughness: 0.8, metalness: 0.0 },
      { name: "Limestone", prompt: "cream limestone blocks, subtle fossil patterns", roughness: 0.6, metalness: 0.0 },
    ]
  },
  {
    category: "Concrete & Brick",
    materials: [
      { name: "Raw Concrete", prompt: "raw poured concrete, industrial grey with form marks", roughness: 0.9, metalness: 0.0 },
      { name: "Polished Concrete", prompt: "polished concrete floor, smooth grey finish", roughness: 0.3, metalness: 0.0 },
      { name: "Red Brick", prompt: "classic red clay brick wall, mortar joints", roughness: 0.8, metalness: 0.0 },
      { name: "White Brick", prompt: "painted white brick, modern clean finish", roughness: 0.7, metalness: 0.0 },
      { name: "Cinder Block", prompt: "grey cinder block wall, industrial texture", roughness: 0.9, metalness: 0.0 },
      { name: "Stucco", prompt: "textured stucco wall, mediterranean finish", roughness: 0.8, metalness: 0.0 },
    ]
  },
  {
    category: "Tiles",
    materials: [
      { name: "Ceramic Tile", prompt: "glossy white ceramic tile, clean subway pattern", roughness: 0.1, metalness: 0.0 },
      { name: "Terracotta", prompt: "terracotta floor tiles, warm orange clay", roughness: 0.7, metalness: 0.0 },
      { name: "Moroccan", prompt: "colorful moroccan zellige tiles, blue and white geometric", roughness: 0.4, metalness: 0.0 },
      { name: "Hexagon", prompt: "black and white hexagon floor tiles, vintage pattern", roughness: 0.3, metalness: 0.0 },
      { name: "Mosaic", prompt: "small glass mosaic tiles, iridescent blue green", roughness: 0.2, metalness: 0.1 },
      { name: "Porcelain", prompt: "large format porcelain tile, marble look", roughness: 0.2, metalness: 0.0 },
    ]
  },
  {
    category: "Fabric",
    materials: [
      { name: "Denim", prompt: "blue denim fabric, woven cotton jeans texture", roughness: 0.9, metalness: 0.0 },
      { name: "Leather", prompt: "brown leather, soft grain texture", roughness: 0.6, metalness: 0.0 },
      { name: "Velvet", prompt: "deep red velvet fabric, luxurious soft pile", roughness: 0.8, metalness: 0.0 },
      { name: "Canvas", prompt: "natural canvas fabric, woven linen texture", roughness: 0.9, metalness: 0.0 },
      { name: "Silk", prompt: "shimmering silk fabric, smooth elegant sheen", roughness: 0.2, metalness: 0.1 },
      { name: "Wool Knit", prompt: "chunky wool knit, cable knit sweater pattern", roughness: 1.0, metalness: 0.0 },
    ]
  },
  {
    category: "Ground",
    materials: [
      { name: "Grass", prompt: "green grass lawn, natural outdoor ground cover", roughness: 0.9, metalness: 0.0 },
      { name: "Dirt", prompt: "brown dirt soil, natural earth ground", roughness: 1.0, metalness: 0.0 },
      { name: "Gravel", prompt: "grey gravel stones, small pebbles ground cover", roughness: 0.9, metalness: 0.0 },
      { name: "Sand", prompt: "beach sand, fine grain tan desert texture", roughness: 0.9, metalness: 0.0 },
      { name: "Asphalt", prompt: "black asphalt road, rough pavement texture", roughness: 0.8, metalness: 0.0 },
      { name: "Snow", prompt: "fresh white snow, powder winter ground", roughness: 0.7, metalness: 0.0 },
    ]
  },
  {
    category: "Organic",
    materials: [
      { name: "Tree Bark", prompt: "rough tree bark, natural brown wood texture", roughness: 1.0, metalness: 0.0 },
      { name: "Moss", prompt: "green moss, soft natural forest floor", roughness: 0.9, metalness: 0.0 },
      { name: "Coral", prompt: "pink coral reef texture, organic marine pattern", roughness: 0.7, metalness: 0.0 },
      { name: "Scales", prompt: "reptile scales, iridescent snake skin pattern", roughness: 0.4, metalness: 0.1 },
      { name: "Fur", prompt: "soft animal fur, brown fluffy texture", roughness: 1.0, metalness: 0.0 },
      { name: "Seashell", prompt: "pearlescent seashell, iridescent mother of pearl", roughness: 0.2, metalness: 0.2 },
    ]
  }
]

export const promptModifiers = {
  weathering: [
    "pristine new condition",
    "slightly worn",
    "moderately weathered",
    "heavily weathered and aged",
    "ancient and decayed"
  ],
  scale: [
    "fine detailed micro texture",
    "small scale pattern",
    "medium scale",
    "large scale pattern",
    "macro oversized detail"
  ],
  style: [
    "photorealistic",
    "stylized game art",
    "hand-painted",
    "sci-fi futuristic",
    "fantasy medieval"
  ]
}
