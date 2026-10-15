"""Config and taxonomy for the video classifier."""
from dataclasses import dataclass, field
from typing import Dict, List

# Map Video-MME sub-categories to our 6 chunking buckets
CATEGORY_TAXONOMY: Dict[str, List[str]] = {
    "Sports_HighMotion": [
        "Athletics", "Football", "Basketball", "Exercise", "Acrobatics", "Other Sports"
    ],
    "Gameplay_Esports": [
        "Esports"
    ],
    "Podcast_TalkingHead": [
        "News Report", "Finance & Commerce", "Technology", "Humanity & History"
    ],
    "Wildlife_Nature": [
        "Biology & Medicine", "Geography", "Astronomy"
    ],
    "LifeRecord_Egocentric": [
        "Daily Life", "Travel", "Fashion", "Food", "Life Tip", "Handicraft"
    ],
    "Performance_Narrative": [
        "Stage Play", "Magic Show", "Movie & TV Show", "Documentary", 
        "Animation", "Variety Show", "Literature & Art", "Multilingual"
    ]
}

# Reverse lookup
SUB_TO_CATEGORY: Dict[str, str] = {}
for cat, subs in CATEGORY_TAXONOMY.items():
    for sub in subs:
        SUB_TO_CATEGORY[sub] = cat

# What chunking approach fits each category
CHUNKING_POLICIES: Dict[str, Dict[str, str]] = {
    "Sports_HighMotion": {
        "strategy": "Dense Optical Flow Window (2-5s)",
        "trigger": "Motion kinetic delta threshold > 0.35",
        "rationale": "High-velocity actions require tight temporal windows to avoid missing critical sports plays."
    },
    "Gameplay_Esports": {
        "strategy": "UI/State & Event-Triggered Chunking",
        "trigger": "HUD changes, score shifts, round transitions",
        "rationale": "Game mechanics have discrete state boundaries; chunking on state shifts preserves game logic."
    },
    "Podcast_TalkingHead": {
        "strategy": "Audio/Transcript-Driven Topic Chunking (30-120s)",
        "trigger": "Speaker turn / semantic pause > 1.2s",
        "rationale": "Visual frames are mostly static talking heads; indexing must follow semantic dialogue shifts."
    },
    "Wildlife_Nature": {
        "strategy": "Visual Scene Continuity Chunking (10-45s)",
        "trigger": "Visual embedding cosine distance > 0.40",
        "rationale": "Captures habitat transitions and animal entrance/exit events without breaking calm tracking shots."
    },
    "LifeRecord_Egocentric": {
        "strategy": "Hand-Object Interaction Window (5-15s)",
        "trigger": "Hand interaction / object contact delta",
        "rationale": "First-person activity revolves around manipulating tools, ingredients, or items."
    },
    "Performance_Narrative": {
        "strategy": "Cinematic Shot & Narrative Scene Chunking",
        "trigger": "Shot cut detection / lighting transition",
        "rationale": "Follows directorial scene cuts and theatrical transitions for coherent narrative indexing."
    }
}

CLASS_NAMES = list(CATEGORY_TAXONOMY.keys())
NUM_CLASSES = len(CLASS_NAMES)

@dataclass
class ModelConfig:
    num_frames: int = 8                # Sparse keyframes sampled per video
    frame_size: int = 224              # Standard input resolution
    visual_dim: int = 576              # MobileNetV3-Small feature dim
    motion_dim: int = 8                # Motion energy bins
    audio_dim: int = 8                 # Audio presence / spectral bins
    hidden_dim: int = 128              # Projection and attention dimension
    dropout: float = 0.2
    temperature: float = 1.0           # Temperature scaling for probability calibration
    uncertainty_threshold: float = 0.55 # Below this, flag as uncertain / trigger dense pass
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 25
    device: str = "cpu"                # Portable CPU-first design for Grader Bot
