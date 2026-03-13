"""Label definitions for FASHN Human Parser."""

from typing import Dict, List

# 18-class label mappings
IDS_TO_LABELS: Dict[int, str] = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}

LABELS_TO_IDS: Dict[str, int] = {v: k for k, v in IDS_TO_LABELS.items()}

# Body coverage mappings for virtual try-on
CATEGORY_TO_BODY_COVERAGE: Dict[str, str] = {
    "tops": "upper",
    "bottoms": "lower",
    "one-pieces": "full",
}

BODY_COVERAGE_TO_LABELS: Dict[str, List[str]] = {
    "upper": ["top", "dress", "scarf"],
    "lower": ["skirt", "pants", "belt"],
    "full": ["top", "dress", "scarf", "skirt", "pants", "belt"],
}

# Labels typically preserved during virtual try-on
IDENTITY_LABELS: List[str] = ["face", "hair", "jewelry", "bag", "glasses", "hat"]
