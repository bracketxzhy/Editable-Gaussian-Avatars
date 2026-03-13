"""FASHN Human Parser - Semantic segmentation for fashion and virtual try-on."""

from .labels import (BODY_COVERAGE_TO_LABELS, CATEGORY_TO_BODY_COVERAGE,
                     IDENTITY_LABELS, IDS_TO_LABELS, LABELS_TO_IDS)
from .parser import FashnHumanParser

__version__ = "0.1.0"
__all__ = [
    "FashnHumanParser",
    "IDS_TO_LABELS",
    "LABELS_TO_IDS",
    "CATEGORY_TO_BODY_COVERAGE",
    "BODY_COVERAGE_TO_LABELS",
    "IDENTITY_LABELS",
]
