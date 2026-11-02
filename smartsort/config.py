"""
SmartSort configuration — ILLUSTRATIVE / UNCALIBRATED thresholds.

All confidence cut-offs and weight bands below are design defaults taken from the
NTU AI Literacy SmartSort deck and the public trial bay UI. They are **not**
calibrated against production Return Right RVM telemetry. Do not treat them as
deployed operating parameters.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# Class taxonomy (aligned with the live trial app category IDs)
# ---------------------------------------------------------------------------
# Trial app CATEGORY_LABEL map:
#   pet-bottle, aluminium-can, rubbish, food-waste, non-eligible,
#   crushed, tampered, unknown
#
# The deck groups these as: PET, aluminium, rubbish, food, crushed/tampered.

CLASS_IDS = (
    "pet-bottle",
    "aluminium-can",
    "rubbish",
    "food-waste",
    "non-eligible",
    "crushed",
    "tampered",
    "unknown",
)

CLASS_LABELS: Dict[str, str] = {
    "pet-bottle": "PET bottle",
    "aluminium-can": "Aluminium can",
    "rubbish": "Rubbish",
    "food-waste": "Food waste",
    "non-eligible": "Non-eligible pack",
    "crushed": "Crushed container",
    "tampered": "Tampered container",
    "unknown": "Unknown",
}

# Only empty eligible PET bottles and aluminium cans may receive the 10¢ refund.
ELIGIBLE_CATEGORIES: FrozenSet[str] = frozenset({"pet-bottle", "aluminium-can"})

# Deck-facing short names → canonical IDs (for CLI / docs convenience)
DECK_CLASS_ALIASES: Dict[str, str] = {
    "pet": "pet-bottle",
    "PET": "pet-bottle",
    "aluminium": "aluminium-can",
    "aluminum": "aluminium-can",
    "can": "aluminium-can",
    "rubbish": "rubbish",
    "food": "food-waste",
    "crushed": "crushed",
    "tampered": "tampered",
}

# ---------------------------------------------------------------------------
# Weight bands (grams) — ILLUSTRATIVE empty-container ranges from the deck
# ---------------------------------------------------------------------------
# Empty PET 8–70 g; empty aluminium can 10–25 g.
WEIGHT_BAND: Dict[str, Tuple[float, float]] = {
    "pet-bottle": (8.0, 70.0),
    "aluminium-can": (10.0, 25.0),
}

# Trial UI accepts manual entry in [1, 5000] g before band checks.
WEIGHT_ENTRY_MIN_G: float = 1.0
WEIGHT_ENTRY_MAX_G: float = 5000.0

# ---------------------------------------------------------------------------
# Decision thresholds — ILLUSTRATIVE / UNCALIBRATED
# ---------------------------------------------------------------------------
CONF_MIN: float = 0.55  # Hold if vision confidence is below this
ACCEPT_MIN: float = 0.70  # Pay only if fused score is at least this

# Fusion weights (cv vs weight fit); must sum to 1.0 for interpretability.
FUSION_W_CV: float = 0.60
FUSION_W_WEIGHT: float = 0.40
# Optional material-sensing contribution when a sensor reading is available.
FUSION_W_MATERIAL: float = 0.15  # re-normalised when material score is used

# BCRS refund in Singapore cents / dollars
REFUND_CENTS: int = 10
REFUND_SGD: float = 0.10

# ---------------------------------------------------------------------------
# Target metrics (design goals only — NOT measured results)
# ---------------------------------------------------------------------------
TARGET_ACCURACY: float = 0.95
TARGET_FPR: float = 0.02  # fraud paid
TARGET_FNR: float = 0.05  # valid held
TARGET_LATENCY_S: float = 2.0

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
# Detector: Ultralytics YOLOv8n is the default; YOLO11n is a drop-in alternative.
DEFAULT_DETECTOR_WEIGHTS: str = "yolov8n.pt"
ALTERNATIVE_DETECTOR_WEIGHTS: str = "yolo11n.pt"

# Classifier backbone for transfer learning (torchvision)
DEFAULT_CLASSIFIER_BACKBONE: str = "mobilenet_v3_small"  # or resnet50
CLASSIFIER_NUM_CLASSES: int = len(CLASS_IDS)
CLASSIFIER_IMAGE_SIZE: int = 224

# OCR / deposit-mark hints
BCRS_DEPOSIT_MARK_KEYWORDS: Tuple[str, ...] = (
    "BCRS",
    "RETURN RIGHT",
    "10 CENTS",
    "10¢",
    "DEPOSIT",
)

THRESHOLDS_NOTE = (
    "ILLUSTRATIVE / UNCALIBRATED — thresholds and weight bands are design "
    "defaults from the SmartSort deck / trial UI, not production calibrations."
)
