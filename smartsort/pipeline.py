"""
SmartSort decision pipeline — ``evaluate_return`` matching the deck pseudocode.

    category, cv_conf = classify(image)
    if cv_conf < CONF_MIN:          return HOLD("low vision confidence")
    if category not in ELIGIBLE:    return HOLD("ineligible: " + category)
    band = WEIGHT_BAND[category]
    if weight outside band:         return HOLD("weight mismatch")
    score = fuse(cv_conf, weight_fit(weight_g, band))
    if score >= ACCEPT_MIN:         return PAY(0.10)
    return HOLD("fusion score too low")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from smartsort import config
from smartsort.fusion import fuse, weight_fit
from smartsort.sensors.material import MaterialReading
from smartsort.sensors.weight import in_band


@dataclass
class Decision:
    action: str  # "PAY" | "HOLD"
    refund_sgd: float
    refund_cents: int
    eligible: bool
    category: str
    cv_confidence: float
    fusion_score: float
    weight_g: float
    weight_score: float
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    material: Optional[str] = None
    label: str = ""
    thresholds_note: str = config.THRESHOLDS_NOTE

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "refund_sgd": self.refund_sgd,
            "refund_cents": self.refund_cents,
            "eligible": self.eligible,
            "category": self.category,
            "cv_confidence": self.cv_confidence,
            "fusion_score": self.fusion_score,
            "weight_g": self.weight_g,
            "weight_score": self.weight_score,
            "reasons": list(self.reasons),
            "latency_ms": self.latency_ms,
            "material": self.material,
            "label": self.label,
            "thresholds_note": self.thresholds_note,
        }


ClassifyFn = Callable[[np.ndarray], Tuple[str, float]]


def _default_classify(image: np.ndarray) -> Tuple[str, float]:
    """Best-effort classify using TransferClassifier (untrained head is fine for plumbing)."""
    from smartsort.models.classifier import TransferClassifier

    clf = TransferClassifier(freeze_backbone=True)
    result = clf.predict(image)
    return result.category, result.confidence


def evaluate_return(
    image: np.ndarray,
    weight_g: float,
    classify_fn: Optional[ClassifyFn] = None,
    material_reading: Optional[MaterialReading] = None,
    conf_min: float = config.CONF_MIN,
    accept_min: float = config.ACCEPT_MIN,
) -> Decision:
    """
    Run the four-step SmartSort check and return a PAY or HOLD decision.
    """
    t0 = time.perf_counter()
    classify_fn = classify_fn or _default_classify

    category, cv_conf = classify_fn(image)
    # Normalise deck aliases
    category = config.DECK_CLASS_ALIASES.get(category, category)
    label = config.CLASS_LABELS.get(category, category)
    reasons: List[str] = []

    if cv_conf < conf_min:
        reasons.append("low vision confidence")
        return _hold(
            category, cv_conf, weight_g, 0.0, 0.0, reasons, label, t0,
            material=material_reading.material if material_reading else None,
        )

    if category not in config.ELIGIBLE_CATEGORIES:
        reasons.append(f"ineligible: {category}")
        return _hold(
            category, cv_conf, weight_g, 0.0, 0.0, reasons, label, t0,
            material=material_reading.material if material_reading else None,
        )

    band = config.WEIGHT_BAND[category]
    w_score = weight_fit(weight_g, band)
    if not in_band(weight_g, band):
        reasons.append("weight mismatch")
        return _hold(
            category, cv_conf, weight_g, w_score, 0.0, reasons, label, t0,
            material=material_reading.material if material_reading else None,
        )

    material_score = None
    material_name = None
    if material_reading is not None and material_reading.ok:
        material_name = material_reading.material
        # Map material agreement with the vision category
        expected = "pet" if category == "pet-bottle" else "aluminium"
        if material_reading.is_eligible_material and material_reading.material == expected:
            material_score = material_reading.confidence
        elif material_reading.is_eligible_material:
            material_score = 0.4 * material_reading.confidence
        else:
            material_score = 0.0
            reasons.append(f"material mismatch: {material_reading.material}")

    score = fuse(cv_conf, w_score, material_score)

    if score >= accept_min and not (
        material_reading is not None
        and material_reading.ok
        and not material_reading.is_eligible_material
    ):
        latency = (time.perf_counter() - t0) * 1000.0
        return Decision(
            action="PAY",
            refund_sgd=config.REFUND_SGD,
            refund_cents=config.REFUND_CENTS,
            eligible=True,
            category=category,
            cv_confidence=cv_conf,
            fusion_score=score,
            weight_g=weight_g,
            weight_score=w_score,
            reasons=["eligible, empty, in-band"],
            latency_ms=latency,
            material=material_name,
            label=label,
        )

    if score < accept_min:
        reasons.append("fusion score too low")
    return _hold(
        category, cv_conf, weight_g, w_score, score, reasons, label, t0,
        material=material_name,
    )


def _hold(
    category: str,
    cv_conf: float,
    weight_g: float,
    w_score: float,
    fusion_score: float,
    reasons: List[str],
    label: str,
    t0: float,
    material: Optional[str] = None,
) -> Decision:
    latency = (time.perf_counter() - t0) * 1000.0
    return Decision(
        action="HOLD",
        refund_sgd=0.0,
        refund_cents=0,
        eligible=False,
        category=category,
        cv_confidence=cv_conf,
        fusion_score=fusion_score,
        weight_g=weight_g,
        weight_score=w_score,
        reasons=reasons or ["held for operator review"],
        latency_ms=latency,
        material=material,
        label=label,
    )
