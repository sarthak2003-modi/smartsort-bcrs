"""
04 · Fuse — combine vision confidence with weight (and optional material) fit.
"""

from __future__ import annotations

from typing import Optional, Tuple

from smartsort import config


def weight_fit(weight_g: float, band: Tuple[float, float], soft_edge: float = 0.0) -> float:
    """
    Return a [0, 1] score for how well ``weight_g`` sits inside ``band``.

    Default (``soft_edge=0``): **plateau** membership — 1.0 anywhere inside the
    closed band, 0.0 outside. This matches the deck's empty-container in-band
    gate (fraud is out-of-band mass, not "too far from the midpoint").

    With ``soft_edge>0``, scores ramp linearly from 0 to 1 across a shoulder of
    that many grams just outside each edge (still 1.0 fully inside).
    """
    lo, hi = band
    if hi <= lo:
        raise ValueError(f"Invalid weight band: {band}")
    if lo <= weight_g <= hi:
        return 1.0
    if soft_edge <= 0:
        return 0.0
    if weight_g < lo:
        return _clip01(1.0 - (lo - weight_g) / soft_edge)
    return _clip01(1.0 - (weight_g - hi) / soft_edge)


def fuse(
    cv_conf: float,
    weight_score: float,
    material_score: Optional[float] = None,
    w_cv: float = config.FUSION_W_CV,
    w_weight: float = config.FUSION_W_WEIGHT,
    w_material: float = config.FUSION_W_MATERIAL,
) -> float:
    """
    Weighted fusion into one confidence score in [0, 1].

    When ``material_score`` is provided, the three weights are re-normalised so
    they still sum to 1.0.
    """
    cv_conf = _clip01(cv_conf)
    weight_score = _clip01(weight_score)

    if material_score is None:
        total_w = w_cv + w_weight
        if total_w <= 0:
            raise ValueError("Fusion weights must be positive")
        return _clip01((w_cv * cv_conf + w_weight * weight_score) / total_w)

    material_score = _clip01(material_score)
    total_w = w_cv + w_weight + w_material
    if total_w <= 0:
        raise ValueError("Fusion weights must be positive")
    return _clip01(
        (w_cv * cv_conf + w_weight * weight_score + w_material * material_score)
        / total_w
    )


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
