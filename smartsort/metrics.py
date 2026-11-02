"""
Evaluation metrics vs SmartSort design targets.

Targets (deck — design goals only, not measured production results):
  accuracy ≥ 95%, FPR (fraud paid) < 2%, FNR (valid held) < 5%, latency < 2 s

This module never fabricates accuracy numbers; it only scores labelled trials
you provide and compares them to the targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from smartsort import config


@dataclass(frozen=True)
class TrialCase:
    """One labelled evaluation example."""

    y_true_eligible: bool  # True if the item truly deserved PAY
    y_pred_eligible: bool  # True if the pipeline returned PAY
    latency_s: float = 0.0
    category_true: Optional[str] = None
    category_pred: Optional[str] = None


@dataclass(frozen=True)
class MetricsReport:
    n: int
    accuracy: Optional[float]
    fpr: Optional[float]  # fraud paid: pred PAY & true not eligible
    fnr: Optional[float]  # valid held: pred HOLD & true eligible
    mean_latency_s: Optional[float]
    classification_accuracy: Optional[float]
    meets_accuracy: Optional[bool]
    meets_fpr: Optional[bool]
    meets_fnr: Optional[bool]
    meets_latency: Optional[bool]
    targets: dict

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "mean_latency_s": self.mean_latency_s,
            "classification_accuracy": self.classification_accuracy,
            "meets_accuracy": self.meets_accuracy,
            "meets_fpr": self.meets_fpr,
            "meets_fnr": self.meets_fnr,
            "meets_latency": self.meets_latency,
            "targets": self.targets,
            "note": (
                "Targets are design goals from the SmartSort deck. "
                "No trained weights or measured results are shipped with this repo."
            ),
        }


def evaluate_trials(cases: Sequence[TrialCase]) -> MetricsReport:
    if not cases:
        return MetricsReport(
            n=0,
            accuracy=None,
            fpr=None,
            fnr=None,
            mean_latency_s=None,
            classification_accuracy=None,
            meets_accuracy=None,
            meets_fpr=None,
            meets_fnr=None,
            meets_latency=None,
            targets=_targets(),
        )

    n = len(cases)
    correct = sum(1 for c in cases if c.y_true_eligible == c.y_pred_eligible)
    # FPR: among truly ineligible, fraction paid
    neg = [c for c in cases if not c.y_true_eligible]
    pos = [c for c in cases if c.y_true_eligible]
    fpr = (sum(1 for c in neg if c.y_pred_eligible) / len(neg)) if neg else None
    fnr = (sum(1 for c in pos if not c.y_pred_eligible) / len(pos)) if pos else None
    accuracy = correct / n
    mean_lat = sum(c.latency_s for c in cases) / n

    cls_pairs = [
        (c.category_true, c.category_pred)
        for c in cases
        if c.category_true is not None and c.category_pred is not None
    ]
    cls_acc = (
        sum(1 for a, b in cls_pairs if a == b) / len(cls_pairs) if cls_pairs else None
    )

    return MetricsReport(
        n=n,
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        mean_latency_s=mean_lat,
        classification_accuracy=cls_acc,
        meets_accuracy=accuracy >= config.TARGET_ACCURACY,
        meets_fpr=(fpr < config.TARGET_FPR) if fpr is not None else None,
        meets_fnr=(fnr < config.TARGET_FNR) if fnr is not None else None,
        meets_latency=mean_lat < config.TARGET_LATENCY_S,
        targets=_targets(),
    )


def _targets() -> dict:
    return {
        "accuracy": config.TARGET_ACCURACY,
        "fpr": config.TARGET_FPR,
        "fnr": config.TARGET_FNR,
        "latency_s": config.TARGET_LATENCY_S,
    }
