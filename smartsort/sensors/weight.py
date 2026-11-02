"""
03 · Weigh — load-cell interface plus a manual-entry stub for the trial bay.

Physical load cells are not present in this prototype; ``ManualWeightEntry``
stands in for the chute scale exactly as the browser trial app does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from smartsort import config


@dataclass(frozen=True)
class WeightReading:
    grams: float
    source: str  # "load_cell" | "manual" | "simulated"
    ok: bool
    message: str = ""


def parse_weight_grams(value) -> WeightReading:
    """Validate a weight value the same way the trial UI does (1–5000 g)."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return WeightReading(0.0, "manual", False, "weightGrams is required and must be a number.")
    if isinstance(value, str):
        text = value.strip()
        try:
            # reject non-numeric strings early
            float(text)
        except ValueError:
            return WeightReading(0.0, "manual", False, "weightGrams must be a number of grams.")
        value = float(text)
    try:
        grams = float(value)
    except (TypeError, ValueError):
        return WeightReading(0.0, "manual", False, "weightGrams must be a number of grams.")
    if not (grams == grams):  # NaN
        return WeightReading(0.0, "manual", False, "weightGrams must be a number of grams.")
    if grams < config.WEIGHT_ENTRY_MIN_G or grams > config.WEIGHT_ENTRY_MAX_G:
        return WeightReading(
            grams,
            "manual",
            False,
            f"weightGrams must be between {config.WEIGHT_ENTRY_MIN_G:g} and "
            f"{config.WEIGHT_ENTRY_MAX_G:g} grams.",
        )
    return WeightReading(round(grams, 1), "manual", True)


def in_band(weight_g: float, band: Tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= weight_g <= hi


class WeightSource(Protocol):
    def read(self) -> WeightReading: ...


class ManualWeightEntry:
    """Browser-trial style stub: operator / user types the mass in grams."""

    def __init__(self, grams: float) -> None:
        self._grams = grams

    def read(self) -> WeightReading:
        return parse_weight_grams(self._grams)


class LoadCell:
    """
    Placeholder for a real serial / HX711 load-cell driver.

    Until hardware is wired, ``simulate=True`` returns a fixed or noisy reading.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        simulate: bool = True,
        simulated_grams: float = 18.0,
    ) -> None:
        self.port = port
        self.simulate = simulate
        self.simulated_grams = simulated_grams

    def read(self) -> WeightReading:
        if self.simulate or not self.port:
            return WeightReading(self.simulated_grams, "simulated", True, "simulated load cell")
        raise NotImplementedError(
            f"Real load-cell driver for port {self.port!r} is not implemented in this prototype."
        )
