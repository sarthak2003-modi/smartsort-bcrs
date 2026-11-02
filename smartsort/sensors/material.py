"""
NIR / capacitive material sensing — interface + simulated stub.

Confirms PET plastic or aluminium vs paper, glass, or other contaminants.
No physical NIR hardware ships with this repo; use ``SimulatedMaterialSensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


MATERIAL_CLASSES = ("pet", "aluminium", "paper", "glass", "other")


@dataclass(frozen=True)
class MaterialReading:
    material: str
    confidence: float
    source: str
    ok: bool
    message: str = ""

    @property
    def is_eligible_material(self) -> bool:
        return self.material in {"pet", "aluminium"}


class MaterialSource(Protocol):
    def read(self) -> MaterialReading: ...


class MaterialSensor:
    """
    Placeholder for a real NIR / capacitive probe.

    Raises ``NotImplementedError`` unless ``simulate`` is left True (default).
    """

    def __init__(
        self,
        device_path: Optional[str] = None,
        simulate: bool = True,
        simulated_material: str = "pet",
        simulated_confidence: float = 0.9,
    ) -> None:
        self.device_path = device_path
        self.simulate = simulate
        self.simulated_material = simulated_material
        self.simulated_confidence = simulated_confidence

    def read(self) -> MaterialReading:
        if self.simulate or not self.device_path:
            return MaterialReading(
                material=self.simulated_material,
                confidence=self.simulated_confidence,
                source="simulated",
                ok=True,
                message="simulated NIR/capacitive reading",
            )
        raise NotImplementedError(
            f"Real material sensor at {self.device_path!r} is not implemented."
        )


class SimulatedMaterialSensor(MaterialSensor):
    """Explicit alias used by demos and tests."""

    def __init__(self, material: str = "pet", confidence: float = 0.9) -> None:
        super().__init__(
            simulate=True,
            simulated_material=material,
            simulated_confidence=confidence,
        )
