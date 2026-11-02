"""Physical sensing stubs: load-cell weight and NIR / capacitive material."""

from smartsort.sensors.material import MaterialReading, MaterialSensor, SimulatedMaterialSensor
from smartsort.sensors.weight import LoadCell, ManualWeightEntry, WeightReading

__all__ = [
    "LoadCell",
    "ManualWeightEntry",
    "WeightReading",
    "MaterialSensor",
    "SimulatedMaterialSensor",
    "MaterialReading",
]
