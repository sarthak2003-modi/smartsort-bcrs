"""
YOLO-family object detection wrapper.

Default weights: Ultralytics **YOLOv8n** (``yolov8n.pt``).
``yolo11n.pt`` (YOLO11n) is a drop-in alternative — pass it as ``weights``.

If the ``ultralytics`` package is not installed, the wrapper degrades gracefully
and returns an empty detection list with a clear reason string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from smartsort import config

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2


@dataclass(frozen=True)
class DetectionResult:
    detections: Tuple[Detection, ...]
    model_name: str
    available: bool
    reason: str = ""


class YOLODetector:
    """
    Thin Ultralytics YOLO wrapper.

    Parameters
    ----------
    weights :
        Checkpoint name or path. Default ``yolov8n.pt`` (YOLOv8n).
        Pass ``yolo11n.pt`` for YOLO11n without any other code changes.
    conf :
        Minimum detection confidence.
    """

    def __init__(
        self,
        weights: str = config.DEFAULT_DETECTOR_WEIGHTS,
        conf: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.device = device
        self._model: Any = None
        self._import_error: Optional[str] = None

    def load(self) -> bool:
        """Attempt to import ultralytics and load weights. Returns True on success."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._import_error = (
                f"ultralytics not installed ({exc}). "
                "Install with: pip install ultralytics. "
                f"Default weights remain {config.DEFAULT_DETECTOR_WEIGHTS}; "
                f"{config.ALTERNATIVE_DETECTOR_WEIGHTS} is a drop-in alternative."
            )
            return False
        self._model = YOLO(self.weights)
        return True

    @property
    def available(self) -> bool:
        if self._model is None and self._import_error is None:
            self.load()
        return self._model is not None

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run detection on an RGB uint8 image. Falls back if YOLO unavailable."""
        if not self.available:
            return DetectionResult(
                detections=tuple(),
                model_name=self.weights,
                available=False,
                reason=self._import_error or "YOLO model not loaded",
            )

        # Ultralytics accepts RGB numpy arrays
        results = self._model.predict(
            source=image,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        dets: List[Detection] = []
        for r in results:
            names = r.names or {}
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, cls_id))
                conf = float(box.conf.item())
                xyxy = tuple(float(x) for x in box.xyxy[0].tolist())
                dets.append(Detection(label=label, confidence=conf, xyxy=xyxy))  # type: ignore[arg-type]
        return DetectionResult(
            detections=tuple(dets),
            model_name=self.weights,
            available=True,
        )
