"""
02 · Classify — transfer learning on MobileNetV3 / ResNet50 (torchvision).

Freeze-then-finetune pattern for edge-friendly BCRS category recognition.
Pretrained ImageNet weights download at runtime from torchvision; this package
does **not** ship fine-tuned BCRS weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from smartsort import config

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ClassifierResult:
    category: str
    confidence: float
    probabilities: Tuple[float, ...]
    backbone: str


class TransferClassifier:
    """
    Transfer-learning image classifier.

    Parameters
    ----------
    backbone :
        ``mobilenet_v3_small`` (default, edge-friendly) or ``resnet50``.
    weights_path :
        Optional path to a fine-tuned ``state_dict``. When omitted, the model
        uses ImageNet-pretrained backbone weights with a randomly initialised
        classification head — useful for plumbing demos, **not** for accuracy claims.
    freeze_backbone :
        If True, freeze feature extractor parameters (stage 1 of freeze-then-finetune).
    device :
        ``cpu`` or ``cuda``. Defaults to CPU.
    """

    def __init__(
        self,
        backbone: str = config.DEFAULT_CLASSIFIER_BACKBONE,
        weights_path: Optional[PathLike] = None,
        class_ids: Optional[Sequence[str]] = None,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ) -> None:
        self.backbone_name = backbone
        self.class_ids: List[str] = list(class_ids or config.CLASS_IDS)
        self.device = device
        self._model = None
        self._transform = None
        self._weights_path = Path(weights_path) if weights_path else None
        self._freeze_backbone = freeze_backbone

    # ------------------------------------------------------------------ build
    def build(self) -> "TransferClassifier":
        """Instantiate the torchvision model (lazy; call before train/infer)."""
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torch and torchvision are required for TransferClassifier"
            ) from exc

        num_classes = len(self.class_ids)
        name = self.backbone_name.lower()

        if name in {"mobilenet_v3_small", "mobilenetv3", "mobilenet"}:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
            backbone_params: Iterable = model.features.parameters()
            self._transform = weights.transforms()
            resolved = "mobilenet_v3_small"
        elif name in {"resnet50", "resnet"}:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
            backbone_params = (
                p for n, p in model.named_parameters() if not n.startswith("fc.")
            )
            self._transform = weights.transforms()
            resolved = "resnet50"
        else:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. "
                "Use 'mobilenet_v3_small' or 'resnet50'."
            )

        if self._freeze_backbone:
            for p in backbone_params:
                p.requires_grad = False

        if self._weights_path is not None:
            state = torch.load(self._weights_path, map_location="cpu")
            model.load_state_dict(state)

        model.to(self.device)
        model.eval()
        self._model = model
        self.backbone_name = resolved
        return self

    def unfreeze_backbone(self) -> None:
        """Stage 2 of freeze-then-finetune: enable gradients on all parameters."""
        if self._model is None:
            raise RuntimeError("Call build() first")
        for p in self._model.parameters():
            p.requires_grad = True
        self._freeze_backbone = False

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    # ----------------------------------------------------------------- infer
    def predict(self, image: np.ndarray) -> ClassifierResult:
        """Run softmax classification on an RGB uint8 HxWx3 image."""
        import torch
        from PIL import Image

        model = self.model
        assert self._transform is not None
        pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return ClassifierResult(
            category=self.class_ids[idx],
            confidence=float(probs[idx]),
            probabilities=tuple(float(x) for x in probs),
            backbone=self.backbone_name,
        )

    def save(self, path: PathLike) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


class MockClassifier:
    """Deterministic stand-in for tests and offline demos (no torch weights)."""

    def __init__(
        self,
        category: str = "pet-bottle",
        confidence: float = 0.92,
        class_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.category = category
        self.confidence = confidence
        self.class_ids = list(class_ids or config.CLASS_IDS)

    def predict(self, image: np.ndarray) -> ClassifierResult:  # noqa: ARG002
        probs = [0.0] * len(self.class_ids)
        if self.category in self.class_ids:
            probs[self.class_ids.index(self.category)] = self.confidence
        else:
            probs[0] = self.confidence
        # redistribute residual mass uniformly
        residual = max(0.0, 1.0 - self.confidence)
        others = [i for i, c in enumerate(self.class_ids) if c != self.category]
        if others and residual:
            share = residual / len(others)
            for i in others:
                probs[i] = share
        return ClassifierResult(
            category=self.category,
            confidence=self.confidence,
            probabilities=tuple(probs),
            backbone="mock",
        )
