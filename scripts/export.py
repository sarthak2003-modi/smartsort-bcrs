#!/usr/bin/env python3
"""Export classifier / detector weights to ONNX (and TFLite when tooling allows)."""

from __future__ import annotations

import argparse
from pathlib import Path

from smartsort import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier", type=Path, default=None, help="classifier .pt state_dict")
    p.add_argument("--backbone", default=config.DEFAULT_CLASSIFIER_BACKBONE)
    p.add_argument(
        "--detector-weights",
        default=config.DEFAULT_DETECTOR_WEIGHTS,
        help=f"YOLO weights (default {config.DEFAULT_DETECTOR_WEIGHTS}; "
        f"alt {config.ALTERNATIVE_DETECTOR_WEIGHTS})",
    )
    p.add_argument("--export-detector", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path("artifacts/export"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--tflite", action="store_true", help="Also attempt TFLite via onnx-tf if installed")
    return p.parse_args()


def export_classifier(args: argparse.Namespace) -> Path:
    import torch

    from smartsort.models.classifier import TransferClassifier

    clf = TransferClassifier(
        backbone=args.backbone,
        weights_path=args.classifier,
        freeze_backbone=True,
        device=args.device,
    ).build()
    model = clf.model.eval()
    dummy = torch.randn(1, 3, config.CLASSIFIER_IMAGE_SIZE, config.CLASSIFIER_IMAGE_SIZE, device=args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{clf.backbone_name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Wrote {out}")
    if args.tflite:
        print(
            "TFLite conversion is optional and depends on onnx2tf / tensorflow; "
            "not attempted automatically to keep CPU deps light."
        )
    return out


def export_detector(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"ultralytics required for detector export: {exc}") from exc
    model = YOLO(args.detector_weights)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model.export(format="onnx", imgsz=640)
    print(
        f"Exported detector from {args.detector_weights} "
        f"(YOLOv8n default; YOLO11n via --detector-weights {config.ALTERNATIVE_DETECTOR_WEIGHTS})"
    )


def main() -> None:
    args = parse_args()
    if args.classifier is not None or not args.export_detector:
        # Allow exporting randomly-initialised head + ImageNet backbone for plumbing tests
        export_classifier(args)
    if args.export_detector:
        export_detector(args)


if __name__ == "__main__":
    main()
