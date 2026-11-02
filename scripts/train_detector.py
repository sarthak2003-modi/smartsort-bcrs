#!/usr/bin/env python3
"""
Train / fine-tune a YOLO detector via Ultralytics.

Default base weights: **yolov8n.pt** (YOLOv8n).
Pass ``--weights yolo11n.pt`` for YOLO11n (drop-in alternative).

Writes a data.yaml template if one does not exist.
"""

from __future__ import annotations

# Allow running as `python scripts/...` or `python app/...` without editable install.
import sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
from textwrap import dedent

from smartsort import config


DATA_YAML_TEMPLATE = dedent(
    f"""\
    # SmartSort YOLO dataset template
    # Default detector: {config.DEFAULT_DETECTOR_WEIGHTS} (YOLOv8n)
    # Drop-in alternative: {config.ALTERNATIVE_DETECTOR_WEIGHTS} (YOLO11n)
    path: data/yolo
    train: images/train
    val: images/val
    names:
      0: pet-bottle
      1: aluminium-can
      2: rubbish
      3: food-waste
      4: non-eligible
      5: crushed
      6: tampered
    """
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-yaml", type=Path, default=Path("data/yolo/data.yaml"))
    p.add_argument(
        "--weights",
        default=config.DEFAULT_DETECTOR_WEIGHTS,
        help=f"Base weights (default {config.DEFAULT_DETECTOR_WEIGHTS}; "
        f"try {config.ALTERNATIVE_DETECTOR_WEIGHTS})",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="artifacts/yolo")
    p.add_argument("--name", default="smartsort-detect")
    p.add_argument("--write-template-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_yaml.parent.mkdir(parents=True, exist_ok=True)
    if not args.data_yaml.exists() or args.write_template_only:
        args.data_yaml.write_text(DATA_YAML_TEMPLATE, encoding="utf-8")
        print(f"Wrote template {args.data_yaml}")
        if args.write_template_only:
            return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            f"ultralytics required: {exc}\n"
            f"Default weights: {config.DEFAULT_DETECTOR_WEIGHTS}; "
            f"alternative: {config.ALTERNATIVE_DETECTOR_WEIGHTS}"
        ) from exc

    if not (Path("data/yolo/images/train").exists()):
        raise SystemExit(
            "No YOLO images at data/yolo/images/train. "
            "Add labelled data or use --write-template-only. "
            "This repo does not ship a detection dataset."
        )

    model = YOLO(args.weights)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
