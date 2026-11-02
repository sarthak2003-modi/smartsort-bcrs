#!/usr/bin/env python3
"""
End-to-end SmartSort demo CLI.

Uses a mocked classifier by default so no GPU / fine-tuned weights are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smartsort.capture import load_image, synthetic_chute_image
from smartsort.models.classifier import MockClassifier
from smartsort.oplog import OperatorLog
from smartsort.pipeline import evaluate_return
from smartsort.sensors.material import SimulatedMaterialSensor
from smartsort.sensors.weight import ManualWeightEntry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=None, help="RGB image path (optional)")
    p.add_argument("--weight-g", type=float, default=18.0)
    p.add_argument("--category", default="pet-bottle", help="Mock classifier category")
    p.add_argument("--cv-conf", type=float, default=0.92)
    p.add_argument("--material", default="pet", help="Simulated material (pet|aluminium|paper|...)")
    p.add_argument("--no-material", action="store_true")
    p.add_argument("--log", type=Path, default=Path("artifacts/oplog.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        image = load_image(args.image)
    else:
        image = synthetic_chute_image(seed=args.seed)
        print(f"[demo] using synthetic image shape={image.shape}")

    weight = ManualWeightEntry(args.weight_g).read()
    if not weight.ok:
        raise SystemExit(weight.message)

    mock = MockClassifier(category=args.category, confidence=args.cv_conf)

    material = None
    if not args.no_material:
        material = SimulatedMaterialSensor(material=args.material).read()

    decision = evaluate_return(
        image,
        weight.grams,
        classify_fn=lambda img: (mock.predict(img).category, mock.predict(img).confidence),
        material_reading=material,
    )

    log = OperatorLog(args.log)
    entry = log.append(decision, extra={"demo": True, "weight_source": weight.source})
    print(json.dumps({"decision": decision.to_dict(), "log_id": entry["id"]}, indent=2))


if __name__ == "__main__":
    main()
