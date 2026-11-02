#!/usr/bin/env python3
"""Prepare a folder-per-class dataset with train/val split and light augmentation."""

from __future__ import annotations

# Allow running as `python scripts/...` or `python app/...` without editable install.
import sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import random
import shutil
from pathlib import Path

from smartsort import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Root with one subfolder per class")
    p.add_argument("--dst", type=Path, default=Path("data/processed"))
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-copies", type=int, default=0, help="Extra PIL-augmented copies per train image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    class_dirs = [d for d in sorted(args.src.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class folders under {args.src}")

    for split in ("train", "val"):
        for d in class_dirs:
            (args.dst / split / d.name).mkdir(parents=True, exist_ok=True)

    for cls_dir in class_dirs:
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        rng.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio)) if images else 0
        val_set = set(images[:n_val])
        for img in images:
            split = "val" if img in val_set else "train"
            dest = args.dst / split / cls_dir.name / img.name
            shutil.copy2(img, dest)
            if split == "train" and args.augment_copies > 0:
                _augment(img, args.dst / split / cls_dir.name, args.augment_copies, rng)

    print(f"Prepared dataset at {args.dst}")
    print(f"Expected class folders include: {', '.join(config.CLASS_IDS)}")


def _augment(src: Path, dst_dir: Path, copies: int, rng: random.Random) -> None:
    try:
        from PIL import Image, ImageEnhance, ImageOps
    except ImportError:
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        for i in range(copies):
            out = im.copy()
            if rng.random() < 0.5:
                out = ImageOps.mirror(out)
            if rng.random() < 0.5:
                out = out.rotate(rng.uniform(-15, 15), expand=True, fillcolor=(0, 0, 0))
            out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
            out.save(dst_dir / f"{src.stem}_aug{i}{src.suffix}")


if __name__ == "__main__":
    main()
