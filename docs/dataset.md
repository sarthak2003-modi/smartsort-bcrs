# Dataset

## Status

**No dataset is shipped with this repository.** The SmartSort deck notes that the
team self-collected and labelled a small sample set of valid containers and common
contaminants for the module prototype. Those images are not redistributed here.

## Expected layout

Folder-per-class (ImageFolder-compatible):

```
data/raw/
  pet-bottle/
  aluminium-can/
  rubbish/
  food-waste/
  non-eligible/
  crushed/
  tampered/
```

Prepare train/val splits (and optional PIL augmentations):

```bash
python scripts/prepare_dataset.py --src data/raw --dst data/processed --val-ratio 0.2 --augment-copies 2
```

## YOLO detection layout

```
data/yolo/
  data.yaml          # from scripts/train_detector.py --write-template-only
  images/train/
  images/val/
  labels/train/      # YOLO txt labels
  labels/val/
```

Default detector base weights: **`yolov8n.pt`**. Alternative: **`yolo11n.pt`**.

## Labelling guidance

- Capture chute-like framing and lighting where possible.
- Include worn labels, crumpled bottles, and partial occlusions.
- Span diverse brands so the model does not overfit a handful of SKUs (see ethics doc).
- Never include bystander faces; frame the chute only.
