# Methodology

## Computer vision

### 1. Object detection & classification (YOLO-family / CNN)

A YOLO-family detector recognises container **type, shape and condition**
directly from the chute image (intact bottle/can vs crushed or tampered packs,
plus obvious rubbish / food waste).

**Default weights:** Ultralytics **YOLOv8n** (`yolov8n.pt`) — small enough for
edge pilots. **YOLO11n** (`yolo11n.pt`) is a drop-in alternative: pass it as the
`weights` / `--weights` argument with no other code changes
(`smartsort/models/detector.py`, `scripts/train_detector.py`).

If `ultralytics` is not installed, the detector wrapper returns an empty result
with a clear reason string instead of crashing.

Classification into SmartSort categories uses a separate transfer-learning head
(below). Category IDs match the live trial app:

| ID | Label |
|----|-------|
| `pet-bottle` | PET bottle |
| `aluminium-can` | Aluminium can |
| `rubbish` | Rubbish |
| `food-waste` | Food waste |
| `non-eligible` | Non-eligible pack |
| `crushed` | Crushed container |
| `tampered` | Tampered container |
| `unknown` | Unknown |

Eligible for payout: **`pet-bottle`**, **`aluminium-can`** only.

### 2. Transfer learning on edge-ready backbones

Pretrained **MobileNetV3-Small** (default) or **ResNet50** ImageNet backbones
from `torchvision` are fine-tuned on BCRS item photos:

1. **Freeze** the feature extractor; train the classification head.
2. **Unfreeze** and fine-tune at a lower learning rate.

This keeps inference realistic for low-power RVM hardware. See
`smartsort/models/classifier.py` and `scripts/train_classifier.py`.

> This repository does **not** ship fine-tuned BCRS weights or a dataset.
> ImageNet / COCO pretrained weights download at runtime from torchvision /
> Ultralytics when you build or train.

### 3. OCR & deposit-mark scanning

Optical character recognition plus barcode decode confirm a genuine participating
BCRS mark / deposit legend. Implementation (`smartsort/models/ocr.py`) optionally
uses:

- `pyzbar` for barcodes
- EasyOCR (preferred) or `pytesseract` for text

Keywords include `BCRS`, `RETURN RIGHT`, `10 CENTS` / `10¢`, `DEPOSIT`. Missing
optional deps degrade gracefully.

## Sensors

Vision alone can be fooled by a contaminant shaped like a bottle — so SmartSort
fuses physical signals.

### Load-cell weight

Empty-container **illustrative** bands (deck):

| Category | Band (g) |
|----------|----------|
| PET bottle | 8–70 |
| Aluminium can | 10–25 |

`ManualWeightEntry` stands in for a physical load cell in the trial bay
(same approach as the browser prototype).

### NIR / capacitive material sensing

Confirms PET plastic or aluminium vs paper, glass, or other. Hardware interface
is stubbed; `SimulatedMaterialSensor` supplies readings for demos and tests.

### Fusion layer

`weight_fit(weight_g, band)` → [0, 1] score (1 at band centre, 0 at/outside edges).  
`fuse(cv_conf, weight_score, material_score?)` → one confidence score.  
PAY if score ≥ `ACCEPT_MIN`, else HOLD.

All thresholds in `smartsort/config.py` are labelled **illustrative / uncalibrated**.
