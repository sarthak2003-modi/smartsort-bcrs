<p align="center">
  <img src="docs/assets/hero-banner.png" alt="SmartSort — AI vision + sensor fusion for fraud-proof BCRS returns" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://docs.ultralytics.com/"><img src="https://img.shields.io/badge/YOLOv8-00A6ED?logo=yolo&logoColor=white" alt="YOLOv8"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E" alt="MIT licence"/></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-22%20passing-16A34A" alt="22 tests passing"/></a>
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app/trial"><img src="https://img.shields.io/badge/Live%20demo-try%20it-0D9488?logo=vercel&logoColor=white" alt="Live demo"/></a>
</p>

<p align="center">
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app/trial"><strong>▶ Try the live demo</strong></a>
  &nbsp;·&nbsp;
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app/trial">Trial bay</a>
  &nbsp;·&nbsp;
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app/log">Operator log</a>
  &nbsp;·&nbsp;
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app">Site root</a>
</p>

**AI vision + sensor fusion for fraud detection in Singapore's Beverage Container Return Scheme (BCRS).**

SmartSort sits inside a Return Right reverse-vending-machine (RVM) chute and
confirms an item is genuine **before** the 10¢ refund is released — photograph,
classify, weigh, fuse, then **PAY** or **HOLD**.

> NTU AI Literacy group project · Group of 6 · AI Exploration Project  
> Credits: **Bui Le Phuong**, **Wong Qi En Joann**, **Wang Zhishan**, **Shi Jiaxiao**,
> **Xu Mengyun**, **Modi Sarthak Ketanbhai**

**Important:** thresholds are **illustrative / uncalibrated**. No fine-tuned
weights or measured accuracy numbers are shipped. Design targets only — never
fabricated results. The **live demo uses a hosted vision-language model as a
stand-in** for on-device vision; this Python repo is the offline / trainable
counterpart.

---

## Pipeline

<p align="center">
  <img src="docs/assets/pipeline.png" alt="SmartSort pipeline: Capture → Classify → Weigh & sense → Fuse & decide → PAY 10¢ or HOLD" width="100%"/>
</p>

| Step | Module | Role |
|------|--------|------|
| 01 Capture | `smartsort/capture.py` | Chute photograph (file, camera, or synthetic) |
| 02 Classify | `smartsort/models/` | YOLO detection + MobileNet/ResNet + OCR |
| 03 Weigh & sense | `smartsort/sensors/` | Load-cell bands + NIR / capacitive |
| 04 Fuse & decide | `fusion.py`, `pipeline.py` | One score → PAY 10¢ or HOLD |

### Classes (aligned with the public trial bay)

| ID | Eligible? |
|----|-----------|
| `pet-bottle` | yes (empty 8–70 g) |
| `aluminium-can` | yes (empty 10–25 g) |
| `rubbish`, `food-waste`, `non-eligible`, `crushed`, `tampered`, `unknown` | no → HOLD |

---

## Live demo gallery

Real screenshots from the hosted trial
([temporary-express-oasis-aafsrpj.vercel.app](https://temporary-express-oasis-aafsrpj.vercel.app)):

| Landing | PAY at 25 g |
|:-------:|:-----------:|
| <img src="docs/assets/demo-landing.png" alt="SmartSort landing page" width="420"/> | <img src="docs/assets/demo-pay.png" alt="Eligible PET bottle — refund 10¢" width="420"/> |
| Site root · Start the trial | PET bottle · 25 g · Eligible · Refund 10¢ |

| HOLD at 200 g | Operator log |
|:-------------:|:------------:|
| <img src="docs/assets/demo-hold.png" alt="Weight mismatch HOLD at 200 g" width="420"/> | <img src="docs/assets/demo-oplog.png" alt="Operator log with PAY and HOLD entries" width="420"/> |
| PET bottle · 200 g · Weight check failed · 0¢ | Trial log · PAY and HOLD side by side |

<p align="center">
  <a href="https://temporary-express-oasis-aafsrpj.vercel.app/trial"><strong>▶ Open the trial bay</strong></a>
</p>

---

## Computer vision methods

<p align="center">
  <img src="docs/assets/cv-methods.png" alt="YOLOv8n detection, MobileNetV3/ResNet-50 transfer learning, OCR deposit mark" width="100%"/>
</p>

| Piece | Default | Notes |
|-------|---------|-------|
| Detector | **YOLOv8n** (`yolov8n.pt`) | Ultralytics; **YOLO11n** is a drop-in alternative |
| Classifier | MobileNetV3-Small (or ResNet50) | Freeze-then-finetune transfer learning |
| OCR / barcode | EasyOCR or pytesseract + pyzbar | Optional imports; graceful fallback |

---

## Decision logic

<details>
<summary><strong>evaluate_return flowchart</strong> (CONF_MIN → eligible → weight band → fusion → ACCEPT_MIN)</summary>

<br/>

<p align="center">
  <img src="docs/assets/decision-logic.png" alt="evaluate_return decision flowchart" width="720"/>
</p>

```text
function evaluate_return(image, weight_g):
    category, cv_conf = classify(image)
    if cv_conf < CONF_MIN:
        return HOLD("low vision confidence")
    if category not in ELIGIBLE_CATEGORIES:
        return HOLD("ineligible: " + category)
    band = WEIGHT_BAND[category]
    if weight_g outside band:
        return HOLD("weight mismatch")
    score = fuse(cv_conf, weight_fit(weight_g, band))
    if score >= ACCEPT_MIN:
        return PAY(0.10)
    return HOLD("fusion score too low")
```

Thresholds (`CONF_MIN = 0.55`, `ACCEPT_MIN = 0.70`) are **illustrative**.

</details>

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # so `python scripts/...` can import the smartsort package

# Unit tests (mocked classifier — no GPU / weights needed)
pytest -q

# End-to-end demo CLI with synthetic image + mock classifier
python scripts/demo_cli.py --weight-g 18 --category pet-bottle --cv-conf 0.92

# Gradio trial bay
pip install gradio
python app/gradio_app.py
```

### Train / eval / export

```bash
# After you add your own images under data/raw/<class_id>/
python scripts/prepare_dataset.py --src data/raw --dst data/processed
python scripts/train_classifier.py --data data/processed --backbone mobilenet_v3_small

# YOLO detector (default yolov8n.pt; use --weights yolo11n.pt for YOLO11n)
python scripts/train_detector.py --write-template-only
python scripts/train_detector.py --weights yolov8n.pt

python scripts/evaluate.py --cases path/to/cases.json
python scripts/export.py --outdir artifacts/export
python scripts/export.py --export-detector --detector-weights yolov8n.pt
```

---

## Project structure

```text
smartsort-bcrs/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── smartsort/
│   ├── config.py          # classes, bands, CONF_MIN, ACCEPT_MIN (illustrative)
│   ├── capture.py         # 01 camera / image loading
│   ├── fusion.py          # weight_fit + fuse
│   ├── pipeline.py        # evaluate_return → PAY / HOLD
│   ├── oplog.py           # operator JSONL / CSV log
│   ├── metrics.py         # accuracy / FPR / FNR / latency vs targets
│   ├── models/
│   │   ├── classifier.py  # MobileNetV3 / ResNet50 transfer learning
│   │   ├── detector.py    # YOLOv8n wrapper (YOLO11n drop-in)
│   │   └── ocr.py         # barcode + deposit-mark OCR
│   └── sensors/
│       ├── weight.py      # load cell + manual-entry stub
│       └── material.py    # NIR / capacitive + simulated stub
├── scripts/               # prepare_dataset, train_*, evaluate, export, demo_cli
├── app/                   # Gradio + Streamlit trial bay UIs
├── tests/                 # pytest (pipeline, fusion, weight; mocked CV)
└── docs/
    ├── assets/            # hero, pipeline, CV, decision, targets, demo shots
    ├── architecture.md
    ├── methodology.md
    ├── dataset.md
    ├── ethics_and_limitations.md
    ├── results_targets.md
    └── speaker_script_slides_4_5.md
```

---

## Design targets (not measured results)

<p align="center">
  <img src="docs/assets/targets.png" alt="Design targets: accuracy ≥95%, FPR <2%, FNR <5%, latency <2 s — not measured" width="100%"/>
</p>

| Metric | Target |
|--------|--------|
| Vision accuracy | ≥ 95% |
| FPR (fraud paid) | < 2% |
| FNR (valid held) | < 5% |
| Latency | < 2 s |

See [`docs/results_targets.md`](docs/results_targets.md) and
[`docs/ethics_and_limitations.md`](docs/ethics_and_limitations.md).

---

## Responsible AI (summary)

- **Bias:** training data must span brands, lighting and wear, or valid returns
  can be wrongly held.
- **Privacy:** camera frames the **chute only** — no bystanders, no facial
  recognition; retain images only as long as the check needs.
- **Human review:** HOLD triggers an operator path, not an unexplained denial.
- **Honesty:** the live trial uses a **hosted VLM stand-in**; this repo ships
  **no trained weights** and **no measured accuracy numbers**.

Full notes: [`docs/ethics_and_limitations.md`](docs/ethics_and_limitations.md).

---

## License

MIT — see [LICENSE](LICENSE).

BCRS / Return Right / NEA references are for educational context in an NTU
coursework prototype and are not an official partnership.
