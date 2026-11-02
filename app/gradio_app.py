#!/usr/bin/env python3
"""Gradio trial bay mirroring the SmartSort browser prototype."""

from __future__ import annotations

# Allow running as `python scripts/...` or `python app/...` without editable install.
import sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path

import numpy as np

from smartsort import config
from smartsort.models.classifier import MockClassifier, TransferClassifier
from smartsort.oplog import OperatorLog
from smartsort.pipeline import evaluate_return
from smartsort.sensors.material import SimulatedMaterialSensor


def build_app(use_torch: bool = False):
    import gradio as gr

    log = OperatorLog(Path("artifacts/gradio_oplog.jsonl"))

    def run(image, weight_g, mock_category, mock_conf):
        if image is None:
            return "Upload a photograph of the return.", "", ""
        if isinstance(image, np.ndarray) and image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = np.asarray(image, dtype=np.uint8)[..., :3]

        if use_torch:
            clf = TransferClassifier(freeze_backbone=True)
            result = clf.predict(image)
            classify_fn = lambda img: (result.category, result.confidence)  # noqa: E731
            used = f"torch:{result.backbone}"
        else:
            mock = MockClassifier(category=mock_category, confidence=float(mock_conf))
            classify_fn = lambda img: (mock.predict(img).category, mock.predict(img).confidence)  # noqa: E731
            used = "mock"

        material = SimulatedMaterialSensor(
            material="pet" if mock_category == "pet-bottle" else "aluminium"
            if mock_category == "aluminium-can"
            else "other"
        ).read()
        decision = evaluate_return(image, float(weight_g), classify_fn=classify_fn, material_reading=material)
        log.append(decision, extra={"ui": "gradio", "classifier": used})
        verdict = (
            f"{'✅ PAY' if decision.action == 'PAY' else '🛑 HOLD'} · "
            f"{decision.label} · refund {decision.refund_cents}¢"
        )
        detail = (
            f"cv_conf={decision.cv_confidence:.3f} · fusion={decision.fusion_score:.3f} · "
            f"weight={decision.weight_g}g · latency={decision.latency_ms:.1f}ms\n"
            f"reasons: {', '.join(decision.reasons)}\n"
            f"{config.THRESHOLDS_NOTE}"
        )
        recent = log.read_all()[-8:]
        log_text = "\n".join(
            f"{e['at']} · {e['action']} · {e.get('label')} · {e['weight_g']}g · {e['refund_cents']}¢"
            for e in reversed(recent)
        )
        return verdict, detail, log_text

    with gr.Blocks(title="SmartSort Trial Bay") as demo:
        gr.Markdown(
            """
            # SmartSort · Trial Bay
            Photograph a return, enter its weight, and get a category then PAY 10¢ or HOLD.
            Classes align with the public trial app: `pet-bottle`, `aluminium-can`, `rubbish`,
            `food-waste`, `non-eligible`, `crushed`, `tampered`, `unknown`.
            Empty PET **8–70 g** · empty can **10–25 g** (illustrative bands).
            """
        )
        with gr.Row():
            image = gr.Image(type="numpy", label="Chute photograph")
            with gr.Column():
                weight = gr.Number(value=18.0, label="Weight (grams)")
                mock_cat = gr.Dropdown(
                    choices=list(config.CLASS_IDS),
                    value="pet-bottle",
                    label="Mock category (used when torch classifier is off)",
                )
                mock_conf = gr.Slider(0.0, 1.0, value=0.92, step=0.01, label="Mock CV confidence")
                btn = gr.Button("Evaluate return", variant="primary")
        verdict = gr.Textbox(label="Verdict")
        detail = gr.Textbox(label="Detail")
        oplog = gr.Textbox(label="Operator log (local)", lines=8)
        btn.click(run, [image, weight, mock_cat, mock_conf], [verdict, detail, oplog])
    return demo


if __name__ == "__main__":
    build_app(use_torch=False).launch()
