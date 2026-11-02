"""Streamlit trial bay — run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from smartsort import config
from smartsort.models.classifier import MockClassifier
from smartsort.oplog import OperatorLog
from smartsort.pipeline import evaluate_return
from smartsort.sensors.material import SimulatedMaterialSensor

st.set_page_config(page_title="SmartSort Trial Bay", layout="centered")
st.title("SmartSort · Trial Bay")
st.caption(config.THRESHOLDS_NOTE)

uploaded = st.file_uploader("Chute photograph", type=["jpg", "jpeg", "png", "webp"])
weight_g = st.number_input("Weight (grams)", min_value=1.0, max_value=5000.0, value=18.0)
category = st.selectbox("Mock category", list(config.CLASS_IDS), index=0)
cv_conf = st.slider("Mock CV confidence", 0.0, 1.0, 0.92)

if st.button("Evaluate return") and uploaded is not None:
    image = np.asarray(Image.open(uploaded).convert("RGB"), dtype=np.uint8)
    mock = MockClassifier(category=category, confidence=cv_conf)
    material = SimulatedMaterialSensor(
        material="pet" if category == "pet-bottle" else "aluminium"
        if category == "aluminium-can"
        else "other"
    ).read()
    decision = evaluate_return(
        image,
        float(weight_g),
        classify_fn=lambda img: (mock.predict(img).category, mock.predict(img).confidence),
        material_reading=material,
    )
    OperatorLog(Path("artifacts/streamlit_oplog.jsonl")).append(decision)
    if decision.action == "PAY":
        st.success(f"PAY · refund {decision.refund_cents}¢ · {decision.label}")
    else:
        st.error(f"HOLD · {', '.join(decision.reasons)}")
    st.json(decision.to_dict())
elif uploaded is None:
    st.info("Upload a photograph to run a check.")
