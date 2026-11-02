"""
OCR + barcode / deposit-mark scanning for participating BCRS marks.

Optional imports: ``pyzbar`` (barcodes), ``easyocr`` or ``pytesseract`` (OCR).
When libraries are missing the helpers return ``available=False`` rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from smartsort import config


@dataclass(frozen=True)
class DepositMarkResult:
    found: bool
    texts: Tuple[str, ...]
    barcodes: Tuple[str, ...]
    available: bool
    reason: str = ""


def _ocr_easyocr(image: np.ndarray) -> Tuple[List[str], Optional[str]]:
    try:
        import easyocr
    except ImportError as exc:
        return [], f"easyocr not installed ({exc})"
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rows = reader.readtext(image)
    texts = [str(t[1]) for t in rows if t and len(t) >= 2]
    return texts, None


def _ocr_tesseract(image: np.ndarray) -> Tuple[List[str], Optional[str]]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        return [], f"pytesseract/Pillow not installed ({exc})"
    text = pytesseract.image_to_string(Image.fromarray(image))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines, None


def _decode_barcodes(image: np.ndarray) -> Tuple[List[str], Optional[str]]:
    try:
        from pyzbar import pyzbar
        from PIL import Image
    except ImportError as exc:
        return [], f"pyzbar not installed ({exc})"
    pil = Image.fromarray(image)
    decoded = pyzbar.decode(pil)
    return [d.data.decode("utf-8", errors="replace") for d in decoded], None


def scan_deposit_mark(
    image: np.ndarray,
    keywords: Optional[Sequence[str]] = None,
) -> DepositMarkResult:
    """
    Look for a participating BCRS deposit mark / barcode on the container image.

    Tries EasyOCR first, then pytesseract; barcodes via pyzbar when present.
    """
    keywords = list(keywords or config.BCRS_DEPOSIT_MARK_KEYWORDS)
    reasons: List[str] = []

    barcodes, bar_err = _decode_barcodes(image)
    if bar_err:
        reasons.append(bar_err)

    texts, ocr_err = _ocr_easyocr(image)
    if ocr_err:
        reasons.append(ocr_err)
        texts, ocr_err2 = _ocr_tesseract(image)
        if ocr_err2:
            reasons.append(ocr_err2)

    haystack = " ".join(texts + barcodes).upper()
    found = any(k.upper() in haystack for k in keywords)

    available = bool(texts or barcodes) or not reasons
    # If every backend missing, mark unavailable
    if not texts and not barcodes and reasons:
        available = False

    return DepositMarkResult(
        found=found,
        texts=tuple(texts),
        barcodes=tuple(barcodes),
        available=available,
        reason="; ".join(reasons),
    )
