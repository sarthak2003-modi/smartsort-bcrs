"""01 · Capture — load chute-camera frames for the SmartSort pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


def load_image(source: PathLike) -> np.ndarray:
    """
    Load an RGB image from disk.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB array.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required for capture.load_image") from exc

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def load_image_from_array(array: np.ndarray) -> np.ndarray:
    """Validate and normalise an in-memory image to HxWx3 uint8 RGB."""
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 array, got shape {array.shape}")
    out = array[..., :3]
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def open_camera(device_index: int = 0, width: int = 640, height: int = 480):
    """
    Open a chute / webcam capture via OpenCV (optional dependency).

    Returns an ``cv2.VideoCapture`` instance. Caller must ``.release()``.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for open_camera; pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {device_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def grab_frame(cap) -> np.ndarray:
    """Grab one BGR frame from an OpenCV capture and return RGB uint8."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to grab camera frame")
    # OpenCV is BGR
    return frame[:, :, ::-1].copy()


def synthetic_chute_image(
    seed: int = 0,
    size: int = 224,
    colour: Optional[tuple] = None,
) -> np.ndarray:
    """Create a synthetic RGB image for demos / tests (no real container)."""
    rng = np.random.default_rng(seed)
    if colour is None:
        colour = tuple(int(x) for x in rng.integers(40, 200, size=3))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = colour
    # Draw a simple "bottle-like" rectangle for visual sanity checks
    margin = size // 4
    img[margin : size - margin, size // 3 : 2 * size // 3] = (
        min(255, colour[0] + 40),
        min(255, colour[1] + 40),
        min(255, colour[2] + 40),
    )
    return img
