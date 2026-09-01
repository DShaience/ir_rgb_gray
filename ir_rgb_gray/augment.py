"""Simulate low-quality/low-resolution capture conditions, to test (or later
train) robustness against images degraded the way real cheap IR/RGB sensors
and heavy video compression degrade them: resolution loss, sensor noise, and
JPEG block artifacts.
"""
from __future__ import annotations

import cv2
import numpy as np

SEVERITY_PRESETS = {
    "mild": {"scale": 0.5, "noise_sigma": 4, "jpeg_quality": 40},
    "heavy": {"scale": 0.22, "noise_sigma": 10, "jpeg_quality": 15},
}


def simulate_low_quality(img: np.ndarray, severity: str = "mild", rng: np.random.Generator | None = None) -> np.ndarray:
    """Downscale-then-upscale (resolution loss), add sensor noise, and
    re-encode through JPEG (compression artifacts). Returns an array the
    same shape/dtype as the input so it drops straight into classify_image's
    feature pipeline.
    """
    if rng is None:
        rng = np.random.default_rng()
    params = SEVERITY_PRESETS[severity]
    h, w = img.shape[:2]

    small_w, small_h = max(1, int(w * params["scale"])), max(1, int(h * params["scale"]))
    degraded = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

    noise = rng.normal(0, params["noise_sigma"], degraded.shape)
    degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg_quality"]])
    if ok:
        degraded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return degraded
