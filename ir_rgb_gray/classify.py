"""Reusable classifier: rgb / grayscale / ir_whitehot / ir_blackhot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import FEATURE_NAMES, channel_color_stats, load_image, monochrome_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "monochrome_clf.joblib")

# An image is treated as real RGB only if it's stored with >=3 channels AND
# the channels actually diverge -- a 3-channel file with R==G==B is still
# visually monochrome and gets routed to the grayscale/IR stage instead.
# Retune with `python tools/train.py` once you've labeled some rgb + grayscale
# examples (it reports the observed mean_channel_diff distribution for both).
COLOR_CHANNEL_DIFF_THRESHOLD = 4.0  # out of 255


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    stage: str
    details: dict = field(default_factory=dict)


class ImageClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._label_encoder = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            import joblib

            bundle = joblib.load(self.model_path)
            self._model = bundle["model"]
            self._label_encoder = bundle["label_encoder"]

    def classify(self, path: str) -> ClassificationResult:
        bgr, meta = load_image(path)
        color_stats = channel_color_stats(bgr)

        is_color = meta["channels"] >= 3 and color_stats["mean_channel_diff"] > COLOR_CHANNEL_DIFF_THRESHOLD
        if is_color:
            return ClassificationResult(
                label="rgb",
                confidence=1.0,
                stage="deterministic-channel-check",
                details={**meta, **color_stats},
            )

        gray = bgr[:, :, 0] if meta["channels"] == 1 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mono_feats = monochrome_features(gray)
        label, confidence, stage = self._classify_monochrome(mono_feats)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            stage=stage,
            details={**meta, **color_stats, **mono_feats},
        )

    def _classify_monochrome(self, feats: dict) -> tuple[str, float, str]:
        if self._model is not None:
            vec = np.array([[feats[name] for name in FEATURE_NAMES]])
            proba = self._model.predict_proba(vec)[0]
            idx = int(np.argmax(proba))
            label = self._label_encoder.inverse_transform([idx])[0]
            return label, float(proba[idx]), "trained-model"
        return self._heuristic_monochrome(feats)

    @staticmethod
    def _heuristic_monochrome(feats: dict) -> tuple[str, float, str]:
        """No trained model yet -- rough rule-of-thumb used until you label a
        batch and run tools/train.py. Thermal imagery tends to be smoother
        (lower laplacian_var/edge_density) than an optical grayscale photo,
        and its "hot" side usually forms one compact blob covering a minority
        of the frame rather than being scattered texture or covering most of
        the image. Confidence is deliberately kept low here.
        """
        is_smooth = feats["laplacian_var"] < 150 and feats["edge_density"] < 0.08
        if not is_smooth:
            return "grayscale", 0.4, "heuristic-fallback"

        bright_is_compact_minority = (
            feats["bright_area_frac"] < 0.35 and feats["bright_largest_blob_frac"] > 0.5 * feats["bright_area_frac"]
        )
        dark_is_compact_minority = (
            feats["dark_area_frac"] < 0.35 and feats["dark_largest_blob_frac"] > 0.5 * feats["dark_area_frac"]
        )

        if bright_is_compact_minority and not dark_is_compact_minority:
            return "ir_whitehot", 0.4, "heuristic-fallback"
        if dark_is_compact_minority and not bright_is_compact_minority:
            return "ir_blackhot", 0.4, "heuristic-fallback"
        return "grayscale", 0.3, "heuristic-fallback"


_default_classifier: ImageClassifier | None = None


def classify_image(path: str) -> ClassificationResult:
    """Classify a single image as rgb / grayscale / ir_whitehot / ir_blackhot."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ImageClassifier()
    return _default_classifier.classify(path)
