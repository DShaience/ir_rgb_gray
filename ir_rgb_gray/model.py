"""Train the stage-B (grayscale / ir_whitehot / ir_blackhot) classifier
from a labeled manifest CSV produced by tools/label_tool.py.
"""
from __future__ import annotations

import csv
import os

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from .augment import SEVERITY_PRESETS, simulate_low_quality
from .classify import COLOR_CHANNEL_DIFF_THRESHOLD, DEFAULT_MODEL_PATH
from .features import FEATURE_NAMES, channel_color_stats, load_image, monochrome_features


def read_manifest(manifest_csv: str) -> list[tuple[str, str]]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return [(row["path"], row["label"]) for row in csv.DictReader(f)]


def _features_from_gray(gray: np.ndarray) -> list[float]:
    feats = monochrome_features(gray)
    return [feats[name] for name in FEATURE_NAMES]


def build_dataset(manifest_csv: str, augment_degraded: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features for every non-rgb labeled image.

    With augment_degraded=True, each image also contributes one mild- and
    one heavy-degraded copy (see ir_rgb_gray.augment) so the model sees what
    low-quality/low-resolution captures look like, not just clean ones. The
    returned `groups` list tags augmented copies with their source path so
    train() can keep them in the same CV fold and avoid leakage.
    """
    rng = np.random.default_rng(0)
    X, y, groups = [], [], []
    for path, label in read_manifest(manifest_csv):
        if label == "rgb":
            continue  # stage A (channel-diff check) handles rgb, not this model
        try:
            bgr, meta = load_image(path)
        except ValueError as exc:
            print(f"skipping {path}: {exc}")
            continue
        gray = bgr[:, :, 0] if meta["channels"] == 1 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        X.append(_features_from_gray(gray))
        y.append(label)
        groups.append(path)

        if augment_degraded:
            for severity in SEVERITY_PRESETS:
                degraded = simulate_low_quality(gray, severity=severity, rng=rng)
                X.append(_features_from_gray(degraded))
                y.append(label)
                groups.append(path)

    return np.array(X), np.array(y), groups


def train(
    manifest_csv: str,
    out_path: str = DEFAULT_MODEL_PATH,
    n_splits: int = 5,
    augment_degraded: bool = False,
) -> None:
    X, y, groups = build_dataset(manifest_csv, augment_degraded=augment_degraded)
    if len(set(y)) < 2:
        raise ValueError(
            f"need at least 2 distinct monochrome labels to train, got {sorted(set(y))} "
            f"from {len(y)} labeled examples -- label more images with tools/label_tool.py"
        )

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    n_splits = max(2, min(n_splits, int(np.min(np.bincount(y_enc)))))
    model = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0)

    if augment_degraded:
        # group by source path so augmented copies of the same image never
        # split across train/test -- otherwise the score is inflated by
        # near-duplicate leakage rather than real generalization
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_pred = cross_val_predict(model, X, y_enc, cv=cv, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_pred = cross_val_predict(model, X, y_enc, cv=cv)
    print(f"cross-validated report ({n_splits}-fold, n={len(y)}, augment_degraded={augment_degraded}):")
    print(classification_report(y_enc, y_pred, target_names=encoder.classes_, zero_division=0))

    model.fit(X, y_enc)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": model, "label_encoder": encoder, "feature_names": FEATURE_NAMES}, out_path)
    print(f"saved model to {out_path}")


def suggest_color_threshold(manifest_csv: str) -> None:
    """Report the mean_channel_diff distribution for labeled rgb vs everything
    else, so COLOR_CHANNEL_DIFF_THRESHOLD in classify.py can be retuned if needed.
    """
    rgb_diffs, mono_diffs = [], []
    for path, label in read_manifest(manifest_csv):
        try:
            bgr, _ = load_image(path)
        except ValueError as exc:
            print(f"skipping {path}: {exc}")
            continue
        diff = channel_color_stats(bgr)["mean_channel_diff"]
        (rgb_diffs if label == "rgb" else mono_diffs).append(diff)

    if rgb_diffs:
        print(f"rgb examples (n={len(rgb_diffs)}): mean_channel_diff min={min(rgb_diffs):.2f} max={max(rgb_diffs):.2f}")
    if mono_diffs:
        print(f"monochrome examples (n={len(mono_diffs)}): mean_channel_diff min={min(mono_diffs):.2f} max={max(mono_diffs):.2f}")
    print(f"current threshold: {COLOR_CHANNEL_DIFF_THRESHOLD}")
    if rgb_diffs and mono_diffs and min(rgb_diffs) <= max(mono_diffs):
        print("warning: rgb and monochrome mean_channel_diff ranges overlap -- inspect these images")
