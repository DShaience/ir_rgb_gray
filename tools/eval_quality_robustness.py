"""Check how much classification accuracy degrades on low-quality/low-res
versions of the labeled samples in data/manifest.csv. Writes the degraded
copies to data/samples_degraded/<severity>/ so you can also look at them.

Usage:
    conda run -n ir_rgb_gray python tools/eval_quality_robustness.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_rgb_gray.augment import SEVERITY_PRESETS, simulate_low_quality
from ir_rgb_gray.classify import ImageClassifier
from ir_rgb_gray.model import read_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.csv")
DEGRADED_ROOT = os.path.join(ROOT, "data", "samples_degraded")


def run_pass(classifier: ImageClassifier, rows: list[tuple[str, str]], severity: str | None, rng: np.random.Generator):
    correct = 0
    confusions = {}
    for path, label in rows:
        if severity is None:
            eval_path = path
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            degraded = simulate_low_quality(img, severity=severity, rng=rng)
            out_dir = os.path.join(DEGRADED_ROOT, severity, label)
            os.makedirs(out_dir, exist_ok=True)
            eval_path = os.path.join(out_dir, os.path.basename(path))
            cv2.imwrite(eval_path, degraded)

        result = classifier.classify(eval_path)
        if result.label == label:
            correct += 1
        else:
            confusions.setdefault(label, {}).setdefault(result.label, 0)
            confusions[label][result.label] += 1
    return correct, len(rows), confusions


def main() -> None:
    rows = read_manifest(MANIFEST_PATH)
    classifier = ImageClassifier()
    rng = np.random.default_rng(0)

    print(f"evaluating {len(rows)} labeled real images at each quality level\n")

    for severity in [None, *SEVERITY_PRESETS.keys()]:
        label = severity or "original"
        correct, total, confusions = run_pass(classifier, rows, severity, rng)
        print(f"{label:10s}: {correct}/{total} correct ({100 * correct / total:.0f}%)")
        for true_label, wrong in confusions.items():
            wrong_str = ", ".join(f"{v}x->{k}" for k, v in wrong.items())
            print(f"    {true_label}: {wrong_str}")
    print(f"\ndegraded copies written under {DEGRADED_ROOT}")


if __name__ == "__main__":
    main()
