"""Train (or retrain) the grayscale / ir_whitehot / ir_blackhot classifier.

Usage:
    conda run -n ir_rgb_gray python tools/train.py
    conda run -n ir_rgb_gray python tools/train.py --manifest data/manifest.csv --out models/monochrome_clf.joblib
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_rgb_gray.classify import DEFAULT_MODEL_PATH
from ir_rgb_gray.model import suggest_color_threshold, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="also train on mild/heavy degraded (low-res, noisy, jpeg-compressed) copies of each image",
    )
    args = parser.parse_args()

    print("-- channel-diff threshold check (rgb vs everything else) --")
    suggest_color_threshold(args.manifest)
    print()
    print("-- training stage-B classifier (grayscale / ir_whitehot / ir_blackhot) --")
    train(args.manifest, args.out, augment_degraded=args.augment)


if __name__ == "__main__":
    main()
