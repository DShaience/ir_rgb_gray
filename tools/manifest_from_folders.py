"""Build data/manifest.csv from images already sorted into class-named
folders under data/samples/{rgb,grayscale,ir_whitehot,ir_blackhot}/. Use this
when adding a new pre-organized batch (e.g. another public dataset dropped
into its matching folder); use tools/label_tool.py instead for hand-labeling
a pile of unsorted images.
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.csv")

FOLDER_TO_LABEL = {
    "rgb": "rgb",
    "grayscale": "grayscale",
    "ir_whitehot": "ir_whitehot",
    "ir_blackhot": "ir_blackhot",
}


def main() -> None:
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for folder, label in FOLDER_TO_LABEL.items():
            for path in sorted(glob.glob(os.path.join(ROOT, "data", "samples", folder, "*"))):
                writer.writerow([path, label])
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
