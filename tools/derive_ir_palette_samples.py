"""Converts real photos in data/samples/{ir,rgb} into pipeline-ready sample
classes. Re-run any time more source photos are dropped into those folders.

- IR: the Wikimedia thermal photos are FLIR's default "Ironbow" false-color
  palette, not the grayscale blackhot/whitehot palette this project targets.
  Ironbow's luminance increases roughly monotonically with temperature, so
  converting to grayscale luminance is a reasonable stand-in for whitehot;
  inverting that gives a blackhot stand-in. Real thermal texture/content is
  preserved even though the exact palette mapping is approximate.
- Grayscale: real RGB photos converted to single-channel grayscale.
"""
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IR_SRC = os.path.join(ROOT, "data", "samples", "ir")
RGB_SRC = os.path.join(ROOT, "data", "samples", "rgb")
WHITEHOT_OUT = os.path.join(ROOT, "data", "samples", "ir_whitehot")
BLACKHOT_OUT = os.path.join(ROOT, "data", "samples", "ir_blackhot")
GRAYSCALE_OUT = os.path.join(ROOT, "data", "samples", "grayscale")

for d in (WHITEHOT_OUT, BLACKHOT_OUT, GRAYSCALE_OUT):
    os.makedirs(d, exist_ok=True)

ir_files = sorted(os.listdir(IR_SRC))
for i, name in enumerate(ir_files):
    img = cv2.imread(os.path.join(IR_SRC, name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem = os.path.splitext(name)[0]
    if i % 2 == 0:
        cv2.imwrite(os.path.join(WHITEHOT_OUT, f"{stem}_whitehot.png"), gray)
    else:
        cv2.imwrite(os.path.join(BLACKHOT_OUT, f"{stem}_blackhot.png"), 255 - gray)

rgb_files = sorted(os.listdir(RGB_SRC))
for name in rgb_files:
    img = cv2.imread(os.path.join(RGB_SRC, name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem = os.path.splitext(name)[0]
    cv2.imwrite(os.path.join(GRAYSCALE_OUT, f"{stem}_gray.png"), gray)

print(f"whitehot: {len(os.listdir(WHITEHOT_OUT))}")
print(f"blackhot: {len(os.listdir(BLACKHOT_OUT))}")
print(f"grayscale: {len(os.listdir(GRAYSCALE_OUT))}")
print(f"rgb (unchanged, kept as-is): {len(rgb_files)}")
