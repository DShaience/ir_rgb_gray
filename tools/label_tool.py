"""Quick hand-labeling tool.

Usage:
    conda run -n ir_rgb_gray python tools/label_tool.py path/to/unlabeled_dir

Shows each image and waits for a keypress to record a label:
    r = rgb          g = grayscale
    w = ir_whitehot  b = ir_blackhot
    s = skip         q = quit (already-recorded labels are kept, safe to resume)

Labels are appended to data/manifest.csv as (path,label) rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import tkinter as tk

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data", "manifest.csv")

KEY_LABELS = {
    "r": "rgb",
    "g": "grayscale",
    "w": "ir_whitehot",
    "b": "ir_blackhot",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DISPLAY_MAX = 640


def already_labeled() -> set[str]:
    if not os.path.exists(MANIFEST_PATH):
        return set()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return {os.path.abspath(row["path"]) for row in csv.DictReader(f)}


def append_label(path: str, label: str) -> None:
    is_new = not os.path.exists(MANIFEST_PATH)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["path", "label"])
        writer.writerow([os.path.abspath(path), label])


class LabelApp:
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.index = 0

        self.root = tk.Tk()
        self.root.title("label tool -- r/g/w/b=label  s=skip  q=quit")
        self.status = tk.Label(self.root, text="", font=("Consolas", 11))
        self.status.pack()
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        for key in list(KEY_LABELS) + ["s", "q"]:
            self.root.bind(key, self.on_key)

        self.show_current()

    def show_current(self) -> None:
        if self.index >= len(self.paths):
            self.status.config(text="done -- no more images, close the window")
            self.image_label.config(image="")
            return

        path = self.paths[self.index]
        img = Image.open(path)
        img.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
        self._photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._photo)
        self.status.config(
            text=f"[{self.index + 1}/{len(self.paths)}] {os.path.basename(path)}  "
            "(r=rgb g=grayscale w=whitehot b=blackhot s=skip q=quit)"
        )

    def on_key(self, event: "tk.Event") -> None:
        if event.keysym == "q":
            self.root.destroy()
            return
        if event.keysym != "s":
            label = KEY_LABELS.get(event.keysym)
            if label:
                append_label(self.paths[self.index], label)
        self.index += 1
        self.show_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="folder of unlabeled images to walk")
    args = parser.parse_args()

    done = already_labeled()
    candidates = sorted(
        os.path.join(args.directory, name)
        for name in os.listdir(args.directory)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    todo = [p for p in candidates if os.path.abspath(p) not in done]

    print(f"{len(todo)} images to label ({len(candidates) - len(todo)} already labeled)")
    if not todo:
        return

    LabelApp(todo).run()


if __name__ == "__main__":
    main()
