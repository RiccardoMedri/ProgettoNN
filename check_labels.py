#!/usr/bin/env python3
"""Visualize YOLO-OBB labels by drawing polygons on images.

This utility reads images and their corresponding label files and
saves copies with oriented bounding boxes rendered. It is meant as a
sanity check to ensure that label polygons align with the objects in
the dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_dataset(data_yaml: Path, split: str) -> Tuple[Path, Path, List[str]]:
    """Return image directory, label directory and class names for a split."""
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["path"]) if "path" in cfg else Path(".")
    img_dir = base / cfg[split]
    lbl_dir = base / cfg[split].replace("images", "labels")
    class_names = cfg.get("names", [])
    return img_dir, lbl_dir, class_names


def parse_label_file(path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            nums = list(map(float, parts[1:]))
            pts = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
            items.append((cls, pts))
    return items


def draw_polygons(img_path: Path, lbl_path: Path, out_path: Path, names: List[str]) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for cls, kpts in parse_label_file(lbl_path):
        pts_px = [(x * w, y * h) for (x, y) in kpts]
        draw.polygon(pts_px, outline="red", width=2)
        if 0 <= cls < len(names):
            cx = sum(x for x, _ in pts_px) / 4.0
            cy = sum(y for _, y in pts_px) / 4.0
            draw.text((cx, cy), names[cls], fill="yellow")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw OBB polygons on images")
    ap.add_argument("--data", default="config/dataset.yaml", type=str,
                    help="Path to dataset YAML")
    ap.add_argument("--split", default="train", type=str,
                    help="Dataset split to visualise (train/val/test)")
    ap.add_argument("--out", default="debug/vis", type=str,
                    help="Output directory for visualised images")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    img_dir, lbl_dir, names = load_dataset(data_yaml, args.split)
    out_dir = Path(args.out) / args.split

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        out_path = out_dir / img_path.name
        draw_polygons(img_path, lbl_path, out_path, names)

    print(f"[OK] Wrote visualised images to: {out_dir}")


if __name__ == "__main__":
    main()