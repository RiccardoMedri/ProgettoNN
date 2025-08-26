#!/usr/bin/env python3
"""
data_preprocessing.py
=====================

Preprocess + augment data for OBB detection.

What it does
------------
- Always writes a letterboxed copy of each image/label into processed/{split}.
- On the *train* split, generates extra augmented copies per image:
  * OBB-safe color jitter (brightness/contrast/saturation)
  * OBB-safe random crop (polygons re-mapped, outside ones dropped)
  * Optional random affine (flip/rotate/scale/translate/shear)
- All output labels are in YOLO-OBB (quad points) with normalized coords.
- 'base_copies' is read from the config (augment.augment2.base_copies) with
  fallback to augment.max_extra_per_image for backward compat.

Notes
-----
- Uses Image.Resampling.BILINEAR to avoid Pillow deprecation warnings.
- Cropping happens BEFORE letterbox; affine happens AFTER letterbox.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageOps

try:
    # Pillow >= 10
    from PIL.Image import Resampling as _Resampling
    _BILINEAR = _Resampling.BILINEAR
except Exception:
    # Pillow < 10 fallback
    _BILINEAR = Image.BILINEAR

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_label_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_label_file(path: Path, rows: List[Tuple[int, List[Tuple[float, float]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, kpts in rows:
            f.write(str(cls) + " " + " ".join(f"{x:.6f} {y:.6f}" for (x, y) in kpts) + "\n")


# -----------------------------------------------------------------------------
# Geometry / labels helpers
# -----------------------------------------------------------------------------

def parse_yolo_obb_line(line: str) -> Optional[Tuple[int, List[Tuple[float, float]]]]:
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        cls = int(float(parts[0]))
        nums = list(map(float, parts[1:]))
    except Exception:
        return None
    kpts = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
    return cls, kpts


def poly_area_xy(kpts: List[Tuple[float, float]]) -> float:
    if len(kpts) != 4:
        return 0.0
    xys = kpts + [kpts[0]]
    s = 0.0
    for i in range(4):
        s += xys[i][0] * xys[i + 1][1] - xys[i + 1][0] * xys[i][1]
    return abs(0.5 * s)


def canonicalize_clockwise(kpts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cx = sum(x for x, _ in kpts) / 4.0
    cy = sum(y for _, y in kpts) / 4.0
    pts = []
    for (x, y) in kpts:
        ang = math.atan2(y - cy, x - cx)
        pts.append((x, y, ang))
    pts.sort(key=lambda t: t[2])  # CCW
    pts = [(x, y) for (x, y, _) in pts]
    start = min(range(4), key=lambda i: (pts[i][1], pts[i][0]))  # top-left-ish
    return pts[start:] + pts[:start]


def dedupe_rows(rows: List[Tuple[int, List[Tuple[float, float]]]]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    seen = set()
    out = []
    for cls, kpts in rows:
        key = (cls,) + tuple(round(v, 6) for xy in kpts for v in xy)
        if key in seen:
            continue
        seen.add(key)
        out.append((cls, kpts))
    return out


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# -----------------------------------------------------------------------------
# Letterbox (YOLO-style)
# -----------------------------------------------------------------------------

def letterbox(image: Image.Image, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[Image.Image, float, Tuple[float, float]]:
    w0, h0 = image.size
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    img = image.resize((nw, nh), _BILINEAR)
    pad_w, pad_h = new_w - nw, new_h - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=color)
    return img, r, (left, top)


# -----------------------------------------------------------------------------
# Affine (flip/rotate/scale/translate/shear)
# -----------------------------------------------------------------------------

def _mat_mul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def build_affine_matrix(W: int, H: int, hflip: bool, vflip: bool, angle: float, scale: float,
                        shear_x: float, shear_y: float, translate: Tuple[float, float]) -> List[List[float]]:
    a = math.radians(angle)
    sx, sy = math.radians(shear_x), math.radians(shear_y)
    cx, cy = W / 2.0, H / 2.0
    M_flip = [[-1 if hflip else 1, 0, 0], [0, -1 if vflip else 1, 0], [0, 0, 1]]
    cos_a, sin_a = math.cos(a), math.sin(a)
    M_rot = [[scale * cos_a, -scale * sin_a, 0], [scale * sin_a, scale * cos_a, 0], [0, 0, 1]]
    M_shear = [[1, math.tan(sx), 0], [math.tan(sy), 1, 0], [0, 0, 1]]
    M = _mat_mul(_mat_mul(M_rot, M_shear), M_flip)
    tx, ty = translate
    M_trans = [[1, 0, tx * W], [0, 1, ty * H], [0, 0, 1]]
    M = _mat_mul(M_trans, M)
    M_origin = [[1, 0, cx], [0, 1, cy], [0, 0, 1]]
    M = _mat_mul(M_origin, M)
    M_pre = [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]
    M = _mat_mul(M, M_pre)
    return M


def pil_affine(img: Image.Image, M: List[List[float]], fill=(114, 114, 114)) -> Image.Image:
    a, b, c = M[0]
    d, e, f = M[1]
    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample=_BILINEAR, fillcolor=fill)


def apply_affine_to_points(points: List[Tuple[float, float]], M: List[List[float]]) -> List[Tuple[float, float]]:
    out = []
    for x, y in points:
        x2 = M[0][0] * x + M[0][1] * y + M[0][2]
        y2 = M[1][0] * x + M[1][1] * y + M[1][2]
        out.append((x2, y2))
    return out


def apply_random_affine(img_in: Image.Image,
                        items_in: List[Tuple[int, List[Tuple[float, float]]]],
                        size: Tuple[int, int],
                        rng: random.Random,
                        hflip_p: float, vflip_p: float,
                        rotation: float, translate: float,
                        scale_rng: Tuple[float, float], shear: float) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    W, H = size
    hflip = rng.random() < hflip_p
    vflip = rng.random() < vflip_p
    angle = rng.uniform(-rotation, rotation) if rotation > 0 else 0.0
    scale = rng.uniform(scale_rng[0], scale_rng[1]) if scale_rng else 1.0
    tx = rng.uniform(-translate, translate) if translate > 0 else 0.0
    ty = rng.uniform(-translate, translate) if translate > 0 else 0.0
    shear_x = rng.uniform(-shear, shear) if shear > 0 else 0.0
    shear_y = rng.uniform(-shear, shear) if shear > 0 else 0.0

    M = build_affine_matrix(W, H, hflip, vflip, angle, scale, shear_x, shear_y, (tx, ty))
    img_out = pil_affine(img_in, M, fill=(114, 114, 114))

    rows_out: List[Tuple[int, List[Tuple[float, float]]]] = []
    for cls, kpts_norm in items_in:
        kpts_px = [(x * W, y * H) for (x, y) in kpts_norm]
        kpts_px2 = apply_affine_to_points(kpts_px, M)
        kpts_n2 = [(clamp01(x / W), clamp01(y / H)) for (x, y) in kpts_px2]
        kpts_n2 = canonicalize_clockwise(kpts_n2)
        if poly_area_xy(kpts_n2) < 1e-6:
            continue
        rows_out.append((cls, kpts_n2))
    rows_out = dedupe_rows(rows_out)
    return img_out, rows_out


# -----------------------------------------------------------------------------
# Color jitter (non-geometric)
# -----------------------------------------------------------------------------

def apply_color_jitter(img: Image.Image, rng: random.Random,
                       brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0) -> Image.Image:
    out = img
    if brightness > 0:
        out = ImageEnhance.Brightness(out).enhance(1.0 + rng.uniform(-brightness, brightness))
    if contrast > 0:
        out = ImageEnhance.Contrast(out).enhance(1.0 + rng.uniform(-contrast, contrast))
    if saturation > 0:
        out = ImageEnhance.Color(out).enhance(1.0 + rng.uniform(-saturation, saturation))
    return out


# -----------------------------------------------------------------------------
# OBB-safe random crop
# -----------------------------------------------------------------------------

def apply_random_crop(img: Image.Image,
                      items: List[Tuple[int, List[Tuple[float, float]]]],
                      rng: random.Random,
                      crop_prob: float,
                      crop_scale: Tuple[float, float]) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    if crop_prob <= 0 or rng.random() > crop_prob or not items:
        return img, items
    w, h = img.size
    scale = rng.uniform(crop_scale[0], crop_scale[1])
    crop_w = max(1, min(int(w * scale), w))
    crop_h = max(1, min(int(h * scale), h))
    left = 0 if w == crop_w else rng.randint(0, w - crop_w)
    top = 0 if h == crop_h else rng.randint(0, h - crop_h)
    right = left + crop_w
    bottom = top + crop_h
    img_crop = img.crop((left, top, right, bottom))

    items_out: List[Tuple[int, List[Tuple[float, float]]]] = []
    for cls, kpts_norm in items:
        kpts_px = [(x * w, y * h) for (x, y) in kpts_norm]
        pts_crop = [(x - left, y - top) for (x, y) in kpts_px]
        inside = any(0 <= x <= crop_w and 0 <= y <= crop_h for (x, y) in pts_crop)
        if not inside:
            continue
        kpts_n = [(clamp01(x / crop_w), clamp01(y / crop_h)) for (x, y) in pts_crop]
        kpts_n = canonicalize_clockwise(kpts_n)
        if poly_area_xy(kpts_n) < 1e-6:
            continue
        items_out.append((cls, kpts_n))
    return img_crop, items_out


# -----------------------------------------------------------------------------
# Processing per split
# -----------------------------------------------------------------------------

def process_split(split: str,
                  img_dir_in: Path,
                  lbl_dir_in: Path,
                  img_dir_out: Path,
                  lbl_dir_out: Path,
                  image_size: Tuple[int, int],
                  aug_cfg: Dict,
                  base_copies: int,
                  rng: random.Random) -> None:
    is_train = split.lower() == "train"

    # Augment params
    hflip_p   = float(aug_cfg.get("hflip_prob", 0.0))
    vflip_p   = float(aug_cfg.get("vflip_prob", 0.0))
    rotation  = float(aug_cfg.get("rotation", 0.0))
    translate = float(aug_cfg.get("translate", 0.0))
    scale_rng = tuple(aug_cfg.get("scale_range", [1.0, 1.0]))
    shear     = float(aug_cfg.get("shear", 0.0))

    aug2   = aug_cfg.get("augment2", {}) if isinstance(aug_cfg.get("augment2", {}), dict) else {}
    cj     = aug2.get("color_jitter", {}) if isinstance(aug2.get("color_jitter", {}), dict) else {}
    bright = float(cj.get("brightness", 0.0))
    contr  = float(cj.get("contrast", 0.0))
    sat    = float(cj.get("saturation", 0.0))
    crop_p = float(aug2.get("crop_prob", 0.0))
    crop_s = tuple(aug2.get("crop_scale", [1.0, 1.0]))

    img_dir_out.mkdir(parents=True, exist_ok=True)
    lbl_dir_out.mkdir(parents=True, exist_ok=True)

    for fname in sorted(os.listdir(img_dir_in)):
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        img_path = img_dir_in / fname
        lbl_path = lbl_dir_in / f"{stem}.txt"

        img = Image.open(img_path).convert("RGB")

        # parse labels (normalized to original image)
        items: List[Tuple[int, List[Tuple[float, float]]]] = []
        for ln in read_label_file(lbl_path):
            p = parse_yolo_obb_line(ln)
            if p:
                items.append(p)

        # 1) Save original letterboxed version for BOTH splits
        img_lb, r, (dx, dy) = letterbox(img, new_shape=image_size)
        rows: List[Tuple[int, List[Tuple[float, float]]]] = []
        for cls, kpts in items:
            kpts_px = [(x * img.width, y * img.height) for (x, y) in kpts]
            kpts_lb = [((x * r + dx) / image_size[0], (y * r + dy) / image_size[1]) for (x, y) in kpts_px]
            kpts_lb = canonicalize_clockwise(kpts_lb)
            if poly_area_xy(kpts_lb) < 1e-6:
                continue
            rows.append((cls, kpts_lb))
        img_lb.save(img_dir_out / f"{stem}.jpg")
        write_label_file(lbl_dir_out / f"{stem}.txt", dedupe_rows(rows))

        # 2) Augmented copies only for train
        if is_train and base_copies > 0 and rows:
            for idx in range(base_copies):
                aug_img = apply_color_jitter(img, rng, bright, contr, sat)
                aug_items = items.copy()

                # OBB-safe random crop BEFORE letterbox
                aug_img, aug_items = apply_random_crop(aug_img, aug_items, rng, crop_p, crop_s)

                # Letterbox to target size
                lb2, r2, (dx2, dy2) = letterbox(aug_img, new_shape=image_size)

                rows_aug: List[Tuple[int, List[Tuple[float, float]]]] = []
                w_i, h_i = aug_img.size
                for cls, kpts in aug_items:
                    kpts_px2 = [(x * w_i, y * h_i) for (x, y) in kpts]
                    kpts_lb2 = [((x * r2 + dx2) / image_size[0], (y * r2 + dy2) / image_size[1]) for (x, y) in kpts_px2]
                    kpts_lb2 = canonicalize_clockwise(kpts_lb2)
                    if poly_area_xy(kpts_lb2) < 1e-6:
                        continue
                    rows_aug.append((cls, kpts_lb2))

                if not rows_aug:
                    # if cropping removed everything, just skip this extra copy
                    continue

                # Random affine AFTER letterbox
                img_aff, rows_aff = apply_random_affine(
                    lb2, rows_aug, size=image_size, rng=rng,
                    hflip_p=hflip_p, vflip_p=vflip_p, rotation=rotation,
                    translate=translate, scale_rng=scale_rng, shear=shear
                )

                aug_name = f"{stem}_aug{idx}"
                img_aff.save(img_dir_out / f"{aug_name}.jpg")
                write_label_file(lbl_dir_out / f"{aug_name}.txt", dedupe_rows(rows_aff))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess + augment OBB dataset.")
    ap.add_argument("--config", default="config/config.json", type=str, help="Path to config.json")
    ap.add_argument("--raw-images", default=None, type=str, help="Override raw images dir")
    ap.add_argument("--raw-labels", default=None, type=str, help="Override raw labels dir")
    ap.add_argument("--proc-images", default=None, type=str, help="Override processed images dir")
    ap.add_argument("--proc-labels", default=None, type=str, help="Override processed labels dir")
    ap.add_argument("--train-split", default="train", type=str)
    ap.add_argument("--val-split", default="val", type=str)
    ap.add_argument("--seed", default=None, type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Seed: CLI overrides config
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    rng = random.Random(seed)

    # Paths
    paths = cfg.get("paths", {})
    raw_images = Path(args.raw_images or paths.get("raw_images", "files/raw/images"))
    raw_labels = Path(args.raw_labels or paths.get("raw_labels", "files/raw/labels"))
    proc_images = Path(args.proc_images or paths.get("proc_images", "files/processed/images"))
    proc_labels = Path(args.proc_labels or paths.get("proc_labels", "files/processed/labels"))

    # Dataset size
    ds_cfg = cfg.get("dataset", {})
    image_size = tuple(ds_cfg.get("image_size", [640, 640]))

    # Augmentation config (keep your current structure; allow nested augment2)
    aug_cfg = cfg.get("augment", {}).copy()
    if "augment2" in cfg and isinstance(cfg["augment2"], dict):
        # (optional: if user stores augment2 at root, merge it in)
        aug_cfg["augment2"] = cfg["augment2"]

    # Pull augmented copies per image from config (no CLI)
    # Priority: augment.augment2.base_copies -> augment.base_copies -> augment.max_extra_per_image -> 0
    base_copies = int(
        (aug_cfg.get("augment2", {}) or {}).get("base_copies",
            aug_cfg.get("base_copies",
                aug_cfg.get("max_extra_per_image", 0)))
    )
    base_copies = max(0, base_copies)

    # Ensure output dirs exist
    for split in (args.train_split, args.val_split):
        (proc_images / split).mkdir(parents=True, exist_ok=True)
        (proc_labels / split).mkdir(parents=True, exist_ok=True)

    # Process splits
    process_split(
        split=args.val_split,
        img_dir_in=raw_images / args.val_split,
        lbl_dir_in=raw_labels / args.val_split,
        img_dir_out=proc_images / args.val_split,
        lbl_dir_out=proc_labels / args.val_split,
        image_size=image_size,
        aug_cfg=aug_cfg,
        base_copies=0,                 # no augmentation on val
        rng=rng,
    )

    process_split(
        split=args.train_split,
        img_dir_in=raw_images / args.train_split,
        lbl_dir_in=raw_labels / args.train_split,
        img_dir_out=proc_images / args.train_split,
        lbl_dir_out=proc_labels / args.train_split,
        image_size=image_size,
        aug_cfg=aug_cfg,
        base_copies=base_copies,       # from config
        rng=rng,
    )

    print(f"[OK] Wrote processed data to: {proc_images.parent}  (seed={seed}, base_copies={base_copies})")


if __name__ == "__main__":
    main()
