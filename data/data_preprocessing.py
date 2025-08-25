#!/usr/bin/env python3
"""
Preprocessing for YOLO-OBB with augmentation + class-balanced oversampling.

This script:
- Reads raw images/labels from files/raw/{images,labels}/{train,val}
- Writes processed images/labels to files/processed/{images,labels}/{train,val}
- Keeps VAL untouched except for optional letterbox resize to a fixed size
- For TRAIN:
    * Copies the original image/labels (clean, no-aug) after letterbox resize
    * Generates extra augmented variants
    * Oversamples minority classes by creating more variants from images that contain them
- Maintains YOLO-OBB label format: "cls x1 y1 x2 y2 x3 y3 x4 y4" (normalized in [0,1])
- Applies transforms that keep labels in sync (flips, rotation, affine, translate, scale, shear)
- Canonicalizes polygon point order and deduplicates rows
- Drops degenerate polygons (area ~ 0) and out-of-range rows

Defaults are safe if keys are missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageOps

# ----------------------------
# IO helpers
# ----------------------------

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


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        dst.write_bytes(src.read_bytes())


# ----------------------------
# Geometry / labels
# ----------------------------

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
    """Shoelace area for 4-point polygon (expects 4 points)."""
    if len(kpts) != 4:
        return 0.0
    xys = kpts + [kpts[0]]
    s = 0.0
    for i in range(4):
        s += xys[i][0] * xys[i + 1][1] - xys[i + 1][0] * xys[i][1]
    return abs(0.5 * s)


def canonicalize_clockwise(kpts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # centroid
    cx = sum(x for x, _ in kpts) / 4.0
    cy = sum(y for _, y in kpts) / 4.0
    pts = []
    for (x, y) in kpts:
        ang = math.atan2(y - cy, x - cx)
        pts.append((x, y, ang))
    pts.sort(key=lambda t: t[2])  # CCW
    pts = [(x, y) for (x, y, _) in pts]
    # start from top-left-ish
    start = min(range(4), key=lambda i: (pts[i][1], pts[i][0]))
    return pts[start:] + pts[:start]


def dedupe_rows(rows: List[Tuple[int, List[Tuple[float, float]]]]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    seen: Set[Tuple] = set()
    out: List[Tuple[int, List[Tuple[float, float]]]] = []
    for cls, kpts in rows:
        key = (cls,) + tuple(round(v, 6) for xy in kpts for v in xy)
        if key in seen:
            continue
        seen.add(key)
        out.append((cls, kpts))
    return out


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# ----------------------------
# Letterbox resize (like YOLO)
# ----------------------------

def letterbox(image: Image.Image, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)) -> Tuple[Image.Image, float, Tuple[float, float]]:
    """Resize & pad to target, keeping aspect ratio. Returns (img, scale, (dx, dy))."""
    w0, h0 = image.size
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    img = image.resize((nw, nh), Image.BILINEAR)
    pad_w, pad_h = new_w - nw, new_h - nh
    # divide padding equally left/right & top/bottom
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    if left or right or top or bottom:
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=color)
    return img, r, (left, top)


def apply_letterbox_to_points(kpts_norm: List[Tuple[float, float]], orig_size: Tuple[int, int], scale: float, pad: Tuple[float, float], new_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Map normalized points from original image to normalized points in letterboxed image."""
    w0, h0 = orig_size
    W, H = new_size
    dx, dy = pad
    out = []
    for (xn, yn) in kpts_norm:
        x_px = xn * w0
        y_px = yn * h0
        x2 = x_px * scale + dx
        y2 = y_px * scale + dy
        out.append((x2 / W, y2 / H))
    return out


# ----------------------------
# Affine augmentation
# ----------------------------

def build_affine_matrix(W: int, H: int,
                        hflip: bool = False,
                        vflip: bool = False,
                        angle_deg: float = 0.0,
                        scale: float = 1.0,
                        shear_x_deg: float = 0.0,
                        shear_y_deg: float = 0.0,
                        translate_frac: Tuple[float, float] = (0.0, 0.0)) -> List[List[float]]:
    """Return 3x3 affine matrix mapping INPUT->OUTPUT coords (pixels)."""
    cx, cy = W / 2.0, H / 2.0

    def T(tx, ty):
        return [[1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]]

    def S(sx, sy):
        return [[sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]]

    def SH(shx, shy):
        return [[1, math.tan(math.radians(shx)), 0],
                [math.tan(math.radians(shy)), 1, 0],
                [0, 0, 1]]

    def R(a_deg):
        a = math.radians(a_deg)
        c, s = math.cos(a), math.sin(a)
        return [[c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]]

    def Mmul(A, B):
        # 3x3 * 3x3
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    M = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]

    # center to origin
    M = Mmul(T(-cx, -cy), M)

    # flips via negative scale
    sx = -1.0 if hflip else 1.0
    sy = -1.0 if vflip else 1.0
    M = Mmul(S(sx, sy), M)

    # scale
    M = Mmul(S(scale, scale), M)

    # shear
    M = Mmul(SH(shear_x_deg, shear_y_deg), M)

    # rotation
    M = Mmul(R(angle_deg), M)

    # back to center
    M = Mmul(T(cx, cy), M)

    # translation (fraction of size -> pixels)
    tx = translate_frac[0] * W
    ty = translate_frac[1] * H
    M = Mmul(T(tx, ty), M)

    return M


def invert_affine(M: List[List[float]]) -> List[List[float]]:
    """Inverse of 3x3 affine matrix with last row [0,0,1]."""
    a, b, c = M[0]
    d, e, f = M[1]
    # 2x2 inverse
    det = a * e - b * d
    if abs(det) < 1e-12:
        # fallback to identity
        return [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
    inv_a =  e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e =  a / det
    inv_c = -(inv_a * c + inv_b * f)
    inv_f = -(inv_d * c + inv_e * f)
    return [[inv_a, inv_b, inv_c],
            [inv_d, inv_e, inv_f],
            [0,     0,     1]]


def apply_affine_to_points(kpts: List[Tuple[float, float]], M: List[List[float]]) -> List[Tuple[float, float]]:
    out = []
    for (x, y) in kpts:
        xn = M[0][0] * x + M[0][1] * y + M[0][2]
        yn = M[1][0] * x + M[1][1] * y + M[1][2]
        out.append((xn, yn))
    return out


def pil_affine(image: Image.Image, M_in_to_out: List[List[float]], fill=(114, 114, 114)) -> Image.Image:
    """Apply affine using PIL. PIL needs the inverse (output->input)."""
    inv = invert_affine(M_in_to_out)
    a, b, c = inv[0]
    d, e, f = inv[1]
    return image.transform(image.size, Image.AFFINE, data=(a, b, c, d, e, f),
                           resample=Image.BILINEAR, fillcolor=fill)


# ----------------------------
# Balancing logic
# ----------------------------

def count_class_instances(label_lines: List[str]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for ln in label_lines:
        pr = parse_yolo_obb_line(ln)
        if pr is None:
            continue
        cls, _ = pr
        counts[cls] = counts.get(cls, 0) + 1
    return counts


# ----------------------------
# Core pipeline
# ----------------------------

def process_split(
    split: str,
    img_dir_in: Path,
    lbl_dir_in: Path,
    img_dir_out: Path,
    lbl_dir_out: Path,
    image_size: Tuple[int, int],
    aug_cfg: Dict,
    base_copies: int,
    max_extra_per_image: int,
    oversample: bool,
    rng: random.Random,
) -> None:
    """
    For VAL: copy letterboxed originals only.
    For TRAIN: copy originals + N base augmented copies + extra copies for class balancing.
    """
    img_paths = [p for p in img_dir_in.rglob("*") if p.suffix.lower() in IMG_EXTS]
    img_paths.sort()

    # Load class counts (train only)
    per_image_classes: Dict[Path, Set[int]] = {}
    class_counts: Dict[int, int] = {}
    if split == "train":
        for img_path in img_paths:
            rel = img_path.relative_to(img_dir_in).with_suffix(".txt")
            lbl_in = lbl_dir_in / rel
            lines = read_label_file(lbl_in)
            per_image_classes[img_path] = set()
            for ln in lines:
                pr = parse_yolo_obb_line(ln)
                if pr is None:
                    continue
                cls, _ = pr
                per_image_classes[img_path].add(cls)
            cc = count_class_instances(lines)
            for k, v in cc.items():
                class_counts[k] = class_counts.get(k, 0) + v

    # target for balancing
    target_per_class = {}
    if split == "train" and oversample and class_counts:
        maxc = max(class_counts.values())
        for cls, cnt in class_counts.items():
            target_per_class[cls] = maxc  # balance up to majority

    # Aug params/defaults
    hflip_p = float(aug_cfg.get("hflip_prob", 0.5) or 0.0)
    vflip_p = float(aug_cfg.get("vflip_prob", 0.0) or 0.0)
    rotation = float(aug_cfg.get("rotation", 0.0) or 0.0)
    translate = float(aug_cfg.get("translate", 0.0) or 0.0)
    scale_rng = tuple(aug_cfg.get("scale_range", [1.0, 1.0])) if aug_cfg.get("scale_range") else (1.0, 1.0)
    shear = float(aug_cfg.get("shear", 0.0) or 0.0)

    # Pass 1: base copies (+0 for val)
    for img_path in img_paths:
        rel = img_path.relative_to(img_dir_in)
        out_img_base = img_dir_out / rel
        out_lbl_base = lbl_dir_out / rel.with_suffix(".txt")
        out_img_base.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_base.parent.mkdir(parents=True, exist_ok=True)

        # load
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        # labels
        lbl_in = lbl_dir_in / rel.with_suffix(".txt")
        lines = read_label_file(lbl_in)

        # letterbox to target
        img_lb, r, (dx, dy) = letterbox(img, new_shape=image_size)
        W, H = image_size

        # map each normalized quad to letterboxed normalized
        items = []
        for ln in lines:
            pr = parse_yolo_obb_line(ln)
            if pr is None:
                continue
            cls, kpts_norm = pr
            kpts_lb = apply_letterbox_to_points(kpts_norm, (w0, h0), r, (dx, dy), (W, H))
            if poly_area_xy(kpts_lb) < 1e-8:
                continue
            kpts_lb = [(clamp01(x), clamp01(y)) for (x, y) in canonicalize_clockwise(kpts_lb)]
            items.append((cls, kpts_lb))

        items = dedupe_rows(items)
        img_lb.save(out_img_base, quality=95)
        write_label_file(out_lbl_base, items)

        if split != "train":
            continue

        # base augmented copies
        for j in range(base_copies):
            img_aug, items_aug = apply_random_affine(img_lb, items, (W, H),
                                                     rng, hflip_p, vflip_p, rotation, translate, scale_rng, shear)
            suffix = f"_aug{j+1:02d}"
            out_img = img_dir_out / rel.with_stem(rel.stem + suffix)
            out_lbl = lbl_dir_out / rel.with_stem(rel.stem + suffix).with_suffix(".txt")
            img_aug.save(out_img, quality=95)
            write_label_file(out_lbl, items_aug)

    # Pass 2: oversampling by minority classes (train only)
    if split == "train" and oversample and target_per_class:
        # compute current counts after base phase
        cur_counts = {k: 0 for k in target_per_class}
        # scan produced labels to compute counts
        for lblp in lbl_dir_out.rglob("*.txt"):
            for ln in read_label_file(lblp):
                pr = parse_yolo_obb_line(ln)
                if pr is None:
                    continue
                cls, _ = pr
                if cls in cur_counts:
                    cur_counts[cls] += 1

        deficits = {k: max(0, target_per_class[k] - cur_counts.get(k, 0)) for k in target_per_class}
        if any(v > 0 for v in deficits.values()):
            # index images by class
            by_class: Dict[int, List[Path]] = {k: [] for k in deficits}
            for img_path in img_paths:
                present = per_image_classes.get(img_path, set())
                for cls in present:
                    if cls in by_class:
                        by_class[cls].append(img_path)

            # cap extra per image
            produced_extra = {p: 0 for p in img_paths}

            # generate until deficits gone or no more capacity
            while True:
                # pick a class with remaining deficit
                choices = [c for c, d in deficits.items() if d > 0 and by_class.get(c)]
                if not choices:
                    break
                cls = rng.choice(choices)
                # pick an image containing that class with room to add
                candidates = [p for p in by_class[cls] if produced_extra[p] < max_extra_per_image]
                if not candidates:
                    # no capacity for this class
                    deficits[cls] = 0
                    continue
                img_path = rng.choice(candidates)

                # load base letterboxed img and items we wrote earlier
                rel = img_path.relative_to(img_dir_in)
                base_img_path = img_dir_out / rel
                base_lbl_path = lbl_dir_out / rel.with_suffix(".txt")
                if not base_img_path.exists() or not base_lbl_path.exists():
                    # fallback: skip
                    deficits[cls] = 0
                    continue

                img_lb = Image.open(base_img_path).convert("RGB")
                W, H = img_lb.size
                items = []
                for ln in read_label_file(base_lbl_path):
                    pr = parse_yolo_obb_line(ln)
                    if pr is None:
                        continue
                    items.append(pr)

                # apply augmentation
                img_aug, items_aug = apply_random_affine(img_lb, items, (W, H),
                                                         rng, hflip_p, vflip_p, rotation, translate, scale_rng, shear)

                # save with running counter per image
                produced_extra[img_path] += 1
                idx = produced_extra[img_path]
                suffix = f"_bal{idx:02d}"
                out_img = img_dir_out / rel.with_stem(rel.stem + suffix)
                out_lbl = lbl_dir_out / rel.with_stem(rel.stem + suffix).with_suffix(".txt")
                img_aug.save(out_img, quality=95)
                write_label_file(out_lbl, items_aug)

                # update counts/deficits by scanning only new labels
                for cls2, _ in items_aug:
                    if cls2 in deficits:
                        deficits[cls2] = max(0, deficits[cls2] - 1)

                if not any(v > 0 for v in deficits.values()):
                    break


def apply_random_affine(
    img_in: Image.Image,
    items_in: List[Tuple[int, List[Tuple[float, float]]]],
    size: Tuple[int, int],
    rng: random.Random,
    hflip_p: float, vflip_p: float, rotation: float, translate: float, scale_rng: Tuple[float, float], shear: float
) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    W, H = size
    # sample params
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

    # points are currently normalized in [0,1] for the letterboxed image -> convert to pixels first
    rows_out: List[Tuple[int, List[Tuple[float, float]]]] = []
    for cls, kpts_norm in items_in:
        kpts_px = [(x * W, y * H) for (x, y) in kpts_norm]
        kpts_px2 = apply_affine_to_points(kpts_px, M)
        # back to normalized
        kpts_n2 = [(clamp01(x / W), clamp01(y / H)) for (x, y) in kpts_px2]
        kpts_n2 = canonicalize_clockwise(kpts_n2)
        if poly_area_xy(kpts_n2) < 1e-6:
            continue
        rows_out.append((cls, kpts_n2))

    rows_out = dedupe_rows(rows_out)
    return img_out, rows_out


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLO-OBB preprocessing with augmentation + class balancing.")
    ap.add_argument("--config", default="config/config.json", type=str, help="Path to config.json")
    ap.add_argument("--raw-images", default=None, type=str, help="Override raw images dir")
    ap.add_argument("--raw-labels", default=None, type=str, help="Override raw labels dir")
    ap.add_argument("--proc-images", default=None, type=str, help="Override processed images dir")
    ap.add_argument("--proc-labels", default=None, type=str, help="Override processed labels dir")
    ap.add_argument("--train-split", default="train", type=str)
    ap.add_argument("--val-split", default="val", type=str)
    ap.add_argument("--base-copies", default=0, type=int, help="Augmented copies per train image (in addition to the original)")
    ap.add_argument("--max-extra-per-image", default=5, type=int, help="Cap for balancing variants per single image")
    ap.add_argument("--no-oversample", action="store_true", help="Disable class-balanced oversampling")
    ap.add_argument("--seed", default=42, type=int)

    args = ap.parse_args()
    cfg = load_config(args.config)

    # paths
    paths = cfg.get("paths", {})
    raw_images = Path(args.raw_images or paths.get("raw_images", "files/raw/images"))
    raw_labels = Path(args.raw_labels or paths.get("raw_labels", "files/raw/labels"))
    proc_images = Path(args.proc_images or paths.get("proc_images", "files/processed/images"))
    proc_labels = Path(args.proc_labels or paths.get("proc_labels", "files/processed/labels"))

    # sizes
    ds = cfg.get("dataset", {})
    W, H = tuple(ds.get("image_size", [640, 640]))

    # aug
    aug_cfg = cfg.get("augment", {})

    # RNG
    rng = random.Random(args.seed)

    # ensure split dirs
    for split in (args.train_split, args.val_split):
        (proc_images / split).mkdir(parents=True, exist_ok=True)
        (proc_labels / split).mkdir(parents=True, exist_ok=True)

    # VAL first (no aug, only letterbox)
    process_split(
        split=args.val_split,
        img_dir_in=raw_images / args.val_split,
        lbl_dir_in=raw_labels / args.val_split,
        img_dir_out=proc_images / args.val_split,
        lbl_dir_out=proc_labels / args.val_split,
        image_size=(W, H),
        aug_cfg=aug_cfg,
        base_copies=0,
        max_extra_per_image=0,
        oversample=False,
        rng=rng,
    )

    # TRAIN with aug + balancing
    process_split(
        split=args.train_split,
        img_dir_in=raw_images / args.train_split,
        lbl_dir_in=raw_labels / args.train_split,
        img_dir_out=proc_images / args.train_split,
        lbl_dir_out=proc_labels / args.train_split,
        image_size=(W, H),
        aug_cfg=aug_cfg,
        base_copies=int(args.base_copies),
        max_extra_per_image=int(args.max_extra_per_image),
        oversample=not args.no_oversample,
        rng=rng,
    )

    print("Done. Processed data at:", proc_images.parent)


if __name__ == "__main__":
    main()
