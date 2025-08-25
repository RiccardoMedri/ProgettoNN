#!/usr/bin/env python3
"""
data_augmentation_2.py
======================

This module performs preprocessing and augmentation for oriented bounding box (OBB) detection
tasks.  It is inspired by the existing ``data_preprocessing.py`` script in the original
repository but extends it with two additional capabilities:

* **Non‑geometric (colour‑based) augmentations** – random brightness, contrast and
  saturation jitter are applied independently to each training image.  These operations
  do not alter the geometry of the image and are therefore safe to use with OBB
  annotations.
* **OBB‑safe random cropping** – a random crop is taken from the image, and each OBB
  polygon is transformed into the cropped coordinate frame.  Polygons that fall
  completely outside the crop are dropped; those that partially overlap are kept but
  clipped to the crop boundaries.  After cropping the image is resized to the target
  shape using a letterbox transformation.

The script can be invoked from the command line in the same way as the original
``data_preprocessing.py``.  By default it reads raw images and labels from
``files/raw/{images,labels}/{split}`` and writes processed files to
``files/processed/{images,labels}/{split}``.  For the validation split only a
letterbox resize is applied.  For the training split the following pipeline is
performed for each input image:

1. Copy the original letterboxed image/labels (no augmentation).
2. Generate ``base_copies`` additional augmented variants:
   a. Apply random colour jitter using brightness/contrast/saturation factors
      drawn from the ``augment2.color_jitter`` configuration.
   b. Optionally sample a random crop region and crop both the image and
      polygons.  The crop probability and scale range are controlled by
      ``augment2.crop_prob`` and ``augment2.crop_scale``.
   c. Resize the cropped image to the configured ``dataset.image_size`` via
      letterboxing, updating polygon coordinates accordingly.
   d. Apply the existing random affine transform (flip/rotation/scale/translate/
      shear) to the letterboxed image and normalised polygons.
3. Save each processed image to disk and emit corresponding YOLO‑OBB labels.

Augmentation parameters are read from ``config.json``.  New keys under the
``augment2`` section are accepted:

.. code-block:: json

    {
      "augment2": {
        "color_jitter": {
          "brightness": 0.2,
          "contrast": 0.2,
          "saturation": 0.2
        },
        "crop_prob": 0.3,
        "crop_scale": [0.6, 1.0]
      }
    }

If keys are missing, sensible defaults are used (no jitter and no cropping).

Note: hue jitter is omitted because rotating hue for PIL images is costly and
 provides little benefit for aerial imagery.  You can add it using a similar
 approach if desired.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps

# Pillow 10: BILINEAR moved under Image.Resampling
try:
    RESAMPLE = Image.Resampling.BILINEAR  # Pillow >= 9.1 / 10+
except AttributeError:
    RESAMPLE = Image.BILINEAR             # Pillow < 9.1 fallback

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str) -> dict:
    """Load a JSON configuration from disk, returning an empty dict if missing."""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_label_file(path: Path) -> List[str]:
    """Read all non‑empty lines from a label file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_label_file(path: Path, rows: List[Tuple[int, List[Tuple[float, float]]]]) -> None:
    """Write YOLO‑OBB labels to a file.  Coordinates must already be normalised.

    Each ``rows`` entry is a tuple ``(class_id, kpts)`` where ``kpts`` is a list
    of four ``(x, y)`` pairs.  Values are rounded to six decimals to reduce file
    size and remove floating point noise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, kpts in rows:
            f.write(str(cls) + " " + " ".join(f"{x:.6f} {y:.6f}" for (x, y) in kpts) + "\n")


def copy_if_needed(src: Path, dst: Path) -> None:
    """Copy a file from ``src`` to ``dst`` only if ``dst`` does not exist or
    differs in size.  Parent directories are created as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        dst.write_bytes(src.read_bytes())


# -----------------------------------------------------------------------------
# Geometry / labels helpers
# -----------------------------------------------------------------------------

def parse_yolo_obb_line(line: str) -> Optional[Tuple[int, List[Tuple[float, float]]]]:
    """Parse a single YOLO‑OBB label line into (class, kpts).

    The expected format is ``cls x1 y1 x2 y2 x3 y3 x4 y4`` where coordinates are
    normalised to the range [0,1].  Returns ``None`` if parsing fails.
    """
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
    """Compute the area of a quadrilateral using the shoelace formula."""
    if len(kpts) != 4:
        return 0.0
    xys = kpts + [kpts[0]]
    s = 0.0
    for i in range(4):
        s += xys[i][0] * xys[i + 1][1] - xys[i + 1][0] * xys[i][1]
    return abs(0.5 * s)


def canonicalize_clockwise(kpts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Order points of a quadrilateral into a consistent clockwise orientation.

    Given four unordered points, this function computes the centroid, sorts the
    points by their angle around the centroid (CCW) and then rotates the list so
    that it starts roughly from the top‑left.  This avoids label ambiguities.
    """
    cx = sum(x for x, _ in kpts) / 4.0
    cy = sum(y for _, y in kpts) / 4.0
    pts = []
    for (x, y) in kpts:
        ang = math.atan2(y - cy, x - cx)
        pts.append((x, y, ang))
    pts.sort(key=lambda t: t[2])  # CCW order
    pts = [(x, y) for (x, y, _) in pts]
    # rotate so that the starting point is the top‑left
    start = min(range(4), key=lambda i: (pts[i][1], pts[i][0]))
    return pts[start:] + pts[:start]


def dedupe_rows(rows: List[Tuple[int, List[Tuple[float, float]]]]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Deduplicate rows by rounding coordinates to six decimals and hashing."""
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
    """Clamp a float to the [0,1] range."""
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# -----------------------------------------------------------------------------
# Letterbox resize (like YOLO)
# -----------------------------------------------------------------------------

def letterbox(image: Image.Image, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)) -> Tuple[Image.Image, float, Tuple[float, float]]:
    """Resize and pad an image to ``new_shape`` while preserving aspect ratio.

    Returns the resized image, the scale factor used, and the (dx, dy) padding
    offsets.  The scale factor and offsets can be used to map normalised
    coordinates back to pixel coordinates and vice versa.
    """
    w0, h0 = image.size
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    img = image.resize((nw, nh), RESAMPLE)
    pad_w, pad_h = new_w - nw, new_h - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=color)
    return img, r, (left, top)


# -----------------------------------------------------------------------------
# Affine transformations
# -----------------------------------------------------------------------------

def build_affine_matrix(W: int, H: int, hflip: bool, vflip: bool, angle: float, scale: float,
                        shear_x: float, shear_y: float, translate: Tuple[float, float]) -> List[List[float]]:
    """Construct a composite affine transformation matrix for the given
    operations.  The matrix maps input pixel coordinates to output pixel
    coordinates.  All geometric parameters are expressed relative to a
    normalised coordinate space (-1..1).
    """
    # Convert degrees to radians
    a = math.radians(angle)
    sx, sy = math.radians(shear_x), math.radians(shear_y)
    # Build matrices for each operation around the image centre
    # Translate centre to origin
    cx, cy = W / 2.0, H / 2.0
    # Flip
    M_flip = [[-1 if hflip else 1, 0, 0], [0, -1 if vflip else 1, 0], [0, 0, 1]]
    # Rotation + scale + shear
    cos_a, sin_a = math.cos(a), math.sin(a)
    # Rotation matrix with scale
    M_rot = [[scale * cos_a, -scale * sin_a, 0], [scale * sin_a, scale * cos_a, 0], [0, 0, 1]]
    # Shear matrix
    M_shear = [[1, math.tan(sx), 0], [math.tan(sy), 1, 0], [0, 0, 1]]
    # Combine rotation and shear
    def mat_mul(A, B):
        return [
            [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]
    M = mat_mul(mat_mul(M_rot, M_shear), M_flip)
    # Translate
    tx, ty = translate
    M_trans = [[1, 0, tx * W], [0, 1, ty * H], [0, 0, 1]]
    M = mat_mul(M_trans, M)
    # Move origin back to centre
    M_origin = [[1, 0, cx], [0, 1, cy], [0, 0, 1]]
    M = mat_mul(M_origin, M)
    # Move centre to origin
    M_pre = [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]
    M = mat_mul(M, M_pre)
    return M


def pil_affine(img: Image.Image, M: List[List[float]], fill=(114, 114, 114)) -> Image.Image:
    """Apply an affine transform specified by matrix ``M`` to a PIL image.
    The matrix is provided in a 3×3 homogeneous form; PIL expects a 6‑tuple
    (a, b, c, d, e, f) mapping (x, y) to (ax + by + c, dx + ey + f).
    """
    a, b, c = M[0]
    d, e, f = M[1]
    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample=RESAMPLE, fillcolor=fill)


def apply_affine_to_points(points: List[Tuple[float, float]], M: List[List[float]]) -> List[Tuple[float, float]]:
    """Apply an affine transformation matrix to a list of 2D points."""
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
    """Apply a random affine transform to both an image and its OBB labels.

    ``items_in`` must contain normalised coordinates (0..1) relative to the current
    image size ``size``.  The function samples flip/rotation/scale/translation/
    shear parameters from the provided ranges and returns the transformed image
    along with the transformed, normalised OBB labels.  Degenerate polygons (area
    below a small threshold) are dropped.
    """
    W, H = size
    # Sample parameters
    hflip = rng.random() < hflip_p
    vflip = rng.random() < vflip_p
    angle = rng.uniform(-rotation, rotation) if rotation > 0 else 0.0
    scale = rng.uniform(scale_rng[0], scale_rng[1]) if scale_rng else 1.0
    tx = rng.uniform(-translate, translate) if translate > 0 else 0.0
    ty = rng.uniform(-translate, translate) if translate > 0 else 0.0
    shear_x = rng.uniform(-shear, shear) if shear > 0 else 0.0
    shear_y = rng.uniform(-shear, shear) if shear > 0 else 0.0
    # Build composite matrix and transform image
    M = build_affine_matrix(W, H, hflip, vflip, angle, scale, shear_x, shear_y, (tx, ty))
    img_out = pil_affine(img_in, M, fill=(114, 114, 114))
    # Transform points
    rows_out: List[Tuple[int, List[Tuple[float, float]]]] = []
    for cls, kpts_norm in items_in:
        # convert to pixels
        kpts_px = [(x * W, y * H) for (x, y) in kpts_norm]
        kpts_px2 = apply_affine_to_points(kpts_px, M)
        # back to normalised coords
        kpts_n2 = [(clamp01(x / W), clamp01(y / H)) for (x, y) in kpts_px2]
        kpts_n2 = canonicalize_clockwise(kpts_n2)
        if poly_area_xy(kpts_n2) < 1e-6:
            continue
        rows_out.append((cls, kpts_n2))
    rows_out = dedupe_rows(rows_out)
    return img_out, rows_out


# -----------------------------------------------------------------------------
# Colour jitter
# -----------------------------------------------------------------------------

def apply_color_jitter(img: Image.Image, rng: random.Random, brightness: float = 0.0,
                       contrast: float = 0.0, saturation: float = 0.0) -> Image.Image:
    """Apply random brightness, contrast and saturation adjustments.

    For each property a random factor ``f`` is sampled uniformly from
    ``[1 - v, 1 + v]`` where ``v`` is the corresponding parameter.  If a
    parameter is zero no change is made.  Hue adjustment is omitted because
    rotating hues is costly and seldom useful for aerial images.
    """
    out = img
    if brightness > 0:
        factor = 1.0 + rng.uniform(-brightness, brightness)
        out = ImageEnhance.Brightness(out).enhance(factor)
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        out = ImageEnhance.Contrast(out).enhance(factor)
    if saturation > 0:
        factor = 1.0 + rng.uniform(-saturation, saturation)
        out = ImageEnhance.Color(out).enhance(factor)
    return out


# -----------------------------------------------------------------------------
# OBB‑safe random cropping
# -----------------------------------------------------------------------------

def apply_random_crop(img: Image.Image,
                      items: List[Tuple[int, List[Tuple[float, float]]]],
                      rng: random.Random,
                      crop_prob: float,
                      crop_scale: Tuple[float, float]) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    """Randomly crop an image and adjust OBB labels accordingly.

    A crop is taken with probability ``crop_prob``.  The width and height of the
    crop are sampled as a fraction of the original dimensions from the range
    ``crop_scale``.  Each polygon is transformed into the crop coordinate frame;
    polygons that do not intersect the crop are discarded.  Remaining polygons
    are normalised to the cropped size and canonicalised.  Cropping is applied
    before letterboxing so that the aspect ratio of the final image is restored
    via the usual letterbox resize.
    """
    if crop_prob <= 0 or rng.random() > crop_prob or not items:
        return img, items
    w, h = img.size
    # sample crop size
    scale = rng.uniform(crop_scale[0], crop_scale[1])
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    # ensure crop is smaller than image
    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    # random top‑left corner
    if w == crop_w:
        left = 0
    else:
        left = rng.randint(0, w - crop_w)
    if h == crop_h:
        top = 0
    else:
        top = rng.randint(0, h - crop_h)
    right = left + crop_w
    bottom = top + crop_h
    # crop image
    img_crop = img.crop((left, top, right, bottom))
    items_out: List[Tuple[int, List[Tuple[float, float]]]] = []
    # transform each polygon
    for cls, kpts_norm in items:
        # Convert to pixel coordinates on original image
        kpts_px = [(x * w, y * h) for (x, y) in kpts_norm]
        # Translate into crop coordinate frame
        pts_crop = [(x - left, y - top) for (x, y) in kpts_px]
        # Check if any vertex lies within the crop; if not, skip
        inside = any(0 <= x <= crop_w and 0 <= y <= crop_h for (x, y) in pts_crop)
        if not inside:
            continue
        # Normalize new points relative to crop dimensions
        kpts_n = [(clamp01(x / crop_w), clamp01(y / crop_h)) for (x, y) in pts_crop]
        kpts_n = canonicalize_clockwise(kpts_n)
        # Filter degenerate polygons
        if poly_area_xy(kpts_n) < 1e-6:
            continue
        items_out.append((cls, kpts_n))
    return img_crop, items_out


# -----------------------------------------------------------------------------
# Processing pipeline for splits
# -----------------------------------------------------------------------------

def process_split(split: str,
                  img_dir_in: Path,
                  lbl_dir_in: Path,
                  img_dir_out: Path,
                  lbl_dir_out: Path,
                  image_size: Tuple[int, int],
                  aug_cfg: Dict[str, float],
                  base_copies: int,
                  rng: random.Random) -> None:
    """Process a single data split (train or val).

    For the validation split only letterboxing is applied.  For the training
    split the function copies the base image and then generates ``base_copies``
    augmented variants using colour jitter, random cropping and affine
    transforms.  The augmentation parameters are passed via ``aug_cfg``.
    """
    is_train = split.lower() == "train"
    # Retrieve augmentation params with sensible defaults
    hflip_p = aug_cfg.get("hflip_prob", 0.0)
    vflip_p = aug_cfg.get("vflip_prob", 0.0)
    rotation = aug_cfg.get("rotation", 0.0)
    translate = aug_cfg.get("translate", 0.0)
    scale_rng = tuple(aug_cfg.get("scale_range", [1.0, 1.0]))
    shear = aug_cfg.get("shear", 0.0)
    # New augmentation2 params
    aug2 = aug_cfg.get("augment2", {}) if isinstance(aug_cfg.get("augment2", {}), dict) else {}
    cj = aug2.get("color_jitter", {}) if isinstance(aug2.get("color_jitter", {}), dict) else {}
    brightness = float(cj.get("brightness", 0.0))
    contrast = float(cj.get("contrast", 0.0))
    saturation = float(cj.get("saturation", 0.0))
    crop_prob = float(aug2.get("crop_prob", 0.0))
    crop_scale = tuple(aug2.get("crop_scale", [1.0, 1.0]))
    # Iterate over images
    img_dir_out.mkdir(parents=True, exist_ok=True)
    lbl_dir_out.mkdir(parents=True, exist_ok=True)
    for fname in sorted(os.listdir(img_dir_in)):
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        img_path = img_dir_in / fname
        lbl_path = lbl_dir_in / f"{stem}.txt"
        # Read image
        img = Image.open(img_path).convert("RGB")
        # Parse labels
        raw_lines = read_label_file(lbl_path)
        items: List[Tuple[int, List[Tuple[float, float]]]] = []
        for ln in raw_lines:
            parsed = parse_yolo_obb_line(ln)
            if parsed:
                items.append(parsed)
        # Always letterbox+save original for val and train
        img_lb, r, (dx, dy) = letterbox(img, new_shape=image_size)
        rows: List[Tuple[int, List[Tuple[float, float]]]] = []
        for cls, kpts in items:
            # Map normalised coords from original to letterboxed
            # kpts are normalised relative to original image
            kpts_px = [(x * img.width, y * img.height) for (x, y) in kpts]
            # scale and offset
            kpts_lb = [((x * r + dx) / image_size[0], (y * r + dy) / image_size[1]) for (x, y) in kpts_px]
            kpts_lb = canonicalize_clockwise(kpts_lb)
            if poly_area_xy(kpts_lb) < 1e-6:
                continue
            rows.append((cls, kpts_lb))
        out_img_path = img_dir_out / f"{stem}.jpg"
        out_lbl_path = lbl_dir_out / f"{stem}.txt"
        img_lb.save(out_img_path)
        write_label_file(out_lbl_path, dedupe_rows(rows))
        # For training, generate augmented copies
        if is_train and base_copies > 0 and rows:
            for copy_idx in range(base_copies):
                # Colour jitter on the original image
                aug_img = apply_color_jitter(img, rng, brightness, contrast, saturation)
                aug_items = items.copy()
                # Random crop
                aug_img, aug_items = apply_random_crop(aug_img, aug_items, rng, crop_prob, crop_scale)
                # Letterbox
                img_lb2, r2, (dx2, dy2) = letterbox(aug_img, new_shape=image_size)
                # Map original (or cropped) items into letterbox
                rows_aug: List[Tuple[int, List[Tuple[float, float]]]] = []
                for cls, kpts in aug_items:
                    # kpts are normalised relative to aug_img
                    w_i, h_i = aug_img.size
                    kpts_px2 = [(x * w_i, y * h_i) for (x, y) in kpts]
                    kpts_lb2 = [((x * r2 + dx2) / image_size[0], (y * r2 + dy2) / image_size[1]) for (x, y) in kpts_px2]
                    kpts_lb2 = canonicalize_clockwise(kpts_lb2)
                    if poly_area_xy(kpts_lb2) < 1e-6:
                        continue
                    rows_aug.append((cls, kpts_lb2))
                # Random affine
                if rows_aug:
                    img_aff, rows_aff = apply_random_affine(
                        img_lb2,
                        rows_aug,
                        size=image_size,
                        rng=rng,
                        hflip_p=hflip_p,
                        vflip_p=vflip_p,
                        rotation=rotation,
                        translate=translate,
                        scale_rng=scale_rng,
                        shear=shear,
                    )
                    # Save augmented copy
                    aug_name = f"{stem}_aug{copy_idx}"
                    img_aff.save(img_dir_out / f"{aug_name}.jpg")
                    write_label_file(lbl_dir_out / f"{aug_name}.txt", dedupe_rows(rows_aff))


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Data augmentation with OBB‑safe cropping and colour jitter.")
    ap.add_argument("--config", default="config/config.json", type=str, help="Path to config.json")
    ap.add_argument("--raw-images", default=None, type=str, help="Override raw images dir")
    ap.add_argument("--raw-labels", default=None, type=str, help="Override raw labels dir")
    ap.add_argument("--proc-images", default=None, type=str, help="Override processed images dir")
    ap.add_argument("--proc-labels", default=None, type=str, help="Override processed labels dir")
    ap.add_argument("--train-split", default="train", type=str)
    ap.add_argument("--val-split", default="val", type=str)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # Pull augmented copies per image from config (no CLI)
    aug = cfg.get("augment", {}) or {}
    # prefer augment.base_copies; fallback to augment.max_extra_per_image
    base_copies = int(aug.get("base_copies", aug.get("max_extra_per_image", 0)))
    base_copies = max(0, base_copies)
    # Paths
    paths = cfg.get("paths", {})
    raw_images = Path(args.raw_images or paths.get("raw_images", "files/raw/images"))
    raw_labels = Path(args.raw_labels or paths.get("raw_labels", "files/raw/labels"))
    proc_images = Path(args.proc_images or paths.get("proc_images", "files/processed/images"))
    proc_labels = Path(args.proc_labels or paths.get("proc_labels", "files/processed/labels"))
    # Dataset size
    ds_cfg = cfg.get("dataset", {})
    W, H = tuple(ds_cfg.get("image_size", [640, 640]))
    # Augmentation parameters (including new ones nested under augment2)
    aug_cfg = cfg.get("augment", {}).copy()
    # To allow nested augment2, we insert it as a subdict on aug_cfg if present
    # if "augment2" in cfg:
        # Do not override if augment2 is nested under augment
    #    aug_cfg["augment2"] = cfg["augment2"]
    rng = random.Random(args.seed)
    # Ensure output directories exist
    for split in (args.train_split, args.val_split):
        (proc_images / split).mkdir(parents=True, exist_ok=True)
        (proc_labels / split).mkdir(parents=True, exist_ok=True)
    # Process validation split (no augmentation)
    process_split(
        split=args.val_split,
        img_dir_in=raw_images / args.val_split,
        lbl_dir_in=raw_labels / args.val_split,
        img_dir_out=proc_images / args.val_split,
        lbl_dir_out=proc_labels / args.val_split,
        image_size=(W, H),
        aug_cfg=aug_cfg,
        base_copies=base_copies,
        rng=rng,
    )
    # Process training split with augmentation
    process_split(
        split=args.train_split,
        img_dir_in=raw_images / args.train_split,
        lbl_dir_in=raw_labels / args.train_split,
        img_dir_out=proc_images / args.train_split,
        lbl_dir_out=proc_labels / args.train_split,
        image_size=(W, H),
        aug_cfg=aug_cfg,
        base_copies=base_copies,
        rng=rng,
    )
    print("Done. Processed data at:", proc_images.parent)


if __name__ == "__main__":
    main()