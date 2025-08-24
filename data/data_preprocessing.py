import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import torch
import torchvision.transforms.v2 as T
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from utils.utils import load_config


# --------------------------
# I/O helpers (YOLO OBB)
# --------------------------

def read_yolo_obb_label(label_path: str) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    """
    Reads YOLO OBB labels (normalized):
      each line: cls x1 y1 x2 y2 x3 y3 x4 y4
    Returns:
      keypoints:  [ [ (x,y)*4 ], ... ]  # normalized [0..1]
      classes:    [ cls, ... ]
    """
    keypoints, classes = [], []
    if not os.path.exists(label_path):
        return keypoints, classes
    with open(label_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 9:
                continue
            cls = int(vals[0])
            coords = list(map(float, vals[1:]))
            pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            keypoints.append(pts)
            classes.append(cls)
    return keypoints, classes


def write_yolo_obb_label(label_path: str, keypoints: List[List[Tuple[float, float]]], classes: List[int]):
    """
    Writes YOLO OBB labels (normalized).
    """
    with open(label_path, 'w') as f:
        for cls, quad in zip(classes, keypoints):
            flat = [str(x) for (x, y) in quad for x in (x, y)]
            f.write(f"{cls} {' '.join(flat)}\n")


# --------------------------
# Transforms
# --------------------------

def build_base_transform(size_hw: List[int]) -> T.Compose:
    """
    Base deterministic pipeline: converts to Tensor image and resizes only.
    No normalization here; keep it for training time.
    """
    return T.Compose([
        T.ToImage(),                          # ensures CxHxW
        T.Resize(size_hw),                    # (H, W)
        T.ToDtype(torch.float32, scale=True)  # [0..1] float
    ])


def build_aug_transform(cfg: dict) -> T.Compose:
    """
    OBB-safe augmentation pipeline.
    Only geometric transforms that update 'keypoints' are used.
    """
    aug = cfg.get('augment', {})
    degrees = float(aug.get('rotation', 20))          # ±deg
    hflip_p = float(aug.get('hflip_prob', 0.5))
    vflip_p = float(aug.get('vflip_prob', 0.0))       # optional
    translate = float(aug.get('translate', 0.1))      # fraction of image dims
    scale_min, scale_max = map(float, aug.get('scale_range', [0.9, 1.1]))
    shear = float(aug.get('shear', 0.0))

    size_hw = cfg['dataset']['image_size']

    tfms = [
        T.ToImage(),
        # IMPORTANT: apply geometric augs BEFORE resize so Resize is last and consistent
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomVerticalFlip(p=vflip_p) if vflip_p > 0 else T.Identity(),
        # Affine includes rotation/translate/scale/shear and is keypoint-aware in v2
        T.RandomAffine(
            degrees=degrees,
            translate=(translate, translate) if translate > 0 else None,
            scale=(scale_min, scale_max) if scale_min != 1.0 or scale_max != 1.0 else None,
            shear=shear if shear > 0 else None,
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.Resize(size_hw),
        T.ToDtype(torch.float32, scale=True),
        # Color jitter etc. (doesn't affect keypoints)
        T.ColorJitter(
            brightness=aug.get('brightness', 0.2),
            contrast=aug.get('contrast', 0.2),
            saturation=aug.get('saturation', 0.2),
            hue=aug.get('hue', 0.02)
        ) if aug.get('color_jitter', True) else T.Identity(),
    ]
    # Remove Identities to keep it tidy
    tfms = [t for t in tfms if not isinstance(t, T.Identity)]
    return T.Compose(tfms)


# --------------------------
# Geometry utils
# --------------------------

def denorm_keypoints(keypoints: List[List[Tuple[float, float]]], W: int, H: int) -> torch.Tensor:
    """
    Convert normalized keypoints -> pixel coordinates.
    Returns tensor [N, 4, 2] in pixels.
    """
    if not keypoints:
        return torch.zeros((0, 4, 2), dtype=torch.float32)
    kp = torch.tensor(keypoints, dtype=torch.float32)  # [N, 4, 2] in [0..1]
    kp[..., 0] *= W
    kp[..., 1] *= H
    return kp


def renorm_keypoints(kp_px: torch.Tensor, W: int, H: int) -> List[List[Tuple[float, float]]]:
    """
    Convert pixel keypoints -> normalized [0..1] and clamp to [0,1].
    kp_px: [N, 4, 2] pixels
    Returns nested Python list [[(x,y)*4], ...].
    """
    if kp_px.numel() == 0:
        return []
    kp = kp_px.clone()
    kp[..., 0] /= max(W, 1)
    kp[..., 1] /= max(H, 1)
    kp.clamp_(0.0, 1.0)
    out = []
    for quad in kp.tolist():
        quad_list = [(quad[i][0], quad[i][1]) for i in range(4)]
        out.append(quad_list)
    return out


# --------------------------
# Oversampling helpers
# --------------------------

def scan_class_stats(labels_dir: str) -> Tuple[Counter, Dict[str, set]]:
    """
    Returns:
      - counts: Counter {cls_id: count of objects in split}
      - img2classes: mapping basename -> set of classes present in that image
    """
    counts = Counter()
    img2classes = defaultdict(set)
    for f in os.listdir(labels_dir):
        if not f.endswith('.txt'):
            continue
        path = os.path.join(labels_dir, f)
        with open(path, 'r') as fh:
            for line in fh:
                vals = line.strip().split()
                if len(vals) != 9:
                    continue
                cls = int(vals[0])
                counts[cls] += 1
                img2classes[os.path.splitext(f)[0]].add(cls)
    return counts, img2classes


def oversample_factor_for_image(img_base: str, img2classes: Dict[str, set],
                                class_counts: Counter, target_per_class: int,
                                max_extra: int) -> int:
    """
    For a given image (with possibly multiple classes), compute how many extra
    augmented copies we should generate.
    """
    if img_base not in img2classes:
        return 0
    factors = []
    for c in img2classes[img_base]:
        cnt = max(class_counts.get(c, 1), 1)
        need = max(target_per_class - cnt, 0)
        # proportional factor for this class
        f = int((need + cnt - 1) // cnt)  # ceil_div
        factors.append(f)
    if not factors:
        return 0
    # We will create up to 'max_extra' additional variants
    return min(max(factors), max_extra)


# --------------------------
# Main processing
# --------------------------

def process_split(split: str, cfg: dict, paths: dict):
    raw_images = os.path.join(paths['raw_images'], split)
    raw_labels = os.path.join(paths['raw_labels'], split)
    proc_images = os.path.join(paths['proc_images'], split)
    proc_labels = os.path.join(paths['proc_labels'], split)

    os.makedirs(proc_images, exist_ok=True)
    os.makedirs(proc_labels, exist_ok=True)

    size_hw = cfg['dataset']['image_size']  # [H, W]
    base_tf = build_base_transform(size_hw)
    aug_tf = build_aug_transform(cfg)

    # Oversampling params (train only)
    do_over = (split == 'train')
    aug_cfg = cfg.get('augment', {})
    max_extra_per_image = int(aug_cfg.get('max_extra_per_image', 2))  # limit per image
    balance_mode = aug_cfg.get('balance_mode', 'max')  # 'max' or an int
    target_per_class = None

    class_counts, img2classes = Counter(), {}
    if do_over:
        class_counts, img2classes = scan_class_stats(raw_labels)
        if balance_mode == 'max':
            target_per_class = max(class_counts.values()) if class_counts else 0
        elif isinstance(balance_mode, int):
            target_per_class = balance_mode
        else:
            target_per_class = max(class_counts.values()) if class_counts else 0

    for img_file in tqdm(os.listdir(raw_images), desc=f"Processing {split}"):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(raw_images, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(raw_labels, label_file)

        # Load image and dims
        image = read_image(img_path)  # uint8 [0..255], CxHxW
        _, H0, W0 = image.shape

        # Load normalized labels
        keypoints_n, classes = read_yolo_obb_label(label_path)
        # Denormalize to pixels for geometry-aware transforms
        keypoints_px = denorm_keypoints(keypoints_n, W0, H0)

        # Targets for torchvision v2 transforms
        targets = {
            'keypoints': keypoints_px,                   # [N,4,2] pixels
            'labels': torch.tensor(classes, dtype=torch.int64)
        }

        # -------- Base (no random aug) --------
        # Resize only (and convert types); torchvision v2 transforms are target-aware
        img_b, tgt_b = base_tf(image, targets)
        Hb, Wb = img_b.shape[-2:]
        kps_b_norm = renorm_keypoints(tgt_b['keypoints'], Wb, Hb)

        # Save base
        save_image(img_b, os.path.join(proc_images, img_file))
        write_yolo_obb_label(os.path.join(proc_labels, label_file), kps_b_norm, tgt_b['labels'].tolist())

        # -------- Oversampling (train only) --------
        if do_over and target_per_class and max_extra_per_image > 0:
            base = os.path.splitext(img_file)[0]
            extra = oversample_factor_for_image(
                base, img2classes, class_counts, target_per_class, max_extra_per_image
            )
            for i in range(extra):
                img_a, tgt_a = aug_tf(image, targets)   # NEW random aug each call
                Ha, Wa = img_a.shape[-2:]
                kps_a_norm = renorm_keypoints(tgt_a['keypoints'], Wa, Ha)

                # save with suffix
                out_name = f"{base}_aug{i+1:02d}{Path(img_file).suffix}"
                save_image(img_a, os.path.join(proc_images, out_name))
                write_yolo_obb_label(
                    os.path.join(proc_labels, f"{base}_aug{i+1:02d}.txt"),
                    kps_a_norm,
                    tgt_a['labels'].tolist()
                )


def main():
    cfg = load_config('config/config.json')

    # Optional: set seed for reproducibility of aug choices
    seed = cfg.get('seed', None)
    if seed is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass

    paths = {
        'raw_images': cfg['paths']['raw_images'],
        'raw_labels': cfg['paths']['raw_labels'],
        'proc_images': cfg['paths']['proc_images'],
        'proc_labels': cfg['paths']['proc_labels']
    }

    # Train: base + oversampling; Val: base only
    for split in ['train', 'val']:
        process_split(split, cfg, paths)

    print("[Done] Preprocessing completato.")


if __name__ == '__main__':
    main()