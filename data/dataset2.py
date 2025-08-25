import os
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class YOLODatasetNorm(Dataset):
    """PyTorch Dataset for oriented bounding box detection with per‑channel normalisation.

    This dataset mirrors the functionality of the original ``YOLODataset`` but
    additionally performs mean/standard‑deviation normalisation on each image.
    Bounding box parsing remains unchanged: labels are expected in the YOLO‑OBB
    format of class id followed by 8 normalised coordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Args:
        images_dir (str): Directory containing processed images.
        labels_dir (str): Directory containing corresponding label files.
        mean (List[float], optional): Per‑channel means used to centre each pixel.
            Defaults to the ImageNet means (0.485, 0.456, 0.406).
        std (List[float], optional): Per‑channel standard deviations used to scale
            each pixel.  Defaults to the ImageNet stds (0.229, 0.224, 0.225).
        transforms (callable, optional): Optional transform to apply to the PIL
            image prior to conversion.  When provided it should return a tensor
            with values in [0,1] before normalisation; normalisation is always
            applied after the transform.

    Returns:
        tuple: (image, target) where image is a ``torch.FloatTensor`` of shape
        (C,H,W) and ``target`` is a dict with keys ``boxes`` (tensor of shape
        (N,8)) and ``labels`` (tensor of shape (N,)).
    """

    def __init__(self,
                 images_dir: str,
                 labels_dir: str,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 transforms=None) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        # Use ImageNet statistics as a reasonable default if none provided
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load an image file as a normalised tensor.

        If a transform pipeline is defined it is applied first and must
        output a tensor in [0,1].  Afterwards the image is normalised by
        subtracting ``mean`` and dividing by ``std`` per channel.
        """
        img = Image.open(path).convert("RGB")
        if self.transforms:
            img_t = self.transforms(img)
            # Expect transforms to output [0,1] floats
        else:
            arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
            img_t = torch.from_numpy(arr.transpose(2, 0, 1))  # C,H,W
        # Normalise per channel
        mean = torch.tensor(self.mean, dtype=img_t.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=img_t.dtype).view(-1, 1, 1)
        img_t = (img_t - mean) / std
        return img_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = self._load_image(img_path)
        # Parse labels
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, stem + ".txt")
        boxes: List[List[float]] = []
        labels: List[int] = []
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9 or (len(parts) - 1) % 8 != 0:
                        continue
                    cls_id = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))
                    for i in range(0, len(coords), 8):
                        poly = coords[i:i + 8]
                        boxes.append(poly)
                        labels.append(cls_id)
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 8), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return img, target