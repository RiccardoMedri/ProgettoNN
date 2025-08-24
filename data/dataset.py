import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    """
    PyTorch Dataset per data di detection in formato YOLO-like.

    Ogni immagine deve avere un file label corrispondente *.txt* con righe:
        class_id x1 y1 x2 y2 x3 y3 x4 y4 [x1' y1' ...]*
    dove ogni 8 valori rappresenta i 4 punti normalizzati di un bounding polygon.

    Restituisce:
        image (torch.Tensor): immagine trasformata (C, H, W)
        target (dict): {
            'boxes': Tensor[N, 8] (normalized coords),
            'labels': Tensor[N]
        }
    """
    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Applica trasformazioni solo sull'immagine
        if self.transforms:
            img = self.transforms(img)
        else:
            # converte in tensor se non ci sono transforms
            img = torch.from_numpy(
                np.array(img).transpose(2, 0, 1)
            ).float() / 255.0

        # Parsing label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9 or len(parts[1:]) % 8 != 0:
                        continue  # formato non valido
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    # suddivide in gruppi di 8 valori
                    for i in range(0, len(coords), 8):
                        poly = coords[i:i+8]
                        boxes.append(poly)
                        labels.append(cls_id)

        # Converte in tensori
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 8), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return img, target