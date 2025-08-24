import os
import torch
from torch.utils.data import DataLoader
from data.dataset import YOLODataset
from utils.utils import load_config


"""def create_transforms(cfg):
    
    Costruisce e restituisce le trasformazioni da applicare al Dataset.

    cfg: dict con possibili chiavi:
      - image_size: [w, h]
      - mean: [float, float, float]
      - std: [float, float, float]
      - augment: { 'hflip_prob': float, 'rotation': int }
    
    from torchvision import transforms as T
    size = cfg.get('image_size', [640, 640])
    mean = cfg.get('mean', [0.485, 0.456, 0.406])
    std = cfg.get('std', [0.229, 0.224, 0.225])
    aug = cfg.get('augment', {})

    pipeline = [
        T.Resize(size),
        T.RandomHorizontalFlip(aug.get('hflip_prob', 0.5)),
        T.RandomRotation(aug.get('rotation', 10)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(pipeline)"""


def create_loaders(cfg: dict):
    """
    Returns train_loader, val_loader using config['paths'].
    """
    paths = cfg['paths']
    train_img_dir = os.path.join(paths['proc_images'], 'train')
    train_lbl_dir = os.path.join(paths['proc_labels'], 'train')
    val_img_dir   = os.path.join(paths['proc_images'], 'val')
    val_lbl_dir   = os.path.join(paths['proc_labels'], 'val')

    bs = cfg['training']['batch_size']
    nw = cfg['training']['num_workers']
    pm = cfg['training'].get('pin_memory', torch.cuda.is_available())

    # Dataset
    train_dataset = YOLODataset(
        images_dir=train_img_dir,
        labels_dir=train_lbl_dir,
        transforms=None
    )
    val_dataset = YOLODataset(
        images_dir=val_img_dir,
        labels_dir=val_lbl_dir,
        transforms=None
    )

    # Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm ,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    return train_loader, val_loader