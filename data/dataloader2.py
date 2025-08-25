import os
import torch
from torch.utils.data import DataLoader

from data.dataset2 import YOLODatasetNorm


def create_loaders2(cfg: dict):
    """Construct DataLoaders for the training and validation datasets with normalisation.

    This helper mimics the original ``create_loaders`` but instantiates
    ``YOLODatasetNorm`` which applies per‑channel mean and standard deviation
    normalisation to each image.  The mean and std values are read from
    ``cfg['dataset']['mean']`` and ``cfg['dataset']['std']`` if available;
    otherwise ImageNet statistics are used.  All other training parameters
    (batch size, number of workers, etc.) are taken from the existing config.

    Args:
        cfg (dict): Loaded configuration dictionary.

    Returns:
        tuple: ``(train_loader, val_loader)``
    """
    paths = cfg.get("paths", {})
    train_img_dir = os.path.join(paths.get("proc_images", "files/processed/images"), "train")
    train_lbl_dir = os.path.join(paths.get("proc_labels", "files/processed/labels"), "train")
    val_img_dir = os.path.join(paths.get("proc_images", "files/processed/images"), "val")
    val_lbl_dir = os.path.join(paths.get("proc_labels", "files/processed/labels"), "val")
    # Training parameters
    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 16)
    num_workers = training_cfg.get("num_workers", 4)
    pin_memory = training_cfg.get("pin_memory", torch.cuda.is_available())
    # Normalisation parameters
    ds_cfg = cfg.get("dataset", {})
    mean = ds_cfg.get("mean", [0.485, 0.456, 0.406])
    std = ds_cfg.get("std", [0.229, 0.224, 0.225])
    # Instantiate datasets
    train_dataset = YOLODatasetNorm(
        images_dir=train_img_dir,
        labels_dir=train_lbl_dir,
        mean=mean,
        std=std,
        transforms=None,
    )
    val_dataset = YOLODatasetNorm(
        images_dir=val_img_dir,
        labels_dir=val_lbl_dir,
        mean=mean,
        std=std,
        transforms=None,
    )
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    return train_loader, val_loader