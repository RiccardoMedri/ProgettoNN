# scripts/train.py
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ultralytics import settings, YOLO
import torch
from torch.utils.tensorboard import SummaryWriter


def _pick_project_dir(cfg: dict) -> str:
    # supporta sia logging.project_dir che paths.log_dir/logs_dir
    log_cfg = cfg.get("logging", {})
    paths = cfg.get("paths", {})
    return (
        log_cfg.get("project_dir")
        or paths.get("log_dir")
        or paths.get("logs_dir")
        or "outputs/logs"
    )


def train_ultra(config_path: str) -> Optional[str]:
    """
    Allena un modello YOLO (OBB o standard) usando direttamente Ultralytics.
    Ritorna il path al checkpoint 'best.pt' se disponibile, altrimenti None.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})

    # Sorgente modello:
    # - .yaml => architettura from-scratch
    # - .pt   => pesi pre-addestrati o tuoi pesi
    model_name = model_cfg.get("name", "yolo11n-obb.yaml")
    weights_path = model_cfg.get("weights_path")  # opzionale: override con un .pt specifico
    model_source: Union[str, os.PathLike] = weights_path or model_name

    # Dimensione input
    imgsz = int(model_cfg.get("input_size", 640))

    # Dati: usa il percorso dichiarato in config, fallback sensato a config/dataset.yaml
    data_yaml = paths.get("dataset_yaml") or "config/dataset.yaml"

    # Hyper base
    epochs = int(train_cfg.get("epochs", 50))
    batch = int(train_cfg.get("batch_size", 8))
    workers = int(train_cfg.get("num_workers", 2))
    seed = cfg.get("seed", None)
    if seed is not None:
        seed = int(seed)
    pretrained = bool(model_cfg.get("pretrained", True))

    # Dispositivo
    device = 0 if torch.cuda.is_available() else "cpu"

    # Output/logging
    project = _pick_project_dir(cfg)
    run_name = cfg.get("logging", {}).get("run_name", "exp")

    # Imposta il seme se richiesto
    if seed is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Abilita TensorBoard (scalari gestiti da Ultralytics)
    settings.update({"tensorboard": True})

    # ---- TensorBoard writer & callback (immagini) ----
    tb_logdir = os.path.join(project, run_name, "tb")
    os.makedirs(tb_logdir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_logdir)

    def _log_image_if_exists(tag: str, path: Path, step: int):
        try:
            if path.exists():
                img = Image.open(path).convert("RGB")
                arr = np.array(img)  # H x W x C
                if arr.ndim == 3:
                    arr = arr.transpose(2, 0, 1)  # C x H x W
                tb_writer.add_image(tag, arr, global_step=step)
        except Exception:
            # Non bloccare il training per un errore di logging
            pass

    def on_fit_epoch_end(trainer):
        """
        Callback Ultralytics: chiamato a fine epoch.
        Logga su TensorBoard alcune immagini/artifacts se presenti nella cartella del run.
        """
        try:
            save_dir = Path(trainer.save_dir)
            epoch = getattr(trainer, "epoch", 0)

            _log_image_if_exists("labels",               save_dir / "labels.jpg",               epoch)
            _log_image_if_exists("results",              save_dir / "results.png",              epoch)
            _log_image_if_exists("confusion_matrix",     save_dir / "confusion_matrix.png",     epoch)
            _log_image_if_exists("PR_curve",             save_dir / "PR_curve.png",             epoch)
            _log_image_if_exists("labels_correlogram",   save_dir / "labels_correlogram.jpg",   epoch)
        except Exception:
            pass

    def on_train_end(trainer):
        # Chiudi il writer in modo pulito quando il training finisce
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    # Istanzia modello Ultralytics
    yolo = YOLO(model_source)

    # Registra i callback con l'API ufficiale (niente `callbacks=` dentro .train())
    yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    yolo.add_callback("on_train_end", on_train_end)

    # Opzioni extra passate direttamente a Ultralytics
    extra = train_cfg.get("ultra", {})

    # Lancia il training
    results = yolo.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        project=project,
        name=run_name,
        pretrained=pretrained,
        **extra,
    )

    # Recupera un path “best.pt” affidabile
    ckpt = None
    try:
        # versioni recenti espongono il trainer con save_dir
        save_dir = Path(yolo.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        ckpt = str(best) if best.exists() else None
    except Exception:
        # fallback: prova results.save_dir
        try:
            save_dir = Path(getattr(results, "save_dir", project)) / run_name / "weights"
            best = save_dir / "best.pt"
            ckpt = str(best) if best.exists() else None
        except Exception:
            ckpt = None

    return ckpt