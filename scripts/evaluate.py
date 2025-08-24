# scripts/evaluate.py
import json
from pathlib import Path
from typing import Optional, Union

import torch
from ultralytics import YOLO


def _pick_project_dir(cfg: dict) -> str:
    """
    Sceglie la cartella per i log/output:
    - prima guarda logging.project_dir
    - poi paths.log_dir/logs_dir
    - infine default "outputs/logs"
    """
    log_cfg = cfg.get("logging", {})
    paths = cfg.get("paths", {})
    return (
        log_cfg.get("project_dir")
        or paths.get("log_dir")
        or paths.get("logs_dir")
        or "outputs/logs"
    )


def evaluate_ultra(
    config_path: str,
    weights_path: Optional[str] = None,
    split: str = "val",
) -> dict:
    """
    Valuta un modello YOLO (OBB o standard) con Ultralytics.
    Ritorna un dict di metriche (se disponibile) o un dict vuoto.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})
    train_cfg = cfg.get("training", {})  # usato come fallback per batch size
    paths = cfg.get("paths", {})

    # Sorgente del modello: priorità a --weights; poi model.weights_path; poi model.name
    model_source: Union[str, Path] = (
        weights_path
        or model_cfg.get("weights_path")
        or model_cfg.get("name", "yolo11n-obb.pt")
    )

    # YAML del dataset (assicurati che punti al file giusto, es: config/dataset.yaml)
    data_yaml = paths.get("dataset_yaml") or "config/dataset.yaml"

    # Grandezza immagine e batch
    imgsz = int(model_cfg.get("input_size", 640))
    batch = int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 8)))

    # Dispositivo
    device = 0 if torch.cuda.is_available() else "cpu"

    # Output/logging
    project = _pick_project_dir(cfg)
    run_name = cfg.get("logging", {}).get("eval_name", "eval")

    # Istanzia il modello Ultralytics e lancia la valutazione
    yolo = YOLO(model_source)
    metrics = yolo.val(
        data=data_yaml,
        split=split,      # "val" o "test"
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        save_json=False,  # metti True se vuoi il JSON COCO (se supportato)
        plots=True,       # salva confusion matrix/PR curve/curves varie
        verbose=True,
    )

    # Prova a restituire un dict pulito di metriche
    out = {}
    # results_dict è disponibile nelle versioni recenti
    rd = getattr(metrics, "results_dict", None)
    if isinstance(rd, dict):
        out.update(rd)

    # Aggiungi riferimenti a cartelle utili (save_dir)
    try:
        save_dir = Path(getattr(metrics, "save_dir", Path(project) / run_name))
        out["save_dir"] = str(save_dir)
    except Exception:
        pass

    return out