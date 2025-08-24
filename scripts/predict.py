# scripts/predict.py
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch
from ultralytics import YOLO


def _pick_project_dir(cfg: dict, fallback: str) -> str:
    """
    Sceglie la cartella per i log/output:
    - prima logging.project_dir
    - poi paths.figures_dir (per le predizioni ha senso)
    - poi paths.log_dir / logs_dir
    - infine fallback
    """
    log_cfg = cfg.get("logging", {})
    paths = cfg.get("paths", {})
    return (
        log_cfg.get("project_dir")
        or paths.get("figures_dir")
        or paths.get("log_dir")
        or paths.get("logs_dir")
        or fallback
    )


def predict_ultra(
    config_path: str,
    source: Union[str, Path],
    weights_path: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    save_txt: bool = False,
    save_conf: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Esegue la predizione (detect/OBB) con Ultralytics YOLO e salva gli overlay.

    Ritorna un dict con:
      - save_dir: cartella dove sono stati salvati i risultati
      - num: numero di elementi processati
      - inputs: lista dei path di input
      - saved: lista dei path attesi dei file salvati (best-effort)
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    paths = cfg.get("paths", {})
    log_cfg = cfg.get("logging", {})

    model_source = (
        weights_path
        or model_cfg.get("weights_path")
        or model_cfg.get("name", "yolo11n-obb.pt")
    )

    imgsz = int(model_cfg.get("input_size", 640))
    device = 0 if torch.cuda.is_available() else "cpu"

    # soglie
    conf = float(conf if conf is not None else model_cfg.get("conf", 0.25))
    iou = float(iou if iou is not None else model_cfg.get("iou", 0.7))

    # output
    project = _pick_project_dir(cfg, "outputs/figures")
    run_name = log_cfg.get("pred_name", "preds")

    # Istanzia modello e lancia la predizione
    yolo = YOLO(model_source)
    results = yolo.predict(
        source=str(source),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=run_name,
        stream=False,
        verbose=True,
    )

    # Recupera save_dir in modo robusto
    save_dir: Optional[Path] = None
    try:
        save_dir = Path(yolo.predictor.save_dir)  # versioni recenti
    except Exception:
        try:
            if results:
                save_dir = Path(getattr(results[0], "save_dir", project))  # fallback
            else:
                save_dir = Path(project) / run_name
        except Exception:
            save_dir = Path(project) / run_name

    inputs: List[str] = []
    saved: List[str] = []

    for r in (results or []):
        in_path = str(getattr(r, "path", ""))
        inputs.append(in_path)
        if save and save_dir is not None and in_path:
            saved.append(str(save_dir / Path(in_path).name))

    return {
        "save_dir": str(save_dir) if save_dir is not None else None,
        "num": len(results or []),
        "inputs": inputs,
        "saved": saved,
    }