import cv2
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import torch
from ultralytics import YOLO

def _pick_project_dir(cfg: dict, fallback: str) -> str:
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
) -> Dict[str, Any]:
    """Run YOLO prediction on images or videos, display live results and save annotated video."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg   = cfg.get("model", {})
    predict_cfg = cfg.get("predict", {})
    log_cfg     = cfg.get("logging", {})

    # choose weights – same logic as before
    model_source = (
        weights_path
        or model_cfg.get("weights_path")
        or model_cfg.get("name", "yolo11n-obb.pt")
    )

    # override image size and thresholds from predict dict when present
    # fall back to model section or sensible defaults
    imgsz = predict_cfg.get("image_size", [model_cfg.get("input_size", 640)])[0]
    conf  = float(conf if conf is not None else predict_cfg.get("conf_thres", model_cfg.get("conf", 0.25)))
    iou   = float(iou if iou is not None else predict_cfg.get("iou_thres", 0.7))

    # whether to save annotated video/frames
    save_annotated = bool(predict_cfg.get("save_annotated", True))

    show_result = bool(predict_cfg.get("show_result", False))

    device  = 0 if torch.cuda.is_available() else "cpu"
    project = _pick_project_dir(cfg, "outputs/figures")
    run_name = log_cfg.get("pred_name", "preds")

    yolo = YOLO(model_source)
    # run prediction with streaming so we can display frames
    results = yolo.predict(
        source=str(source),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        save=save_annotated,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=run_name,
        show=show_result,
        stream=True,       # get a generator of per-frame results
        verbose=True,
    )

    # display results in one OpenCV window while processing
    for result in results:
        annotated_frame = result.plot()  # overlay boxes/labels
        cv2.imshow("Live detection", annotated_frame)
        # press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # infer where Ultralytics saved the annotated video/images
    save_dir = None
    try:
        save_dir = Path(yolo.predictor.save_dir)
    except Exception:
        # fallback: project/run_name
        save_dir = Path(project) / run_name

    return {
        "save_dir": str(save_dir) if save_dir is not None else None,
        "num": len(results),
        "inputs": [str(source)],
        "saved": [str(save_dir)] if save_annotated and save_dir is not None else [],
    }