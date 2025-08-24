# models/model.py
import json
from collections.abc import Mapping
from typing import List, Dict, Any, Optional
from ultralytics.utils.ops import xyxyxyxy2xywhr
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class DetectionModel(nn.Module):
    """
    Wrapper per modelli YOLO OBB (Oriented Bounding Boxes).

    __init__(config_path: str)
      - Legge il JSON di config e carica il modello Ultralytics (pt o yaml).
      - Se 'pretrained' è False, resetta i pesi del sotto-modello torch.

    forward(images, targets=None)
      - TRAIN: costruisce il batch-dict {'img','cls','bboxes','batch_idx'} e
               restituisce {'loss': <tensor scalare>}.
      - EVAL/PRED: usa il wrapper YOLO per predire e ritorna list[dict] con
                   'boxes' (N,8), 'labels' (N,), 'scores' (N,).
    """

    def __init__(self, config_path: str):
        super().__init__()
        print("[Model] models/model.py v2 attivo (imgsz iniziale configurata)")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        mcfg = cfg.get("model", {})
        name: str = mcfg.get("name", "yolo11n-obb.yaml")
        pretrained: bool = bool(mcfg.get("pretrained", True))
        weights_path: Optional[str] = mcfg.get("weights_path", None)

        # imgsz: preferisci model.input_size, poi dataset.image_size[0], altrimenti 640
        self.imgsz: int = int(
            mcfg.get("input_size")
            or (cfg.get("dataset", {}).get("image_size", [640, 640])[0])
            or 640
        )

        # Carica YOLO wrapper
        if weights_path:
            yolo = YOLO(weights_path)
        else:
            yolo = YOLO(name)  # .pt (pesi) o .yaml (solo architettura)

        # Se voglio partire da zero, resetto i pesi del sotto-modello torch
        if not pretrained:
            print("[Info] Resetting all model weights for training from scratch.")
            for m in yolo.model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

        # Registra SOLO il sotto-modello torch come child module
        self.model = yolo.model  # ultralytics.nn.tasks.BaseModel (nn.Module)
        self._ensure_ultra_hyp()
        # Wrapper YOLO da usare solo in inferenza (NON deve essere un child module)
        object.__setattr__(self, "_infer", yolo)

    # -------------------------- utilità interne --------------------------


    def _ensure_ultra_hyp(self):
        # Converte model.args in SimpleNamespace e inserisce i pesi minimi richiesti dalla loss OBB
        args = getattr(self.model, "args", None)
        if args is None:
            args = SimpleNamespace()
        elif isinstance(args, dict):
            args = SimpleNamespace(**args)

        # Valori di default ragionevoli (come in YOLO detect; per OBB vanno bene per partire)
        defaults = {
            "box": 7.5,              # peso loss box
            "cls": 0.5,              # peso loss classi
            "dfl": 1.5,              # peso Distribution Focal Loss
            "fl_gamma": 0.0,         # gamma Focal (0 = spenta)
            "label_smoothing": 0.0,  # smoothing etichette
            "task": "obb",           # esplicita il task
        }
        for k, v in defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)

        self.model.args = args


    def _stack_images(self, images: Any) -> torch.Tensor:
        """
        Converte list[Tensor CxHxW] -> Tensor BxCxHxW, ridimensionando a self.imgsz.
        Se è già un Tensor BxCxHxW lo restituisce invariato.
        """
        if isinstance(images, (list, tuple)):
            batch: List[torch.Tensor] = []
            for im in images:
                if im.dtype != torch.float32:
                    im = im.float()
                im = F.interpolate(
                    im.unsqueeze(0),
                    size=(self.imgsz, self.imgsz),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                batch.append(im)
            return torch.stack(batch, dim=0)
        assert isinstance(images, torch.Tensor), "images deve essere Tensor o list[Tensor]"
        return images

    @staticmethod
    def _targets_to_batch_dict(
        x: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Costruisce il batch-dict compatibile con la loss OBB di Ultralytics:
        {'img': BxCxHxW, 'cls': (M,1), 'bboxes': (M,5), 'batch_idx': (M,1)}

        Assunzioni:
        - targets[i]['boxes']: (Ni, 8) con 4 punti normalizzati [x1 y1 x2 y2 x3 y3 x4 y4]
        - targets[i]['labels']: (Ni,) con class id interi
        """
        device = x.device
        cls_list, box5_list, idx_list = [], [], []

        for i, t in enumerate(targets):
            boxes8 = t.get("boxes", None)
            labels = t.get("labels", None)
            if boxes8 is None or labels is None or boxes8.numel() == 0:
                continue

            # Garantisce (Ni,8) in float
            boxes8 = boxes8.to(device).float().view(-1, 8)
            labels = labels.to(device).view(-1)

            # Converte 8 vertici -> xywhr (Ni,5), ancora normalizzato, angolo in radianti
            boxes5 = xyxyxyxy2xywhr(boxes8)  # (Ni,5)

            n = boxes5.shape[0]
            cls_list.append(labels.float().view(-1, 1))          # (n,1)
            box5_list.append(boxes5)                              # (n,5)
            idx_list.append(torch.full((n, 1), i, device=device, dtype=torch.float32))  # (n,1)

        if len(box5_list):
            batch = {
                "img": x,  # BxCxHxW
                "cls": torch.cat(cls_list, 0),                   # (M,1)
                "bboxes": torch.cat(box5_list, 0),               # (M,5)  <-- xywhr
                "batch_idx": torch.cat(idx_list, 0),             # (M,1)
            }
        else:
            batch = {
                "img": x,
                "cls": torch.zeros((0, 1), device=device),
                "bboxes": torch.zeros((0, 5), device=device),    # (0,5)
                "batch_idx": torch.zeros((0, 1), device=device), # (0,1)
            }
        return batch

    @staticmethod
    def _xyxy_to_poly8(xyxy: torch.Tensor) -> torch.Tensor:
        """
        Converte box axis-aligned (N,4) in 8 punti (N,8) come rettangoli (fallback).
        """
        x1, y1, x2, y2 = xyxy.unbind(dim=1)
        poly = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=1)  # (N,8)
        return poly

    # ------------------------------- forward -------------------------------

    def forward(
        self,
        images: Any,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        - TRAIN: restituisce {'loss': <tensor scalare>}
        - EVAL/PRED: restituisce list[dict] con 'boxes'(N,8), 'labels'(N,), 'scores'(N,)
        """
        x = self._stack_images(images)

        if self.training:
            if targets is None:
                raise ValueError("targets richiesto in modalità training")

            # 1) Prepara il batch-dict per OBB (bboxes già in xywhr)
            batch = self._targets_to_batch_dict(x, targets)

            # 2) Predizioni grezze dall'head tramite la pipeline interna corretta
            preds_raw = self.model._predict_once(x)  # gestisce correttamente Concat/route ecc.

            # 3) Adatta il formato a quello atteso dalla loss OBB:
            #    la loss fa: feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
            #    quindi serve una SEQUENZA indicizzabile (lista/tupla), non un dict/oggetto.
            preds = preds_raw
            if isinstance(preds_raw, Mapping):  # dict-like
                if "feats" in preds_raw and any(k in preds_raw for k in ("pred_angle", "angle", "theta")):
                    preds = (preds_raw["feats"],
                            preds_raw.get("pred_angle", preds_raw.get("angle", preds_raw.get("theta"))))
                elif "out" in preds_raw and isinstance(preds_raw["out"], (list, tuple)):
                    preds = preds_raw["out"]
                else:
                    preds = tuple(preds_raw.values())
            elif not isinstance(preds_raw, (list, tuple)):
                # Oggetto con attributi?
                if hasattr(preds_raw, "feats") and any(hasattr(preds_raw, k) for k in ("pred_angle", "angle", "theta")):
                    preds = (preds_raw.feats,
                            getattr(preds_raw, "pred_angle",
                                    getattr(preds_raw, "angle", getattr(preds_raw, "theta"))))
                else:
                    # fallback: impacchetta in una tupla
                    preds = (preds_raw, )

            # 4) Calcola la loss con l’API ufficiale (costruisce il criterion se assente)
            out = self.model.loss(preds, batch)  # (loss_totale, loss_items) oppure tensore/vettore

            # 5) Normalizza: torna sempre un tensore scalare + opzionale vettore componenti
            if isinstance(out, (tuple, list)):
                base = out[0]
                loss_items = out[1] if len(out) > 1 else None
            else:
                base = out
                loss_items = None

            if isinstance(base, torch.Tensor) and base.ndim > 0 and base.numel() > 1:
                loss_tensor = base.sum()
                if loss_items is None:
                    loss_items = base.detach()
            else:
                loss_tensor = base if isinstance(base, torch.Tensor) else torch.as_tensor(base, device=x.device, dtype=torch.float32)

            return {"loss": loss_tensor, "loss_items": loss_items}



        # ---- Inferenza / Valutazione ----
        with torch.no_grad():
            results = self._infer(x, verbose=False)

        out: List[Dict[str, torch.Tensor]] = []
        for res in results:
            boxes8 = None
            scores = None
            labels = None

            # Preferenza: OBB (poligoni)
            if hasattr(res, "obb") and res.obb is not None:
                if hasattr(res.obb, "xyxyxyxy"):
                    boxes8 = res.obb.xyxyxyxy  # (N,8) in pixel
                elif hasattr(res.obb, "xyxyxyxyn"):
                    boxes8 = res.obb.xyxyxyxyn
                if hasattr(res.obb, "conf"):
                    scores = res.obb.conf
                if hasattr(res.obb, "cls"):
                    labels = res.obb.cls

            # Fallback: axis-aligned boxes
            if boxes8 is None and hasattr(res, "boxes") and res.boxes is not None:
                if hasattr(res.boxes, "xyxy"):
                    xyxy = res.boxes.xyxy
                    boxes8 = self._xyxy_to_poly8(xyxy)
                if hasattr(res.boxes, "conf") and scores is None:
                    scores = res.boxes.conf
                if hasattr(res.boxes, "cls") and labels is None:
                    labels = res.boxes.cls

            if boxes8 is None:
                boxes8 = torch.zeros((0, 8), device=x.device)
            if scores is None:
                scores = torch.zeros((0,), device=x.device)
            if labels is None:
                labels = torch.zeros((0,), device=x.device)

            out.append(
                {
                    "boxes": boxes8.to(x.device),
                    "scores": scores.to(x.device),
                    "labels": labels.to(x.device).long(),
                }
            )

        return out