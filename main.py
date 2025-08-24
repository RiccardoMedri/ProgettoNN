# main.py
import argparse
import json

from scripts.train import train_ultra
from scripts.evaluate import evaluate_ultra
from scripts.predict import predict_ultra


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline Ultralytics YOLO (OBB o standard): train, evaluate, predict"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Avvia il training (Ultralytics YOLO)")
    t.add_argument("--config", "-c", required=True, help="Path a config.json")

    # --- evaluate ---
    e = sub.add_parser("evaluate", help="Valuta un checkpoint (.pt) su uno split")
    e.add_argument("--config", "-c", required=True, help="Path a config.json")
    e.add_argument("--weights", "-w", default=None, help="Path a checkpoint .pt (opzionale)")
    e.add_argument(
        "--split", default="val", choices=["val", "test"], help="Split da usare"
    )
    e.add_argument(
        "--output", "-o", default=None, help="File JSON dove salvare le metriche (opzionale)"
    )

    # --- predict ---
    p2 = sub.add_parser("predict", help="Esegui inferenza su file/cartella/video")
    p2.add_argument("--config", "-c", required=True, help="Path a config.json")
    p2.add_argument(
        "--source", "-s", required=True, help="File/dir/video/camera index (come in Ultralytics)"
    )
    p2.add_argument("--weights", "-w", default=None, help="Path a checkpoint .pt (opzionale)")
    p2.add_argument("--conf", type=float, default=None, help="Confidence threshold (opz.)")
    p2.add_argument("--iou", type=float, default=None, help="IoU/NMS threshold (opz.)")
    p2.add_argument("--save_txt", action="store_true", help="Salva anche label .txt")
    p2.add_argument("--save_conf", action="store_true", help="Salva conf nei .txt")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        ckpt = train_ultra(args.config)
        print(f"[Train] Best checkpoint: {ckpt}")

    elif args.mode == "evaluate":
        metrics = evaluate_ultra(args.config, weights_path=args.weights, split=args.split)
        print(json.dumps(metrics, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[Eval] Metrics salvate in: {args.output}")

    elif args.mode == "predict":
        out = predict_ultra(
            config_path=args.config,
            source=args.source,
            weights_path=args.weights,
            conf=args.conf,
            iou=args.iou,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save=True,
        )
        print(f"[Predict] save_dir: {out.get('save_dir')}")
        if out.get("saved"):
            print("[Predict] files:")
            for pth in out["saved"]:
                print("-", pth)


if __name__ == "__main__":
    main()