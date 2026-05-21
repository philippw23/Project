"""YOLOv5 malignancy classifier — Ultralytics built-in trainer.

Trains yolov5l6u on full CLAHE radiographs (SquarePad → 1024×1024).
Detection classes equal malignancy classes: benign=0, intermediate=1, malignant=2.

Usage:
    python src/yolo/train/train.py [--wandb]
    python src/yolo/train/train.py --epochs 100 --batch 4 --wandb
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA = ROOT_DIR / "data" / "yolo" / "dataset.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "yolo"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 malignancy classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--weights", default="yolov5l6u.pt", help="Pretrained checkpoint")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--name", default="run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="yolo-malignancy")
    parser.add_argument("--wandb_entity", default="philipp-wiese")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    from ultralytics import YOLO

    model = YOLO(args.weights)

    # Default Ultralytics loss weights: box=0.05, cls=0.5, dfl=1.5
    # cls=1.0 may be better for this task — classification is the goal, not localization precision.
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
    )


if __name__ == "__main__":
    main(parse_args())
