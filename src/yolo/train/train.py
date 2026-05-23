"""YOLO malignancy classifier — Ultralytics built-in trainer.

Trains a pretrained YOLO model on the bone-tumour dataset.  Detection classes
equal malignancy classes: benign=0, intermediate=1, malignant=2.

After training the best checkpoint is evaluated on the held-out test split and
the same classification metrics used across all baselines are printed.

Usage:
    python src/yolo/train/train.py [--wandb]
    python src/yolo/train/train.py --epochs 100 --batch 4 --wandb
    python src/yolo/train/train.py --print_architecture
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA    = ROOT_DIR / "data" / "yolo" / "dataset.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "yolo"
CLASS_NAMES     = ["benign", "intermediate", "malignant"]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO malignancy classifier.")
    parser.add_argument("--data",      default=str(DEFAULT_DATA))
    parser.add_argument("--weights",   default="yolov5l6u.pt", help="Pretrained checkpoint")
    parser.add_argument("--imgsz",     type=int,   default=1024)
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--lr0",       type=float, default=0.01)
    parser.add_argument("--patience",  type=int,   default=20, help="Early-stopping patience (epochs)")
    parser.add_argument("--workers",   type=int,   default=4)
    parser.add_argument("--device",    default="0")
    parser.add_argument("--project",   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--name",      default="run")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="yolo-malignancy")
    parser.add_argument("--wandb_entity",  default="philipp-wiese")
    parser.add_argument("--print_architecture", action="store_true",
                        help="Print YOLO architecture summary and exit.")
    return parser.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_gt(labels_dir: Path) -> dict[str, dict]:
    """Return {stem: {cls, bbox}} from YOLO-format label files."""
    gt: dict[str, dict] = {}
    for f in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if lines:
            parts = lines[0].split()
            gt[f.stem] = {
                "cls":  int(parts[0]),
                "bbox": [float(x) for x in parts[1:5]] if len(parts) >= 5 else None,
            }
    return gt


def _iou_cxcywh(b1: list[float], b2: list[float]) -> float:
    """IoU between two normalised (cx, cy, w, h) boxes."""
    x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    inter = max(0.0, min(x2a, x2b) - max(x1a, x1b)) * \
            max(0.0, min(y2a, y2b) - max(y1a, y1b))
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / max(union, 1e-6)


def _make_grid(img_paths: list[Path], out_path: Path, thumb: int = 320) -> None:
    from PIL import Image
    n    = len(img_paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * thumb, rows * thumb), color=(30, 30, 30))
    for i, p in enumerate(img_paths):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb, thumb))
        x = (i % cols) * thumb + (thumb - img.width)  // 2
        y = (i // cols) * thumb + (thumb - img.height) // 2
        grid.paste(img, (x, y))
    grid.save(out_path)


# ── Test evaluation ───────────────────────────────────────────────────────────

def evaluate_test_set(
    best_weights: Path,
    data_yaml:    Path,
    out_dir:      Path,
    imgsz:        int,
    device:       str,
) -> None:
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                  confusion_matrix, f1_score)
    from ultralytics import YOLO

    with open(data_yaml, encoding="utf-8") as fh:
        data_cfg = yaml.safe_load(fh)

    data_root    = Path(data_cfg.get("path", str(data_yaml.parent)))
    test_img_dir = data_root / data_cfg["test"]
    test_lbl_dir = data_root / "labels" / "test"

    gt_map = _read_gt(test_lbl_dir)
    if not gt_map:
        print("No test labels found — skipping evaluation.")
        return

    model = YOLO(str(best_weights))

    # ── Prediction grid ───────────────────────────────────────────────────────
    tmp_dir = out_dir / "_pred_tmp"
    model.predict(
        source=str(test_img_dir),
        imgsz=imgsz,
        conf=0.01,
        device=device,
        save=True,
        project=str(tmp_dir.parent),
        name=tmp_dir.name,
        verbose=False,
        exist_ok=True,
    )
    pred_imgs = sorted(tmp_dir.glob("*.jpg")) + sorted(tmp_dir.glob("*.png"))
    if pred_imgs:
        grid_path = out_dir / "test_predictions.jpg"
        _make_grid(pred_imgs, grid_path)
        print(f"Test prediction grid → {grid_path}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = model.predict(
        source=str(test_img_dir),
        imgsz=imgsz,
        conf=0.001,
        device=device,
        verbose=False,
        save=False,
    )

    y_true: list[int]   = []
    y_pred: list[int]   = []
    ious:   list[float] = []
    no_det_stems: list[str] = []

    for r in results:
        stem = Path(r.path).stem
        if stem not in gt_map:
            continue
        entry = gt_map[stem]
        y_true.append(entry["cls"])

        if r.boxes is not None and len(r.boxes) > 0:
            best_idx  = int(r.boxes.conf.argmax().item())
            y_pred.append(int(r.boxes.cls[best_idx].item()))
            # IoU between highest-conf predicted box and GT box
            if entry["bbox"] is not None:
                pred_box = r.boxes.xywhn[best_idx].cpu().tolist()
                ious.append(_iou_cxcywh(pred_box, entry["bbox"]))
        else:
            y_pred.append(-1)
            no_det_stems.append(stem)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    no_det_mask = y_pred_arr == -1
    n_no_det    = int(no_det_mask.sum())

    y_true_det = y_true_arr[~no_det_mask]
    y_pred_det = y_pred_arr[~no_det_mask]

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if n_no_det > 0:
        print(f"No detection: {n_no_det} / {len(y_true)} images")
        for s in no_det_stems:
            print(f"  {s}  (true: {CLASS_NAMES[gt_map[s]['cls']]})")
        print()

    if len(y_true_det) == 0:
        print("No detections at all — cannot compute metrics.")
        return

    acc      = float((y_true_det == y_pred_det).mean())
    bal_acc  = balanced_accuracy_score(y_true_det, y_pred_det)
    macro_f1 = f1_score(y_true_det, y_pred_det, average="macro", zero_division=0)
    mean_iou = float(np.mean(ious)) if ious else float("nan")

    n_eval = len(y_true_det)
    print(f"Accuracy:          {acc:.3f}  (on {n_eval} / {len(y_true)} images with detections)")
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(f"Macro F1:          {macro_f1:.3f}")
    print(f"Mean IoU (bbox):   {mean_iou:.3f}")
    print()

    print(classification_report(
        y_true_det, y_pred_det,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))

    cm    = confusion_matrix(y_true_det, y_pred_det, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if args.print_architecture:
        from yolo.utils.misc import print_yolo_architecture
        print_yolo_architecture(args.weights, args.imgsz)
        return

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_ENTITY",  args.wandb_entity)

    from ultralytics import YOLO

    def _print_val_loss(trainer) -> None:
        v    = getattr(trainer, "validator", None)
        loss = getattr(v, "loss", None) if v else None
        if loss is None:
            return
        items = loss.cpu().tolist() if hasattr(loss, "cpu") else list(loss)
        if len(items) >= 3:
            print(f"  Val  — box: {items[0]:.4f}  cls: {items[1]:.4f}  dfl: {items[2]:.4f}")

    model = YOLO(args.weights)
    model.add_callback("on_fit_epoch_end", _print_val_loss)

    # plots=False suppresses all saved images/curves during training (train_batch*.jpg,
    # val_batch*.jpg, PR curves, confusion_matrix.png, results.png, labels.jpg).
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
        plots=False,
        # Classification-focused loss weights: boost cls relative to box default (7.5).
        cls=3.0,
        box=2.0,
        # X-ray specific: disable mosaic (single-instance images) and colour jitter
        # (grayscale, hue/saturation meaningless).
        mosaic=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
    )

    out_dir      = Path(args.project) / args.name
    best_weights = out_dir / "weights" / "best.pt"
    if best_weights.exists():
        evaluate_test_set(best_weights, Path(args.data), out_dir, args.imgsz, args.device)
    else:
        print(f"best.pt not found at {best_weights} — skipping test evaluation.")


if __name__ == "__main__":
    main(parse_args())
