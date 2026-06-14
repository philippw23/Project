"""YOLO backbone + multi-task downstream classifier.

The YOLO backbone is used as a frozen (or fine-tunable) feature extractor.
Two heads are trained jointly:
  - BBoxHead   : auxiliary bounding-box regression (smooth-L1 vs. GT derived from masks)
  - MalignancyMLP : primary 3-class malignancy classifier (+ age/sex metadata)

Total loss = L_cls + --lambda_bbox * L_bbox

At test time the classification metrics (balanced accuracy, macro F1, confusion
matrix) are reported together with mean IoU between predicted and GT bboxes.

Usage:
    python src/yolo/train/downstream.py --weights yolov5l6u.pt
    python src/yolo/train/downstream.py --weights yolo26n.pt --finetune
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_fscore_support)
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip.data.datasets import IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES
from biomedclip.data.splits import build_stratified_splits
from biomedclip.loss.classification import build_classification_loss, compute_class_weights
from biomedclip.models.classifier import MalignancyMLP
from biomedclip.utils.misc import DEFAULT_OUT_DIR
from yolo.data.dataset import (YOLODownstreamDataset, YOLOEmbeddingDataset,
                                extract_yolo_embeddings)
from yolo.data.transforms import build_train_transform, build_val_transform
from yolo.models.encoders import BBoxHead, build_encoder

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SPLITS = ROOT_DIR / "data" / "internal_dataset" / "split_clahe.json"


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def batch_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-sample IoU for batches of (cx, cy, w, h) boxes."""
    p = _cxcywh_to_xyxy(pred)
    g = _cxcywh_to_xyxy(gt)

    ix1 = torch.max(p[:, 0], g[:, 0])
    iy1 = torch.max(p[:, 1], g[:, 1])
    ix2 = torch.min(p[:, 2], g[:, 2])
    iy2 = torch.min(p[:, 3], g[:, 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union  = (area_p + area_g - inter).clamp(min=1e-6)
    return inter / union


# ── Train / evaluate (frozen backbone — embeddings pre-computed) ──────────────

def train_one_epoch(
    mlp:       nn.Module,
    bbox_head: nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lambda_bbox: float,
    device:    torch.device,
) -> tuple[float, float]:
    mlp.train()
    bbox_head.train()
    total_cls_loss = total_bbox_loss = 0.0

    for emb, age, sex, labels, bboxes_gt, has_bbox in loader:
        emb      = emb.to(device)
        age      = age.to(device)
        sex      = sex.to(device)
        labels   = labels.to(device)
        bboxes_gt = bboxes_gt.to(device)
        has_bbox  = has_bbox.to(device)

        logits     = mlp(emb, age, sex)
        bbox_pred  = bbox_head(emb)

        loss_cls  = criterion(logits, labels)
        if has_bbox.any():
            loss_bbox = F.smooth_l1_loss(bbox_pred[has_bbox], bboxes_gt[has_bbox])
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        loss = loss_cls + lambda_bbox * loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cls_loss  += loss_cls.item()
        total_bbox_loss += loss_bbox.item()

    n = len(loader)
    return total_cls_loss / n, total_bbox_loss / n


@torch.no_grad()
def evaluate(
    mlp:       nn.Module,
    bbox_head: nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    lambda_bbox: float,
    device:    torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
    """Returns (total_loss, cls_loss, acc, preds, labels, mean_iou)."""
    mlp.eval()
    bbox_head.eval()
    total_cls = total_bbox = 0.0
    all_preds, all_labels = [], []
    all_ious: list[float] = []

    for emb, age, sex, labels, bboxes_gt, has_bbox in loader:
        emb      = emb.to(device)
        age      = age.to(device)
        sex      = sex.to(device)
        labels   = labels.to(device)
        bboxes_gt = bboxes_gt.to(device)
        has_bbox  = has_bbox.to(device)

        logits    = mlp(emb, age, sex)
        bbox_pred = bbox_head(emb)

        total_cls  += criterion(logits, labels).item()
        if has_bbox.any():
            total_bbox += F.smooth_l1_loss(bbox_pred[has_bbox], bboxes_gt[has_bbox]).item()
            ious = batch_iou(bbox_pred[has_bbox], bboxes_gt[has_bbox])
            all_ious.extend(ious.cpu().tolist())

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = float((preds == labels).mean())
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    n         = len(loader)
    total_loss = (total_cls + lambda_bbox * total_bbox) / n
    cls_loss   = total_cls / n
    return total_loss, cls_loss, acc, preds, labels, mean_iou


# ── Fine-tune variants (backbone unfrozen, no pre-computed embeddings) ────────

def train_one_epoch_finetune(
    backbone:  nn.Module,
    mlp:       nn.Module,
    bbox_head: nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lambda_bbox: float,
    device:    torch.device,
) -> tuple[float, float]:
    backbone.train()
    mlp.train()
    bbox_head.train()
    total_cls_loss = total_bbox_loss = 0.0

    for batch in loader:
        images    = batch["image"].to(device)
        age       = batch["age"].to(device)
        sex       = batch["sex"].to(device)
        labels    = batch["label"].to(device)
        bboxes_gt = batch["bbox"].to(device)
        has_bbox  = batch["has_bbox"].to(device)

        emb        = backbone(images)
        logits     = mlp(emb, age, sex)
        bbox_pred  = bbox_head(emb)

        loss_cls  = criterion(logits, labels)
        if has_bbox.any():
            loss_bbox = F.smooth_l1_loss(bbox_pred[has_bbox], bboxes_gt[has_bbox])
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        loss = loss_cls + lambda_bbox * loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cls_loss  += loss_cls.item()
        total_bbox_loss += loss_bbox.item()

    n = len(loader)
    return total_cls_loss / n, total_bbox_loss / n


@torch.no_grad()
def evaluate_finetune(
    backbone:  nn.Module,
    mlp:       nn.Module,
    bbox_head: nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    lambda_bbox: float,
    device:    torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
    backbone.eval()
    mlp.eval()
    bbox_head.eval()
    total_cls = total_bbox = 0.0
    all_preds, all_labels = [], []
    all_ious: list[float] = []

    for batch in loader:
        images    = batch["image"].to(device)
        age       = batch["age"].to(device)
        sex       = batch["sex"].to(device)
        labels    = batch["label"].to(device)
        bboxes_gt = batch["bbox"].to(device)
        has_bbox  = batch["has_bbox"].to(device)

        emb       = backbone(images)
        logits    = mlp(emb, age, sex)
        bbox_pred = bbox_head(emb)

        total_cls += criterion(logits, labels).item()
        if has_bbox.any():
            total_bbox += F.smooth_l1_loss(bbox_pred[has_bbox], bboxes_gt[has_bbox]).item()
            ious = batch_iou(bbox_pred[has_bbox], bboxes_gt[has_bbox])
            all_ious.extend(ious.cpu().tolist())

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = float((preds == labels).mean())
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    n          = len(loader)
    total_loss = (total_cls + lambda_bbox * total_bbox) / n
    cls_loss   = total_cls / n
    return total_loss, cls_loss, acc, preds, labels, mean_iou


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO backbone multi-task downstream classifier."
    )
    parser.add_argument("--weights", default="yolov5l6u.pt",
                        help="Pretrained YOLO checkpoint (yolov5l6u.pt or yolo26n.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (square). Default: 640")
    parser.add_argument("--finetune", action="store_true",
                        help="Unfreeze backbone and fine-tune end-to-end")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Backbone LR when --finetune is set")

    parser.add_argument("--head", default="mlp", choices=["mlp", "mlp_no_meta"])
    parser.add_argument("--splits",  default=str(DEFAULT_SPLITS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr_mlp",        type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int,  default=16)
    parser.add_argument("--hidden_dims",   type=str,   nargs="+", default=["128"])

    parser.add_argument("--lambda_bbox", type=float, default=1.0,
                        help="Weight for the auxiliary bbox regression loss")

    parser.add_argument("--loss", default="ce",
                        choices=["ce", "wce", "focal", "cb_focal", "balanced_softmax", "ldam"])
    parser.add_argument("--class_weighting", default="sqrt",
                        choices=["none", "inverse", "sqrt", "effective"])
    parser.add_argument("--focal_gamma",     type=float, default=2.0)
    parser.add_argument("--cb_beta",         type=float, default=0.99)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--ldam_max_margin", type=float, default=0.5)
    parser.add_argument("--ldam_scale",      type=float, default=30.0)

    parser.add_argument("--overfit_n",     type=int,   default=0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="yolo-downstream")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--sweep",         action="store_true")

    args = parser.parse_args(argv)
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    cfg = wandb.config
    for key in ("head", "lr_mlp", "weight_decay", "dropout", "meta_embed_dim",
                "batch_size", "loss", "class_weighting", "focal_gamma",
                "cb_beta", "lambda_bbox"):
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Weights: {args.weights}  |  imgsz: {args.imgsz}  |  "
          f"finetune: {args.finetune}")

    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb not installed.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )
        if args.sweep:
            _apply_sweep_config(args)

    # ── Backbone + heads ──────────────────────────────────────────────────────
    backbone, embed_dim = build_encoder(args.weights, args.imgsz)
    backbone = backbone.to(device)

    if not args.finetune:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
    else:
        backbone.train()

    use_meta  = args.head != "mlp_no_meta"
    mlp       = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout,
                               args.meta_embed_dim, use_meta=use_meta).to(device)
    bbox_head = BBoxHead(embed_dim).to(device)

    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_heads    = sum(p.numel() for p in mlp.parameters()) + \
                 sum(p.numel() for p in bbox_head.parameters())
    print(f"Backbone params: {n_backbone:,} ({'frozen' if not args.finetune else 'trainable'})  |  "
          f"Head params: {n_heads:,}")

    # ── Splits + datasets ─────────────────────────────────────────────────────
    with open(args.splits, encoding="utf-8") as fh:
        raw_splits = json.load(fh)
    splits = {k: raw_splits[k] for k in ("train", "val", "test")}

    if args.overfit_n > 0:
        sub = splits["train"][:args.overfit_n]
        splits = {"train": sub, "val": sub, "test": sub}

    train_ages = [s["age"] for s in splits["train"]]
    age_mean   = float(np.mean(train_ages))
    age_std    = float(np.std(train_ages))

    tf_train = build_train_transform(args.imgsz)
    tf_val   = build_val_transform(args.imgsz)

    train_ds = YOLODownstreamDataset(splits["train"], age_mean, age_std, tf_train)
    val_ds   = YOLODownstreamDataset(splits["val"],   age_mean, age_std, tf_val)
    test_ds  = YOLODownstreamDataset(splits["test"],  age_mean, age_std, tf_val)
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    pin = device.type == "cuda"
    lkw = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": pin}
    train_loader = DataLoader(train_ds, shuffle=True,  **lkw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **lkw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **lkw)

    # ── Pre-compute embeddings (frozen only) ──────────────────────────────────
    if not args.finetune:
        print("Pre-computing val and test embeddings...")
        emb_lkw = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": pin}

        val_emb_ds  = YOLOEmbeddingDataset(*extract_yolo_embeddings(backbone, val_loader,  device))
        test_emb_ds = YOLOEmbeddingDataset(*extract_yolo_embeddings(backbone, test_loader, device))
        val_emb_loader  = DataLoader(val_emb_ds,  shuffle=False, **emb_lkw)
        test_emb_loader = DataLoader(test_emb_ds, shuffle=False, **emb_lkw)

    # ── Class weighting + loss ─────────────────────────────────────────────────
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"].strip().lower()] for s in train_ds.samples],
        dtype=torch.long,
    )
    label_counts  = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    class_weights = compute_class_weights(label_counts, NUM_CLASSES,
                                           args.class_weighting, args.cb_beta, device)
    criterion = build_classification_loss(args, label_counts, NUM_CLASSES,
                                           class_weights, device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    head_params = list(mlp.parameters()) + list(bbox_head.parameters())
    if args.finetune:
        optimizer = torch.optim.AdamW([
            {"params": backbone.parameters(),  "lr": args.lr_backbone},
            {"params": head_params,            "lr": args.lr_mlp},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr_mlp,
                                       weight_decay=args.weight_decay)

    # ── Output dir ────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{ts}_yolo_{args.head}_{args.loss}"
    if use_wandb and wandb.run:
        run_tag = f"{run_tag}_{wandb.run.id}"
    out_dir   = Path(args.out_dir) / "yolo_downstream" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience}, "
          f"lambda_bbox={args.lambda_bbox})\n")

    for epoch in range(1, args.epochs + 1):
        if args.finetune:
            cls_loss, bbox_loss = train_one_epoch_finetune(
                backbone, mlp, bbox_head, train_loader, optimizer,
                criterion, args.lambda_bbox, device,
            )
            val_loss, val_cls_loss, val_acc, val_preds, val_labels, val_iou = evaluate_finetune(
                backbone, mlp, bbox_head, val_loader,
                criterion, args.lambda_bbox, device,
            )
        else:
            cls_loss, bbox_loss = train_one_epoch(
                mlp, bbox_head, train_loader, optimizer,
                criterion, args.lambda_bbox, device,
            )
            val_loss, val_cls_loss, val_acc, val_preds, val_labels, val_iou = evaluate(
                mlp, bbox_head, val_emb_loader,
                criterion, args.lambda_bbox, device,
            )

        val_bal_acc       = balanced_accuracy_score(val_labels, val_preds)
        val_weighted_prec, val_weighted_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted", zero_division=0
        )
        val_combined_acc = 0.5 * val_acc + 0.5 * val_bal_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_cls={cls_loss:.4f} | train_bbox={bbox_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_bal_acc={val_bal_acc:.3f} | val_iou={val_iou:.3f}"
        )
        if use_wandb:
            wandb.log({
                "train/cls_loss":      cls_loss,
                "train/bbox_loss":     bbox_loss,
                "val/loss":            val_loss,
                "val/cls_loss":        val_cls_loss,
                "val/acc":             val_acc,
                "val/balanced_acc":        val_bal_acc,
                "val/precision_weighted":  val_weighted_prec,
                "val/recall_weighted":     val_weighted_rec,
                "val/combined_acc":        val_combined_acc,
                "val/mean_iou":        val_iou,
            }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "mlp_state_dict":       mlp.state_dict(),
                "bbox_head_state_dict": bbox_head.state_dict(),
                "val_loss":             val_loss,
                "val_bal_acc":          val_bal_acc,
                "args":                 vars(args),
            }
            if args.finetune:
                save_dict["backbone_state_dict"] = backbone.state_dict()
            torch.save(save_dict, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    bbox_head.load_state_dict(ckpt["bbox_head_state_dict"])

    if args.finetune:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        print("Re-computing test embeddings from best checkpoint...")
        test_emb_ds     = YOLOEmbeddingDataset(*extract_yolo_embeddings(backbone, test_loader, device))
        emb_lkw         = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": pin}
        test_emb_loader = DataLoader(test_emb_ds, shuffle=False, **emb_lkw)

    test_loss, test_cls_loss, test_acc, test_preds, test_labels, test_iou = evaluate(
        mlp, bbox_head, test_emb_loader, criterion, args.lambda_bbox, device,
    )

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    test_bal_acc = balanced_accuracy_score(test_labels, test_preds)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}")
    print(f"Balanced accuracy: {test_bal_acc:.3f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.3f}")
    test_weighted_prec, test_weighted_rec, _, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0
    )
    print(f"Weighted Precision: {test_weighted_prec:.3f}  |  Weighted Recall: {test_weighted_rec:.3f}")
    print(f"Mean IoU (bbox): {test_iou:.3f}")
    print()
    print(classification_report(test_labels, test_preds,
                                  labels=list(range(NUM_CLASSES)),
                                  target_names=label_names, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES))),
        index=label_names, columns=label_names,
    ).to_string())

    if use_wandb:
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average="macro", zero_division=0,
        )
        per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, labels=list(range(NUM_CLASSES)), zero_division=0,
        )
        log_dict = {
            "test/loss":            test_loss,
            "test/acc":             test_acc,
            "test/balanced_acc":    test_bal_acc,
            "test/precision_macro":    test_prec,
            "test/recall_macro":       test_rec,
            "test/precision_weighted": test_weighted_prec,
            "test/recall_weighted":    test_weighted_rec,
            "test/f1_macro":           test_f1,
            "test/mean_iou":        test_iou,
        }
        for i, name in enumerate(label_names):
            log_dict[f"test/precision_{name}"] = per_class_prec[i]
            log_dict[f"test/recall_{name}"]    = per_class_rec[i]
            log_dict[f"test/f1_{name}"]        = per_class_f1[i]
        wandb.log(log_dict)
        wandb.finish()

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main(parse_args())
