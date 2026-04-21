"""Downstream malignancy classification using the pretrained BiomedCLIP image encoder.

Loads the LoRA-adapted image encoder from a checkpoint produced by
biomedclip_pretrain.py, freezes it, and trains a small MLP classifier on top.

MLP input : [image_embedding (512) | age (1, z-scored) | sex (1, binary)]
MLP output: 3-class logits  (benign=0 / intermediate=1 / malignant=2)

Usage:
    python src/biomedclip_downstream.py \\
        --checkpoint results/biomedclip_pretrain/best_checkpoint.pt \\
        --splits     results/biomedclip_pretrain/splits.json \\
        --excel      data/metadata.xlsx \\
        --use_mask   --epochs 50
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from biomedclip_pretrain import (
    build_train_transform,
    crop_around_mask,
    inject_lora,
    MODEL_TAG,
    DEFAULT_EXCEL,
    DEFAULT_OUT_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = ROOT_DIR / "results" / "biomedclip_pretrain" / "splits.json"
DEFAULT_CHECKPOINT = ROOT_DIR / "results" / "biomedclip_pretrain" / "best_checkpoint.pt"

LABEL_TO_IDX = {"benign": 0, "intermediate": 1, "malignant": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Metadata loading (age + sex from Excel)
# ---------------------------------------------------------------------------

def load_age_sex_lookup(excel_path: Path) -> dict[str, tuple[float, float]]:
    """Return {filename_stem: (age_float, sex_binary)} from the Excel.

    sex: m/male -> 1.0,  f/female -> 0.0.
    Rows with missing or unparseable age/sex are skipped with a warning.
    """
    df = pd.read_excel(
        excel_path,
        sheet_name="internal_data_matched",
        usecols=[0, 4, 5],       # A=filename, E=age, F=sex
        skiprows=1,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.columns = ["filename", "age", "sex"]
    df = df.dropna(subset=["filename"])
    df["filename"] = df["filename"].str.strip()

    lookup: dict[str, tuple[float, float]] = {}
    skipped = 0
    for _, row in df.iterrows():
        stem = Path(row["filename"]).stem
        try:
            age = float(row["age"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        sex_raw = str(row["sex"]).strip().lower()
        if sex_raw in ("m", "male", "1"):
            sex = 1.0
        elif sex_raw in ("f", "female", "0"):
            sex = 0.0
        else:
            skipped += 1
            continue
        lookup[stem] = (age, sex)

    if skipped:
        warnings.warn(f"Skipped {skipped} rows with missing/invalid age or sex.")
    return lookup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DownstreamDataset(Dataset):
    """Returns (image_tensor, age_norm, sex, label_idx) for each sample."""

    def __init__(
        self,
        samples: list[dict],
        age_sex_lookup: dict[str, tuple[float, float]],
        age_mean: float,
        age_std: float,
        preprocess,
        use_mask: bool,
    ) -> None:
        valid = []
        for s in samples:
            stem = Path(s["image"]).stem
            if stem not in age_sex_lookup:
                continue
            if s["label"] not in LABEL_TO_IDX:
                continue
            valid.append(s)
        self.samples = valid
        self.age_sex_lookup = age_sex_lookup
        self.age_mean = age_mean
        self.age_std = age_std
        self.preprocess = preprocess
        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        stem = Path(s["image"]).stem
        age_raw, sex = self.age_sex_lookup[stem]

        image = Image.open(s["image"]).convert("RGB")
        mask_path = Path(s["mask"])
        if self.use_mask and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped = crop_around_mask(image_arr, mask_arr)
            image = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image_tensor = self.preprocess(image)
        age_norm = (age_raw - self.age_mean) / (self.age_std + 1e-6)
        label = LABEL_TO_IDX[s["label"]]

        return {
            "image": image_tensor,
            "age": torch.tensor(age_norm, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class MalignancyMLP(nn.Module):
    """Late-fusion MLP: meta features (age, sex) are projected to meta_embed_dim
    before being concatenated with the image embedding.

        age, sex  →  meta_proj  →  meta_emb (meta_embed_dim)
                                              ↘
        image_emb  ──────────────────────→  cat  →  classifier MLP  →  3 classes
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        dropout: float,
        meta_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.meta_proj = nn.Sequential(
            nn.Linear(2, meta_embed_dim),
            nn.ReLU(),
        )
        in_dim = embed_dim + meta_embed_dim
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        image_emb: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        meta = torch.stack([age, sex], dim=1)          # (B, 2)
        meta_emb = self.meta_proj(meta)                # (B, meta_embed_dim)
        x = torch.cat([image_emb, meta_emb], dim=1)   # (B, embed_dim + meta_embed_dim)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute image embeddings for a full split (avoids re-encoding each epoch)."""
    encoder.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  embedding", leave=False):
            images = batch["image"].to(device)
            emb = F.normalize(encoder(images), dim=-1)
            all_emb.append(emb.cpu())
            all_age.append(batch["age"])
            all_sex.append(batch["sex"])
            all_lbl.append(batch["label"])
    return (
        torch.cat(all_emb),
        torch.cat(all_age),
        torch.cat(all_sex),
        torch.cat(all_lbl),
    )


class EmbeddingDataset(Dataset):
    def __init__(self, emb, age, sex, lbl):
        self.emb = emb
        self.age = age
        self.sex = sex
        self.lbl = lbl

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        return self.emb[idx], self.age[idx], self.sex[idx], self.lbl[idx]


def train_one_epoch(
    mlp: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch. Encoder runs in eval mode (frozen) but re-encodes each batch
    so that preprocess_train augmentations are applied fresh every epoch."""
    mlp.train()
    encoder.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        age    = batch["age"].to(device)
        sex    = batch["sex"].to(device)
        lbl    = batch["label"].to(device)
        with torch.no_grad():
            emb = F.normalize(encoder(images), dim=-1)
        optimizer.zero_grad()
        logits = mlp(emb, age, sex)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    mlp: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    mlp.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for emb, age, sex, lbl in loader:
        emb, age, sex, lbl = emb.to(device), age.to(device), sex.to(device), lbl.to(device)
        logits = mlp(emb, age, sex)
        total_loss += criterion(logits, lbl).item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(lbl.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    return total_loss / len(loader), float(acc), preds, labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream malignancy classifier on top of BiomedCLIP image encoder."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--excel", default=str(DEFAULT_EXCEL))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--lora_layers", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--meta_embed_dim", type=int, default=16,
                        help="Projection dim for age/sex before fusion (default: %(default)s)")
    parser.add_argument("--hidden_dims", type=str, nargs="+", default=[256, 128])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="biomedclip-downstream")
    parser.add_argument("--wandb_run", default=None)
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B team/entity name (default: personal account)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run as wandb sweep agent (hyperparams come from wandb.config)")
    args = parser.parse_args(argv)
    # Normalize hidden_dims: handles plain CLI ("256 128") and wandb format ("[256, 128]")
    raw = " ".join(str(x) for x in args.hidden_dims)
    args.hidden_dims = [int(x) for x in raw.strip("[]").replace(",", " ").split()]
    return args


def _apply_sweep_config(args: argparse.Namespace) -> None:
    """Overwrite args with values from wandb.config when running as sweep agent."""
    cfg = wandb.config
    for key in ("lr", "dropout", "meta_embed_dim", "weight_decay"):
        if key in cfg:
            setattr(args, key, cfg[key])
    if "hidden_dims" in cfg:
        args.hidden_dims = list(cfg["hidden_dims"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- W&B (must run before building MLP so sweep config overrides args) ---
    use_wandb = (args.wandb or args.sweep) and WANDB_AVAILABLE
    if (args.wandb or args.sweep) and not WANDB_AVAILABLE:
        warnings.warn("--wandb/--sweep set but wandb is not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config={k: v for k, v in vars(args).items()},
        )
        if args.sweep:
            _apply_sweep_config(args)

    # --- Load model and checkpoint ---
    print(f"Loading model: {MODEL_TAG}")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_TAG)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Use LoRA config from checkpoint if present, otherwise fall back to CLI args
    lora_cfg = ckpt.get("lora_config") or {}
    lora_layers = lora_cfg.get("lora_layers", args.lora_layers)
    lora_r      = lora_cfg.get("lora_r",      args.lora_r)
    lora_alpha  = lora_cfg.get("lora_alpha",   args.lora_alpha)
    print(f"LoRA config from checkpoint: layers={lora_layers}, r={lora_r}, alpha={lora_alpha}")

    inject_lora(model, lora_layers, lora_r, lora_alpha)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    encoder = model.visual.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    preprocess_train = build_train_transform(preprocess_val)

    # --- Load splits and age/sex metadata ---
    with open(args.splits, encoding="utf-8") as fh:
        splits = json.load(fh)

    age_sex_lookup = load_age_sex_lookup(Path(args.excel))

    # --- Diagnostic: check stem alignment ---
    lookup_keys = list(age_sex_lookup.keys())
    split_stems = [Path(s["image"]).stem for s in splits["downstream_train"][:5]]
    print(f"Lookup sample keys : {lookup_keys[:5]}")
    print(f"Split sample stems : {split_stems}")
    n_matches = sum(1 for s in splits["downstream_train"] if Path(s["image"]).stem in age_sex_lookup)
    print(f"Matching train samples: {n_matches} / {len(splits['downstream_train'])}")

    if n_matches == 0:
        raise RuntimeError(
            "No age/sex entries matched any split sample. "
            "Check that Excel column A filenames match the image stems in splits.json.\n"
            f"  Lookup example: {lookup_keys[:3]}\n"
            f"  Split example:  {split_stems[:3]}"
        )

    # Compute age stats from training split only
    train_ages = [
        age_sex_lookup[Path(s["image"]).stem][0]
        for s in splits["downstream_train"]
        if Path(s["image"]).stem in age_sex_lookup
    ]
    age_mean = float(np.mean(train_ages))
    age_std = float(np.std(train_ages))
    print(f"Age stats (train): mean={age_mean:.1f}, std={age_std:.1f}")

    # --- Build datasets ---
    train_ds = DownstreamDataset(
        splits["downstream_train"], age_sex_lookup, age_mean, age_std,
        preprocess_train, args.use_mask,
    )
    val_ds = DownstreamDataset(
        splits["downstream_val"], age_sex_lookup, age_mean, age_std,
        preprocess_val, args.use_mask,
    )
    test_ds = DownstreamDataset(
        splits["test"], age_sex_lookup, age_mean, age_std,
        preprocess_val, args.use_mask,
    )
    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    use_pin = device.type == "cuda"
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": use_pin}

    # Train loader uses raw images so preprocess_train augmentation runs each epoch
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader_raw  = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader_raw = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # Pre-compute val/test embeddings once (no augmentation needed)
    print("Pre-computing val/test embeddings...")
    val_emb,  val_age,  val_sex,  val_lbl  = extract_embeddings(encoder, val_loader_raw,  device)
    test_emb, test_age, test_sex, test_lbl = extract_embeddings(encoder, test_loader_raw, device)

    embed_dim = val_emb.shape[1]
    print(f"Embedding dim: {embed_dim}")

    # Class weights to handle imbalance (compute from train split labels)
    train_labels_all = torch.tensor(
        [LABEL_TO_IDX[s["label"]] for s in train_ds.samples], dtype=torch.long
    )
    label_counts = torch.bincount(train_labels_all, minlength=NUM_CLASSES).float()
    # class_weights = (label_counts.sum() / (NUM_CLASSES * label_counts)).to(device)
    print(f"Class counts (train): { {IDX_TO_LABEL[i]: int(label_counts[i]) for i in range(NUM_CLASSES)} }")

    val_emb_ds   = EmbeddingDataset(val_emb,   val_age,   val_sex,   val_lbl)
    test_emb_ds  = EmbeddingDataset(test_emb,  test_age,  test_sex,  test_lbl)

    val_loader   = DataLoader(val_emb_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_emb_ds, batch_size=args.batch_size, shuffle=False)

    # --- MLP ---
    mlp = MalignancyMLP(embed_dim, args.hidden_dims, args.dropout, args.meta_embed_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Output dir ---
    out_dir = Path(args.out_dir) / "biomedclip_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unique checkpoint per sweep run to avoid cross-run overwrites
    run_id = (wandb.run.id if use_wandb and wandb.run else None) or "local"
    ckpt_path = out_dir / f"best_mlp_{run_id}.pt"

    # --- Training loop ---
    best_val_loss = float("inf")
    print(f"\nTraining MLP for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(mlp, encoder, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(mlp, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )
        if use_wandb:
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "val/acc": val_acc}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "mlp_state_dict": mlp.state_dict(), "val_loss": val_loss},
                       ckpt_path)

    # --- Final evaluation on test set ---
    mlp.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["mlp_state_dict"])
    test_loss, test_acc, test_preds, test_labels = evaluate(mlp, test_loader, criterion, device)

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(test_labels, test_preds),
        index=label_names, columns=label_names,
    ).to_string())

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.finish()

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main(parse_args())
