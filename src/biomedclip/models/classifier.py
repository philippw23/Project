from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from biomedclip.data.datasets import NUM_CLASSES


class MalignancyMLP(nn.Module):
    """Late-fusion MLP classifier for 3-class bone-tumour malignancy prediction.

    Architecture (late fusion):
        age, sex  →  Linear(2, meta_embed_dim) + ReLU  →  meta_emb
                                                                ↘
        image_emb  ──────────────────────────────────→  cat  →  [Linear+ReLU+Dropout] × N  →  3 logits

    The image embedding (either 512-dim projected or 768-dim pre-projection) is
    concatenated with a small learned projection of the two clinical metadata
    features (age, sex) before being passed through the classifier MLP.

    Output classes:  0 = benign  |  1 = intermediate  |  2 = malignant
    """

    def __init__(
        self,
        embed_dim: int,         # dimensionality of the image embedding (512 or 768)
        hidden_dims: list[int], # widths of the hidden layers, e.g. [256, 128]
        dropout: float,         # dropout probability applied after each hidden ReLU
        meta_embed_dim: int = 32,  # size of the age/sex projection before fusion
    ) -> None:
        super().__init__()

        # Project the two scalar clinical features (age, sex) to a small vector
        # so the MLP can learn a non-linear encoding of them.
        self.meta_proj = nn.Sequential(
            nn.Linear(2, meta_embed_dim),
            nn.ReLU(),
        )

        # Build the classifier stack: Linear → ReLU → Dropout, repeated per hidden layer.
        in_dim = embed_dim + meta_embed_dim  # fused input dimension after concatenation
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, NUM_CLASSES))  # final projection to 3 logits
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        image_emb: torch.Tensor,  # (B, embed_dim) — L2-normalised image embedding
        age: torch.Tensor,         # (B,) — z-scored age
        sex: torch.Tensor,         # (B,) — binary sex indicator (0/1)
    ) -> torch.Tensor:             # (B, 3) — raw class logits
        meta     = torch.stack([age, sex], dim=1)  # (B, 2) — combine clinical scalars
        meta_emb = self.meta_proj(meta)             # (B, meta_embed_dim)
        x        = torch.cat([image_emb, meta_emb], dim=1)  # (B, embed_dim + meta_embed_dim)
        return self.net(x)


def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute L2-normalised image embeddings for an entire data split.

    Runs the frozen encoder once over all batches and stores the results in
    CPU memory.  This avoids re-encoding the same images on every training
    epoch for the validation and test splits, which would otherwise dominate
    runtime for a frozen encoder.

    Returns:
        emb    : (N, embed_dim) — L2-normalised embeddings
        age    : (N,) — z-scored age values
        sex    : (N,) — binary sex indicators
        labels : (N,) — integer class labels
    """
    encoder.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  embedding", leave=False):
            images = batch["image"].to(device)
            # L2-normalise so the downstream MLP receives unit-norm vectors.
            emb    = F.normalize(encoder(images), dim=-1)
            all_emb.append(emb.cpu())   # move to CPU to avoid GPU memory pressure
            all_age.append(batch["age"])
            all_sex.append(batch["sex"])
            all_lbl.append(batch["label"])
    return (
        torch.cat(all_emb),
        torch.cat(all_age),
        torch.cat(all_sex),
        torch.cat(all_lbl),
    )
