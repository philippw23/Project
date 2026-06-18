from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from biomedclip.data.datasets import NUM_CLASSES


class LinearHead(nn.Module):
    """Single linear layer — no metadata, no non-linearities (true linear probe)."""

    def __init__(self, embed_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        image_emb: torch.Tensor,
        age: torch.Tensor | None = None,
        sex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fc(image_emb)


class MalignancyMLP(nn.Module):
    """Non-linear MLP classifier for 3-class bone-tumour malignancy prediction.

    When use_meta=True (default), age and sex are projected and fused with the
    image embedding before the MLP stack (late fusion).  When use_meta=False,
    only the image embedding is used — same non-linear capacity, no clinical info.

    Output classes:  0 = benign  |  1 = intermediate  |  2 = malignant
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        dropout: float,
        meta_embed_dim: int = 32,
        use_meta: bool = True,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.use_meta = use_meta

        if use_meta:
            self.meta_proj = nn.Sequential(nn.Linear(2, meta_embed_dim), nn.ReLU())
            in_dim = embed_dim + meta_embed_dim
        else:
            self.meta_proj = None
            in_dim = embed_dim

        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        image_emb: torch.Tensor,
        age: torch.Tensor | None = None,
        sex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_meta:
            meta     = torch.stack([age, sex], dim=1)
            meta_emb = self.meta_proj(meta)
            x        = torch.cat([image_emb, meta_emb], dim=1)
        else:
            x = image_emb
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
