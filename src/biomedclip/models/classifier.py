from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from biomedclip.data.datasets import NUM_CLASSES


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
        meta     = torch.stack([age, sex], dim=1)        # (B, 2)
        meta_emb = self.meta_proj(meta)                  # (B, meta_embed_dim)
        x        = torch.cat([image_emb, meta_emb], dim=1)
        return self.net(x)


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
            emb    = F.normalize(encoder(images), dim=-1)
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
