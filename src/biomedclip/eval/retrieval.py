from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from biomedclip.data.datasets import BoneTumorPairDataset
from biomedclip.loss.contrastive import clip_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean contrastive (InfoNCE) loss over the validation loader."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        texts  = batch["text"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            image_feat = model.encode_image(images)
            text_feat  = model.encode_text(texts)
            loss       = clip_loss(image_feat, text_feat, model.logit_scale)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    val_samples: list,
    preprocess_val,
    tokenizer,
    device: torch.device,
    use_mask: bool,
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute image-text retrieval metrics on the val set.

    Only samples with non-empty reports are used (need paired image+text).
    Metrics are computed per mini-batch and averaged, matching how val loss
    is computed, so the two are directly comparable.
    Returns I2T/T2I Recall@1, Recall@5, and median rank.
    """
    paired = [(s[0], s[1], s[2]) for s in val_samples if s[2] and str(s[2]).strip()]
    if len(paired) < 2:
        return {}

    model.eval()
    ds     = BoneTumorPairDataset(paired, preprocess_val, tokenizer, use_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    i2t_r1_list, i2t_r5_list, i2t_med_list = [], [], []
    t2i_r1_list, t2i_r5_list, t2i_med_list = [], [], []
    n_pairs = 0

    for batch in loader:
        images = batch["image"].to(device)
        texts  = batch["text"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            img_feat = F.normalize(model.encode_image(images), dim=-1)
            txt_feat = F.normalize(model.encode_text(texts),   dim=-1)

        sim = img_feat.float() @ txt_feat.float().T  # (B, B)
        b   = sim.shape[0]
        n_pairs += b

        def _metrics(sim_matrix: torch.Tensor) -> tuple[float, float, float]:
            ranks = np.array([
                int((sim_matrix[i] > sim_matrix[i, i]).sum().item()) + 1
                for i in range(b)
            ])
            return (ranks <= 1).mean(), (ranks <= 5).mean(), float(np.median(ranks))

        i2t_r1, i2t_r5, i2t_med = _metrics(sim)
        t2i_r1, t2i_r5, t2i_med = _metrics(sim.T)

        i2t_r1_list.append(i2t_r1)
        i2t_r5_list.append(i2t_r5)
        i2t_med_list.append(i2t_med)
        t2i_r1_list.append(t2i_r1)
        t2i_r5_list.append(t2i_r5)
        t2i_med_list.append(t2i_med)

    return {
        "retrieval/i2t_r1":          float(np.mean(i2t_r1_list)),
        "retrieval/i2t_r5":          float(np.mean(i2t_r5_list)),
        "retrieval/i2t_median_rank": float(np.mean(i2t_med_list)),
        "retrieval/t2i_r1":          float(np.mean(t2i_r1_list)),
        "retrieval/t2i_r5":          float(np.mean(t2i_r5_list)),
        "retrieval/t2i_median_rank": float(np.mean(t2i_med_list)),
        "retrieval/n_pairs":         float(n_pairs),
        "retrieval/batch_size":      float(batch_size),
    }
