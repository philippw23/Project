from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from LACE.models.encoders import BiomedCLIPTextEncoder, SharedViT


@torch.no_grad()
def evaluate_retrieval_lace(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    val_loader: DataLoader,
    device: torch.device,
    text_mode: str = "phrase_attn",
) -> dict[str, float]:
    """Image-to-text and text-to-image retrieval on the full val set.

    Embeds every val sample once, builds the full NxN similarity matrix,
    and computes R@1, R@5 and median rank for both directions.
    Only samples with at least one valid beurteilung phrase are included.
    """
    vit.eval()
    text_enc.eval()

    img_embs, txt_embs = [], []

    for batch in val_loader:
        full_img = batch["full_image"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            z_img = vit.forward_cls(full_img)                        # [B, D]

            if text_mode in ("phrase_mean", "phrase_attn"):
                pids   = batch["beur_phrase_ids"].to(device)
                pattn  = batch["beur_phrase_attn"].to(device)
                pmask  = batch["beur_phrase_mask"].to(device)
                # skip samples with no valid phrase
                has_phrase = pmask.any(dim=1)                        # [B]
                if not has_phrase.any():
                    continue
                embs = text_enc._encode_phrase_batch(pids, pattn, pmask)  # [B, J, D]
                n_real = pmask.float().sum(dim=1, keepdim=True).clamp(min=1)
                z_txt = F.normalize(
                    (embs * pmask.float().unsqueeze(-1)).sum(dim=1) / n_real, dim=-1
                )                                                    # [B, D]
                z_img = z_img[has_phrase]
                z_txt = z_txt[has_phrase]
            else:
                ids  = batch["beurteilung_ids"].to(device)
                mask = batch["beurteilung_mask"].to(device)
                z_txt = text_enc.encode_beurteilung(ids, mask)       # [B, D]

        img_embs.append(F.normalize(z_img.float(), dim=-1))
        txt_embs.append(F.normalize(z_txt.float(), dim=-1))

    if sum(e.shape[0] for e in img_embs) < 2:
        return {}

    imgs = torch.cat(img_embs, dim=0)   # [N, D]
    txts = torch.cat(txt_embs, dim=0)   # [N, D]
    N    = imgs.shape[0]

    sim = imgs @ txts.T                 # [N, N]

    def _recall(sim_matrix: torch.Tensor) -> tuple[float, float, float]:
        ranks = np.array([
            int((sim_matrix[i] > sim_matrix[i, i]).sum().item()) + 1
            for i in range(N)
        ])
        return (ranks <= 1).mean(), (ranks <= 5).mean(), float(np.median(ranks))

    i2t_r1, i2t_r5, i2t_med = _recall(sim)
    t2i_r1, t2i_r5, t2i_med = _recall(sim.T)

    return {
        "retrieval/i2t_r1":          float(i2t_r1),
        "retrieval/i2t_r5":          float(i2t_r5),
        "retrieval/i2t_median_rank": float(i2t_med),
        "retrieval/t2i_r1":          float(t2i_r1),
        "retrieval/t2i_r5":          float(t2i_r5),
        "retrieval/t2i_median_rank": float(t2i_med),
        "retrieval/mean_r1":         float((i2t_r1 + t2i_r1) / 2),
        "retrieval/n_pairs":         float(N),
    }
