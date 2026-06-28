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
    text_mode: str = "phrase",
) -> dict[str, float]:
    """Image-to-text and text-to-image retrieval on the full val set.

    Embeds every val sample once, builds the full NxN similarity matrix,
    and computes R@1, R@5 and median rank for both directions.
    In phrase modes the text anchor is the concatenated befund+beurteilung
    phrases encoded as a single sequence; in full/concat modes the
    beurteilung text is used.
    """
    vit.eval()
    text_enc.eval()

    img_embs, txt_embs = [], []

    for batch in val_loader:
        img = batch.get("full_image", batch.get("crop_image")).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            z_img = vit.forward_cls(img)                        # [B, D]

            if text_mode == "mixed":
                z_txt = text_enc.encode_beurteilung(
                    batch["concat_full_ids"].to(device),
                    batch["concat_full_mask"].to(device),
                )
            elif text_mode == "phrase":
                z_txt = text_enc.encode_beurteilung(
                    batch["concat_phrase_ids"].to(device),
                    batch["concat_phrase_mask"].to(device),
                )
            else:
                z_txt = text_enc.encode_beurteilung(
                    batch["beurteilung_ids"].to(device),
                    batch["beurteilung_mask"].to(device),
                )                                                    # [B, D]

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
