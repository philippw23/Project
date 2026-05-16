"""Shared zero-shot classification logic for bone-tumour malignancy prediction.

Provides prompt templates and a cosine-similarity evaluation helper used by
both biomedclip_zeroshot.py and scratch_img_text_zeroshot.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from biomedclip.data.datasets import IDX_TO_LABEL, NUM_CLASSES

TEMPLATES = [
    "This is a {} bone tumor radiograph.",
    "A radiograph showing a {} bone tumor.",
    "Radiographic appearance of a {} bone tumor.",
    "This X-ray demonstrates a {} osseous lesion.",
    "Bone radiograph consistent with a {} neoplasm.",
]

# Class names in the same order as LABEL_TO_IDX: benign=0, intermediate=1, malignant=2
CLASSES = ["benign", "intermediate", "malignant"]


def build_class_embeddings(
    encode_text_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenize_fn: Callable[[list[str]], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return (n_classes, embed_dim) L2-normalised class prototype embeddings.

    Each class embedding is the mean of the five template embeddings for that
    class, then L2-normalised — the standard prompt-ensemble approach from CLIP.
    """
    class_embs = []
    for cls_name in CLASSES:
        prompts = [t.format(cls_name) for t in TEMPLATES]
        tokens = tokenize_fn(prompts).to(device)
        with torch.no_grad():
            embs = encode_text_fn(tokens)         # (n_templates, embed_dim)
            embs = F.normalize(embs, dim=-1)
            class_embs.append(embs.mean(dim=0))   # (embed_dim,)
    class_matrix = torch.stack(class_embs)        # (n_classes, embed_dim)
    return F.normalize(class_matrix, dim=-1)


def zeroshot_eval(
    image_embs: torch.Tensor,
    labels: np.ndarray,
    class_matrix: torch.Tensor,
    split_name: str = "test",
) -> dict[str, float]:
    """Cosine-similarity zero-shot evaluation.

    Args:
        image_embs:   (N, embed_dim) L2-normalised image embeddings (CPU or GPU).
        labels:       (N,) integer ground-truth labels matching CLASSES order.
        class_matrix: (n_classes, embed_dim) from build_class_embeddings().
        split_name:   prefix for printed output.

    Returns:
        dict with acc, balanced_acc, f1_macro keys.
    """
    image_embs   = image_embs.to(class_matrix.device)
    sims         = image_embs @ class_matrix.T   # (N, n_classes)
    preds        = sims.argmax(dim=1).cpu().numpy()

    acc      = float((preds == labels).mean())
    bal_acc  = balanced_accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print(f"\n{'=' * 60}")
    print(f"ZERO-SHOT  [{split_name.upper()}]")
    print(f"{'=' * 60}")
    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(f"Macro F1:          {f1_macro:.3f}")
    print()
    print(classification_report(
        labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=label_names,
        digits=3, zero_division=0,
    ))
    import pandas as pd
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))),
        index=label_names, columns=label_names,
    ).to_string())

    return {"acc": acc, "balanced_acc": bal_acc, "f1_macro": f1_macro}
