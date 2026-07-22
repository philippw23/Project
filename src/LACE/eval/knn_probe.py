from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from LACE.models.encoders import SharedViT


@torch.no_grad()
def _embed_cls(vit: SharedViT, loader: DataLoader, device: torch.device):
    """Return (embeddings [N, D] float32, labels [N] int) for a loader.

    Image-only CLS embeddings via forward_cls; no text encoding, no gradients.
    Labels are read from the batch (threaded through InternalDatasetV2), so the
    embedding-label correspondence is guaranteed regardless of loader ordering.
    """
    embs, labels = [], []
    for batch in loader:
        img = batch.get("input_image", batch.get("crop_image")).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            z = vit.forward_cls(img)                     # [B, D]
        embs.append(z.float().cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(embs).numpy(), torch.cat(labels).numpy()


def _loo_knn_predict(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out cosine k-NN predictions over the whole pool.

    Queries the (k+1) nearest neighbours of every point and drops the first
    (the point itself, at cosine distance 0), so no point is ever its own
    neighbour. Uniform majority vote over the remaining k, ties broken by
    lowest class index.
    """
    classes = np.unique(y)
    knn = KNeighborsClassifier(
        n_neighbors=k + 1, metric="cosine", algorithm="brute", weights="uniform",
    ).fit(X, y)
    neigh_idx    = knn.kneighbors(X, return_distance=False)[:, 1:]   # [N, k], self dropped
    neigh_labels = y[neigh_idx]                                      # [N, k]
    votes = np.stack([(neigh_labels == c).sum(axis=1) for c in classes], axis=1)
    return classes[votes.argmax(axis=1)]


@torch.no_grad()
def evaluate_knn_probe(
    vit: SharedViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    ks: tuple[int, ...] = (5, 20),
) -> dict[str, float]:
    """Cheap downstream-aligned proxy: leave-one-out cosine k-NN on malignancy.

    Pools frozen train + val CLS embeddings and scores a uniform-vote LOO k-NN
    classifier over the whole pool with balanced accuracy + macro F1, for each k.
    Malignancy labels never enter pretraining, so pooling the (encoder-seen) train
    images with held-out val is the standard SSL k-NN protocol, not leakage — it
    just buys a much larger, lower-variance evaluation set (~train+val points).
    Test is deliberately excluded (it stays the downstream ground truth).

    Log-only proxy for checkpoint quality — it does not drive selection.
    train_loader should be a no-augmentation, shuffle=False loader over the
    pretraining train samples so the bank is a clean function of the encoder.
    """
    vit.eval()

    X_train, y_train = _embed_cls(vit, train_loader, device)
    X_val,   y_val   = _embed_cls(vit, val_loader, device)
    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    if len(X) < max(ks) + 1:
        return {}

    metrics: dict[str, float] = {
        "knn/n_train": float(len(X_train)),
        "knn/n_val":   float(len(X_val)),
        "knn/n_pool":  float(len(X)),
    }
    for k in ks:
        preds = _loo_knn_predict(X, y, k)
        metrics[f"knn/bal_acc_k{k}"] = float(balanced_accuracy_score(y, preds))
        metrics[f"knn/f1_k{k}"]      = float(
            f1_score(y, preds, average="macro", labels=np.unique(y))
        )
    return metrics
