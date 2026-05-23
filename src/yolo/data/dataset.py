"""Dataset and embedding helpers for the YOLO downstream classifier."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from biomedclip.data.datasets import IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES
from yolo.data.bbox_from_mask import bboxes_from_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODownstreamDataset(Dataset):
    """Returns image + bbox + age/sex + label for each sample.

    The bounding box is derived from the lesion mask (same coordinate space as
    bboxes_from_mask: normalised to the SquarePad-extended side).  When a mask
    is missing or has no foreground, has_bbox is False and bbox is zeros.

    When a mask contains multiple connected components the one with the largest
    area is used.
    """

    def __init__(
        self,
        samples: list[dict],
        age_mean: float,
        age_std: float,
        transform,
    ) -> None:
        valid = [
            s for s in samples
            if (s.get("label") or "").strip().lower() in LABEL_TO_IDX
        ]
        self.samples  = valid
        self.age_mean = age_mean
        self.age_std  = age_std
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s          = self.samples[idx]
        img_path   = Path(s["image"])
        mask_path  = Path(s["mask"])
        label_str  = s["label"].strip().lower()
        class_id   = LABEL_TO_IDX[label_str]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = bboxes_from_mask(mask_path, class_id, w, h)
        if boxes:
            # Pick the box with the largest area
            _, cx, cy, bw, bh = max(boxes, key=lambda b: b[3] * b[4])
            bbox     = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
            has_bbox = True
        else:
            bbox     = torch.zeros(4, dtype=torch.float32)
            has_bbox = False

        img_tensor = self.transform(img)

        age_norm = (float(s["age"]) - self.age_mean) / (self.age_std + 1e-6)
        sex      = float(s["sex"])

        return {
            "image":    img_tensor,
            "bbox":     bbox,
            "has_bbox": torch.tensor(has_bbox, dtype=torch.bool),
            "age":      torch.tensor(age_norm,  dtype=torch.float32),
            "sex":      torch.tensor(sex,        dtype=torch.float32),
            "label":    torch.tensor(class_id,   dtype=torch.long),
        }


class YOLOEmbeddingDataset(Dataset):
    """Stores pre-computed backbone embeddings + metadata for fast epoch iteration."""

    def __init__(
        self,
        emb:      torch.Tensor,   # (N, D)
        age:      torch.Tensor,   # (N,)
        sex:      torch.Tensor,   # (N,)
        labels:   torch.Tensor,   # (N,)
        bboxes:   torch.Tensor,   # (N, 4)
        has_bbox: torch.Tensor,   # (N,) bool
    ) -> None:
        self.emb      = emb
        self.age      = age
        self.sex      = sex
        self.labels   = labels
        self.bboxes   = bboxes
        self.has_bbox = has_bbox

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.emb[idx],
            self.age[idx],
            self.sex[idx],
            self.labels[idx],
            self.bboxes[idx],
            self.has_bbox[idx],
        )


def extract_yolo_embeddings(
    backbone:  torch.nn.Module,
    loader:    DataLoader,
    device:    torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once over a split and cache all outputs.

    Returns (emb, age, sex, labels, bboxes, has_bbox).
    """
    backbone.eval()
    all_emb, all_age, all_sex, all_lbl = [], [], [], []
    all_bbox, all_has_bbox             = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  embedding", leave=False):
            images = batch["image"].to(device)
            emb    = backbone(images)
            all_emb.append(emb.cpu())
            all_age.append(batch["age"])
            all_sex.append(batch["sex"])
            all_lbl.append(batch["label"])
            all_bbox.append(batch["bbox"])
            all_has_bbox.append(batch["has_bbox"])

    return (
        torch.cat(all_emb),
        torch.cat(all_age),
        torch.cat(all_sex),
        torch.cat(all_lbl),
        torch.cat(all_bbox),
        torch.cat(all_has_bbox),
    )
