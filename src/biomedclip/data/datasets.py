from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .transforms import crop_around_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_TO_IDX = {"benign": 0, "intermediate": 1, "malignant": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES  = 3


class BoneTumorPairDataset(Dataset):
    """(image tensor, text token tensor) pairs for contrastive pretraining."""

    def __init__(
        self,
        samples: list[tuple[Path, Path, str]],
        preprocess,
        tokenizer,
        use_mask: bool,
    ) -> None:
        self.samples   = samples
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.use_mask   = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, mask_path, text = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.use_mask and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr  = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped   = crop_around_mask(image_arr, mask_arr)
            image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image_tensor = self.preprocess(image)
        text_tokens  = self.tokenizer([text], context_length=256).squeeze(0)

        return {"image": image_tensor, "text": text_tokens}


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
        self.samples       = valid
        self.age_sex_lookup = age_sex_lookup
        self.age_mean      = age_mean
        self.age_std       = age_std
        self.preprocess    = preprocess
        self.use_mask      = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s    = self.samples[idx]
        stem = Path(s["image"]).stem
        age_raw, sex = self.age_sex_lookup[stem]

        image     = Image.open(s["image"]).convert("RGB")
        mask_path = Path(s["mask"])
        if self.use_mask and mask_path.exists():
            image_arr = np.array(image.convert("L"), dtype=float)
            mask_arr  = np.array(Image.open(mask_path).convert("L"), dtype=float)
            cropped   = crop_around_mask(image_arr, mask_arr)
            image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")

        image_tensor = self.preprocess(image)
        age_norm     = (age_raw - self.age_mean) / (self.age_std + 1e-6)
        label        = LABEL_TO_IDX[s["label"]]

        return {
            "image": image_tensor,
            "age":   torch.tensor(age_norm, dtype=torch.float32),
            "sex":   torch.tensor(sex,      dtype=torch.float32),
            "label": torch.tensor(label,    dtype=torch.long),
        }


class EmbeddingDataset(Dataset):
    """Wraps pre-computed embeddings, age, sex, and labels."""

    def __init__(self, emb, age, sex, lbl):
        self.emb = emb
        self.age = age
        self.sex = sex
        self.lbl = lbl

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        return self.emb[idx], self.age[idx], self.sex[idx], self.lbl[idx]
