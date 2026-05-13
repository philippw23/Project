from __future__ import annotations

import numpy as np
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from biomedclip.data.datasets import LABEL_TO_IDX
from biomedclip.data.transforms import crop_around_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DownstreamDatasetWithText(Dataset):
    """Training dataset that returns image, tokenized report, age, sex, and label.

    Used only for the training split of baseline 2 (scratch image+text), where
    all 971 training samples have reports. At inference time (val/test) the text
    encoder is discarded and the standard DownstreamDataset is used instead.
    """

    def __init__(
        self,
        samples: list[dict],
        age_sex_lookup: dict[str, tuple[float, float]],
        age_mean: float,
        age_std: float,
        preprocess,
        tokenizer,
        max_text_len: int = 128,
        use_mask: bool = False,
    ) -> None:
        valid = []
        for s in samples:
            stem = Path(s["image"]).stem
            if stem not in age_sex_lookup:
                continue
            if s.get("label") not in LABEL_TO_IDX:
                continue
            valid.append(s)

        self.samples        = valid
        self.age_sex_lookup = age_sex_lookup
        self.age_mean       = age_mean
        self.age_std        = age_std
        self.preprocess     = preprocess
        self.tokenizer      = tokenizer
        self.max_text_len   = max_text_len
        self.use_mask       = use_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s    = self.samples[idx]
        stem = Path(s["image"]).stem

        image = Image.open(s["image"]).convert("RGB")
        if self.use_mask:
            mask_path = Path(s["mask"])
            if mask_path.exists():
                mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
                image_arr = np.array(image.convert("L"), dtype=float)
                cropped   = crop_around_mask(image_arr, mask_arr)
                image     = Image.fromarray(cropped.astype(np.uint8)).convert("RGB")
        image_tensor = self.preprocess(image)

        age_raw, sex = self.age_sex_lookup[stem]
        age_norm     = (age_raw - self.age_mean) / (self.age_std + 1e-6)
        label        = LABEL_TO_IDX[s["label"]]

        report = s.get("report") or ""
        enc = self.tokenizer(
            report if report else "[PAD]",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image":          image_tensor,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "age":            torch.tensor(age_norm, dtype=torch.float32),
            "sex":            torch.tensor(sex,      dtype=torch.float32),
            "label":          torch.tensor(label,    dtype=torch.long),
        }
