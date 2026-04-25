from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from LACE.data.transforms import extract_centered_crop, rasterize_shapes, mask_to_patch_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InternalTripleDataset(Dataset):
    """Returns all fields needed for L_ITA, L_sim, and internal L_ortho.

    Each sample dict must have keys: image, mask, befund, beurteilung.

    Samples without a valid mask fall back to using the full image as the crop
    and return zeros for patch_labels (has_mask=False signals callers to skip
    L_sim and the internal L_ortho contribution for that sample).
    """

    def __init__(
        self,
        samples: list[dict],
        preprocess,
        tokenizer,
        max_text_len: int = 128,
    ) -> None:
        self.samples      = samples
        self.preprocess   = preprocess
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s           = self.samples[idx]
        image_path  = Path(s["image"])
        mask_path   = Path(s["mask"])
        beurteilung = str(s.get("beurteilung") or "").strip()
        befund      = str(s.get("befund") or "").strip()

        image = Image.open(image_path).convert("RGB")
        full_image = self.preprocess(image)

        has_mask = mask_path.exists()
        mask_arr = None
        if has_mask:
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            has_mask = bool(np.any(mask_arr > 0))

        if has_mask:
            crop_pil     = extract_centered_crop(image, mask_arr, crop_size=224)
            patch_labels = mask_to_patch_labels(mask_arr)
        else:
            crop_pil     = image
            patch_labels = torch.zeros(196, dtype=torch.float32)

        crop_image = self.preprocess(crop_pil)

        def _tokenize(text: str) -> dict:
            return self.tokenizer(
                text if text else "[PAD]",
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        b_enc = _tokenize(beurteilung)
        f_enc = _tokenize(befund)

        return {
            "full_image":       full_image,
            "crop_image":       crop_image,
            "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
            "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
            "befund_ids":       f_enc["input_ids"].squeeze(0),
            "befund_mask":      f_enc["attention_mask"].squeeze(0),
            "patch_labels":     patch_labels,
            "has_mask":         torch.tensor(has_mask, dtype=torch.bool),
            "has_befund":       torch.tensor(bool(befund), dtype=torch.bool),
        }


class BTXRDOrthoDataset(Dataset):
    """BTXRD images with rasterized polygon/rectangle masks for L_ortho only.

    Only images that have a matching annotation JSON are included.
    """

    def __init__(
        self,
        btxrd_images_dir: Path,
        btxrd_annot_dir: Path,
        preprocess,
    ) -> None:
        self.preprocess = preprocess
        self.samples: list[tuple[Path, Path]] = []

        for annot_path in sorted(btxrd_annot_dir.glob("*.json")):
            img_path = btxrd_images_dir / f"{annot_path.stem}.jpeg"
            if not img_path.exists():
                img_path = btxrd_images_dir / f"{annot_path.stem}.jpg"
            if img_path.exists():
                self.samples.append((img_path, annot_path))

        print(f"BTXRDOrthoDataset: {len(self.samples)} annotated images found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, annot_path = self.samples[idx]

        with open(annot_path, "r", encoding="utf-8") as fh:
            annot = json.load(fh)

        image    = Image.open(img_path).convert("RGB")
        img_h    = annot["imageHeight"]
        img_w    = annot["imageWidth"]
        mask_arr = rasterize_shapes(annot["shapes"], img_h, img_w)
        patch_labels = mask_to_patch_labels(mask_arr)

        return {
            "image":        self.preprocess(image),
            "patch_labels": patch_labels,
        }
