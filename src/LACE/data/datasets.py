from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from LACE.data.transforms import extract_centered_crop, rasterize_shapes, mask_to_patch_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InternalDatasetV2(Dataset):
    """LACE v2 internal dataset — full image only, no crop.

    Returns the same text fields as InternalTripleDataset but drops crop_image.
    patch_labels (GT mask in 196-patch space) is still included so that
    MaskTokenModule can be supervised via seg_loss on samples where has_mask=True.
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

        image      = Image.open(image_path).convert("RGB")
        full_image = self.preprocess(image)

        has_mask = mask_path.exists()
        if has_mask:
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            has_mask = bool(np.any(mask_arr > 0))
            patch_labels = mask_to_patch_labels(mask_arr) if has_mask else \
                torch.zeros(196, dtype=torch.float32)
        else:
            patch_labels = torch.zeros(196, dtype=torch.float32)

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
            "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
            "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
            "befund_ids":       f_enc["input_ids"].squeeze(0),
            "befund_mask":      f_enc["attention_mask"].squeeze(0),
            "patch_labels":     patch_labels,
            "has_mask":         torch.tensor(has_mask, dtype=torch.bool),
            "has_befund":       torch.tensor(bool(befund), dtype=torch.bool),
        }


class InternalTripleDataset(Dataset):
    """Returns all fields needed for L_ITA, L_sim, and internal L_ortho.

    Each sample dict must have keys: image, mask, befund, beurteilung.

    Samples without a valid mask fall back to using the full image as the crop
    and return zeros for patch_labels (has_mask=False signals callers to skip
    L_sim and the internal L_ortho contribution for that sample).

    text_mode controls how text is encoded:
      "full"        — tokenize full befund/beurteilung strings (existing behaviour)
      "concat"      — join phrase lists with ", " and tokenize as one string
      "phrase_mean" — tokenize each phrase separately; encoder mean-pools CLS
      "phrase_attn" — tokenize each phrase separately; encoder attention-pools CLS

    For phrase_mean/phrase_attn, samples with empty phrase lists are dropped.
    """

    def __init__(
        self,
        samples: list[dict],
        preprocess,
        tokenizer,
        max_text_len: int = 128,
        text_mode: str = "full",
        max_bef_phrases: int = 16,
        max_beur_phrases: int = 16,
        phrase_tok_len: int = 32,
    ) -> None:
        self.preprocess       = preprocess
        self.tokenizer        = tokenizer
        self.max_text_len     = max_text_len
        self.text_mode        = text_mode
        self.max_bef_phrases  = max_bef_phrases
        self.max_beur_phrases = max_beur_phrases
        self.phrase_tok_len   = phrase_tok_len

        if text_mode in ("phrase_mean", "phrase_attn"):
            self.samples = [
                s for s in samples
                if s.get("befund_phrases") and s.get("beurteilung_phrases")
            ]
            dropped = len(samples) - len(self.samples)
            if dropped:
                print(f"InternalTripleDataset: dropped {dropped} samples missing phrase lists.")
        else:
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tok(self, text: str, max_length: int) -> dict:
        return self.tokenizer(
            text if text else "[PAD]",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _encode_phrase_list(
        self, phrases: list[str], max_phrases: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize up to max_phrases phrases, pad the rest with zeros.

        Returns:
            ids  [max_phrases, phrase_tok_len]
            attn [max_phrases, phrase_tok_len]
            mask [max_phrases]  True = real phrase, False = padding
        """
        T = self.phrase_tok_len
        ids_list, attn_list, mask_list = [], [], []
        for i in range(max_phrases):
            if i < len(phrases):
                enc = self._tok(phrases[i], T)
                ids_list.append(enc["input_ids"].squeeze(0))
                attn_list.append(enc["attention_mask"].squeeze(0))
                mask_list.append(True)
            else:
                ids_list.append(torch.zeros(T, dtype=torch.long))
                attn_list.append(torch.zeros(T, dtype=torch.long))
                mask_list.append(False)
        return (
            torch.stack(ids_list),
            torch.stack(attn_list),
            torch.tensor(mask_list, dtype=torch.bool),
        )

    def __getitem__(self, idx: int) -> dict:
        s           = self.samples[idx]
        image_path  = Path(s["image"])
        mask_path   = Path(s["mask"])
        beurteilung  = str(s.get("beurteilung") or "").strip()
        befund       = str(s.get("befund") or "").strip()
        beur_phrases = s.get("beurteilung_phrases") or []
        bef_phrases  = s.get("befund_phrases") or []

        image = Image.open(image_path).convert("RGB")
        full_image = self.preprocess(image)

        has_mask     = False
        crop_pil     = image
        patch_labels = torch.zeros(196, dtype=torch.float32)

        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            if np.any(mask_arr > 0):
                crop_pil, mask_crop = extract_centered_crop(image, mask_arr, crop_size=224)
                patch_labels        = mask_to_patch_labels(mask_crop)
                has_mask            = True

        crop_image = self.preprocess(crop_pil)

        if self.text_mode == "full":
            b_enc = self._tok(beurteilung, self.max_text_len)
            f_enc = self._tok(befund, self.max_text_len)
            text_fields = {
                "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
                "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
                "befund_ids":       f_enc["input_ids"].squeeze(0),
                "befund_mask":      f_enc["attention_mask"].squeeze(0),
                "has_befund":       torch.tensor(bool(befund), dtype=torch.bool),
            }
        elif self.text_mode == "concat":
            beur_text = ", ".join(beur_phrases) if beur_phrases else ""
            bef_text  = ", ".join(bef_phrases)  if bef_phrases  else ""
            b_enc = self._tok(beur_text, self.max_text_len)
            f_enc = self._tok(bef_text,  self.max_text_len)
            text_fields = {
                "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
                "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
                "befund_ids":       f_enc["input_ids"].squeeze(0),
                "befund_mask":      f_enc["attention_mask"].squeeze(0),
                "has_befund":       torch.tensor(bool(bef_phrases), dtype=torch.bool),
            }
        else:  # phrase_mean or phrase_attn
            beur_ids, beur_pattn, beur_pmask = self._encode_phrase_list(
                beur_phrases, self.max_beur_phrases,
            )
            bef_ids, bef_pattn, bef_pmask = self._encode_phrase_list(
                bef_phrases, self.max_bef_phrases,
            )
            text_fields = {
                "beur_phrase_ids":  beur_ids,
                "beur_phrase_attn": beur_pattn,
                "beur_phrase_mask": beur_pmask,
                "bef_phrase_ids":   bef_ids,
                "bef_phrase_attn":  bef_pattn,
                "bef_phrase_mask":  bef_pmask,
                "has_befund":       torch.tensor(bool(bef_phrases), dtype=torch.bool),
            }

        return {
            "full_image":   full_image,
            "crop_image":   crop_image,
            "patch_labels": patch_labels,
            "has_mask":     torch.tensor(has_mask, dtype=torch.bool),
            **text_fields,
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
            img_path = btxrd_images_dir / f"{annot_path.stem}.png"
            if not img_path.exists():
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

        # pad mask to match the square-padded preprocessed image
        side     = max(img_h, img_w)
        pad_top  = (side - img_h) // 2
        pad_left = (side - img_w) // 2
        padded   = np.zeros((side, side), dtype=mask_arr.dtype)
        padded[pad_top:pad_top + img_h, pad_left:pad_left + img_w] = mask_arr
        mask_arr = padded

        patch_labels = mask_to_patch_labels(mask_arr)

        return {
            "image":        self.preprocess(image),
            "patch_labels": patch_labels,
        }
