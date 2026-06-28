from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from torchvision import transforms

from LACE.data.transforms import (
    rasterize_shapes,
    mask_to_patch_labels,
    synchronized_train_transform,
)
from biomedclip.data.transforms import crop_around_mask_pair, pad_to_square

ImageFile.LOAD_TRUNCATED_IMAGES = True



class InternalDatasetV2(Dataset):
    """LACE v2 internal dataset.

    context_fraction controls lesion cropping for images with a valid mask:
      -1.0  → full image, no crop
       0.0  → tight bbox crop around lesion, no padding
      >0.0  → crop with context margin of that fraction
    Images without a mask always use the full image. patch_labels are always
    computed from the (possibly cropped) mask so they stay aligned with the
    196 ViT patch positions used by MaskTokenModule.

    text_mode controls text encoding:
      "full"   — tokenize full befund/beurteilung strings
      "phrase" — tokenize each phrase separately; returns per-phrase embeddings
      "mixed"  — full beurteilung text for L_ITA + befund phrases for L_sim
    """

    def __init__(
        self,
        samples: list[dict],
        preprocess,
        tokenizer,
        max_text_len: int = 128,
        text_mode: str = "phrase",
        max_bef_phrases: int = 16,
        max_beur_phrases: int = 16,
        phrase_tok_len: int = 32,
        max_beur_text_len: int = 256,
        context_fraction: float = 0.15,
    ) -> None:
        self.preprocess        = preprocess
        self.tokenizer         = tokenizer
        self.max_text_len      = max_text_len
        self.text_mode         = text_mode
        self.max_bef_phrases   = max_bef_phrases
        self.max_beur_phrases  = max_beur_phrases
        self.phrase_tok_len    = phrase_tok_len
        self.max_beur_text_len = max_beur_text_len
        self.context_fraction  = context_fraction

        if text_mode == "phrase":
            self.samples = [
                s for s in samples
                if s.get("befund_phrases") and s.get("beurteilung_phrases")
            ]
            dropped = len(samples) - len(self.samples)
            if dropped:
                print(f"InternalDatasetV2: dropped {dropped} samples missing phrase lists.")
        elif text_mode == "mixed":
            self.samples = [
                s for s in samples
                if s.get("beurteilung") or s.get("befund_phrases")
            ]
            dropped = len(samples) - len(self.samples)
            if dropped:
                print(f"InternalDatasetV2: dropped {dropped} samples missing all text.")
        else:
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tok(self, text: str, max_length: int) -> dict:
        # open_clip HFTokenizer wraps a HF tokenizer; unwrap to get full dict output
        hf_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        return hf_tok(
            text if text else "[PAD]",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _encode_phrase_list(
        self, phrases: list[str], max_phrases: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        image    = Image.open(image_path).convert("RGB")
        has_mask = mask_path.exists()
        if has_mask:
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            has_mask = bool(np.any(mask_arr > 0))
            if has_mask and self.context_fraction >= 0:
                img_arr, crop_mask = crop_around_mask_pair(
                    np.array(image), mask_arr,
                    context_fraction=self.context_fraction,
                )
                image    = Image.fromarray(img_arr)
                mask_arr = crop_mask
            else:
                image = Image.fromarray(pad_to_square(np.array(image)))
            patch_labels = mask_to_patch_labels(mask_arr) if has_mask else \
                torch.zeros(196, dtype=torch.float32)
        else:
            image = Image.fromarray(pad_to_square(np.array(image)))
            patch_labels = torch.zeros(196, dtype=torch.float32)
        full_image = self.preprocess(image)

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
        elif self.text_mode == "mixed":
            # Full beurteilung for L_ITA; befund phrases for L_sim
            b_enc = self._tok(beurteilung, self.max_beur_text_len)
            bef_ids, bef_pattn, bef_pmask = self._encode_phrase_list(
                bef_phrases, self.max_bef_phrases,
            )
            concat_full = " ".join(filter(None, [befund, beurteilung]))
            concat_enc = self._tok(concat_full, self.max_beur_text_len)
            text_fields = {
                "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
                "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
                "bef_phrase_ids":   bef_ids,
                "bef_phrase_attn":  bef_pattn,
                "bef_phrase_mask":  bef_pmask,
                "has_befund":       torch.tensor(bool(bef_phrases), dtype=torch.bool),
                "concat_full_ids":  concat_enc["input_ids"].squeeze(0),
                "concat_full_mask": concat_enc["attention_mask"].squeeze(0),
            }
        else:  # phrase
            beur_ids, beur_pattn, beur_pmask = self._encode_phrase_list(
                beur_phrases, self.max_beur_phrases,
            )
            bef_ids, bef_pattn, bef_pmask = self._encode_phrase_list(
                bef_phrases, self.max_bef_phrases,
            )
            all_text = ", ".join(beur_phrases + bef_phrases)
            concat_enc = self._tok(all_text, self.max_text_len)
            text_fields = {
                "beur_phrase_ids":    beur_ids,
                "beur_phrase_attn":   beur_pattn,
                "beur_phrase_mask":   beur_pmask,
                "bef_phrase_ids":     bef_ids,
                "bef_phrase_attn":    bef_pattn,
                "bef_phrase_mask":    bef_pmask,
                "concat_phrase_ids":  concat_enc["input_ids"].squeeze(0),
                "concat_phrase_mask": concat_enc["attention_mask"].squeeze(0),
                "has_befund":         torch.tensor(bool(bef_phrases), dtype=torch.bool),
            }

        return {
            "full_image":   full_image,
            "patch_labels": patch_labels,
            "has_mask":     torch.tensor(has_mask, dtype=torch.bool),
            **text_fields,
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
      "phrase" — tokenize each phrase separately; returns per-phrase embeddings

    For phrase mode, samples with empty phrase lists are dropped.
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
        global_context_fraction: float = 0.4,
        context_fraction: float = 0.15,
        max_beur_text_len: int = 256,
        descriptor_vectors: dict[str, list[int]] | None = None,
        is_train: bool = False,
    ) -> None:
        self.preprocess              = preprocess
        self.is_train                = is_train
        # Normalize stats reused by the synchronized train transform.
        self._norm = next(
            (t for t in getattr(preprocess, "transforms", [])
             if isinstance(t, transforms.Normalize)),
            None,
        )
        self.tokenizer               = tokenizer
        self.max_text_len            = max_text_len
        self.text_mode               = text_mode
        self.max_bef_phrases         = max_bef_phrases
        self.max_beur_phrases        = max_beur_phrases
        self.phrase_tok_len          = phrase_tok_len
        self.global_context_fraction = global_context_fraction
        self.context_fraction        = context_fraction
        self.max_beur_text_len       = max_beur_text_len
        self.descriptor_vectors      = descriptor_vectors or {}

        if text_mode == "phrase":
            self.samples = [
                s for s in samples
                if s.get("befund_phrases") and s.get("beurteilung_phrases")
            ]
            dropped = len(samples) - len(self.samples)
            if dropped:
                print(f"InternalTripleDataset: dropped {dropped} samples missing phrase lists.")
        elif text_mode == "mixed":
            self.samples = [
                s for s in samples
                if s.get("beurteilung") or s.get("befund_phrases")
            ]
            dropped = len(samples) - len(self.samples)
            if dropped:
                print(f"InternalTripleDataset: dropped {dropped} samples missing all text.")
        else:
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tok(self, text: str, max_length: int) -> dict:
        # open_clip HFTokenizer wraps a HF tokenizer; unwrap to get full dict output
        hf_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        return hf_tok(
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

        has_mask     = False
        img_pil      = image          # fallback global image: full X-ray
        crop_pil     = image          # fallback crop: full X-ray
        global_mask  = None
        crop_mask    = None

        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert("L"), dtype=float)
            if np.any(mask_arr > 0):
                img_arr = np.array(image)
                if self.context_fraction >= 0:
                    img_crop_arr, crop_mask = crop_around_mask_pair(img_arr, mask_arr, context_fraction=self.context_fraction)
                    crop_pil  = Image.fromarray(img_crop_arr)
                else:
                    crop_pil  = Image.fromarray(pad_to_square(img_arr))
                    crop_mask = mask_arr  # full image → mask unchanged
                global_arr, global_mask = crop_around_mask_pair(img_arr, mask_arr, context_fraction=self.global_context_fraction)
                img_pil      = Image.fromarray(global_arr)
                has_mask     = True

        def _mask_pil(mask_arr_, ref_pil):
            if mask_arr_ is None:
                return Image.new("L", ref_pil.size, 0)
            return Image.fromarray((mask_arr_ > 0).astype(np.uint8) * 255, mode="L")

        if self.is_train and self._norm is not None:
            # Synchronized geometric aug keeps each crop's patch labels registered
            # to its augmented image (photometric aug is image-only).
            full_image, patch_labels = synchronized_train_transform(
                img_pil,  _mask_pil(global_mask, img_pil),  self._norm.mean, self._norm.std,
            )
            crop_image, crop_patch_labels = synchronized_train_transform(
                crop_pil, _mask_pil(crop_mask, crop_pil),   self._norm.mean, self._norm.std,
            )
        else:
            full_image = self.preprocess(img_pil)
            crop_image = self.preprocess(crop_pil)
            patch_labels = (
                mask_to_patch_labels(global_mask) if has_mask
                else torch.zeros(196, dtype=torch.float32)
            )
            crop_patch_labels = (
                mask_to_patch_labels(crop_mask) if has_mask
                else torch.zeros(196, dtype=torch.float32)
            )

        if self.text_mode == "full":
            b_enc = self._tok(beurteilung, self.max_text_len)
            f_enc = self._tok(befund, self.max_text_len)
            text_fields = {
                "beurteilung_ids":       b_enc["input_ids"].squeeze(0),
                "beurteilung_mask":      b_enc["attention_mask"].squeeze(0),
                "befund_ids":            f_enc["input_ids"].squeeze(0),
                "befund_mask":           f_enc["attention_mask"].squeeze(0),
                "has_beurteilung":       torch.tensor(bool(beurteilung), dtype=torch.bool),
                "has_befund":            torch.tensor(bool(befund), dtype=torch.bool),
            }
        elif self.text_mode == "concat":
            beur_text = ", ".join(beur_phrases) if beur_phrases else ""
            bef_text  = ", ".join(bef_phrases)  if bef_phrases  else ""
            b_enc = self._tok(beur_text, self.max_text_len)
            f_enc = self._tok(bef_text,  self.max_text_len)
            text_fields = {
                "beurteilung_ids":       b_enc["input_ids"].squeeze(0),
                "beurteilung_mask":      b_enc["attention_mask"].squeeze(0),
                "befund_ids":            f_enc["input_ids"].squeeze(0),
                "befund_mask":           f_enc["attention_mask"].squeeze(0),
                "has_beurteilung":       torch.tensor(bool(beur_phrases), dtype=torch.bool),
                "has_befund":            torch.tensor(bool(bef_phrases), dtype=torch.bool),
            }
        elif self.text_mode == "mixed":
            b_enc = self._tok(beurteilung, self.max_beur_text_len)
            bef_ids, bef_pattn, bef_pmask = self._encode_phrase_list(
                bef_phrases, self.max_bef_phrases,
            )
            concat_full = " ".join(filter(None, [befund, beurteilung]))
            concat_enc = self._tok(concat_full, self.max_beur_text_len)
            text_fields = {
                "beurteilung_ids":  b_enc["input_ids"].squeeze(0),
                "beurteilung_mask": b_enc["attention_mask"].squeeze(0),
                "has_beurteilung":  torch.tensor(bool(beurteilung), dtype=torch.bool),
                "bef_phrase_ids":   bef_ids,
                "bef_phrase_attn":  bef_pattn,
                "bef_phrase_mask":  bef_pmask,
                "has_befund":       torch.tensor(bool(bef_phrases), dtype=torch.bool),
                "concat_full_ids":  concat_enc["input_ids"].squeeze(0),
                "concat_full_mask": concat_enc["attention_mask"].squeeze(0),
            }
        else:  # phrase
            beur_ids, beur_pattn, beur_pmask = self._encode_phrase_list(
                beur_phrases, self.max_beur_phrases,
            )
            bef_ids, bef_pattn, bef_pmask = self._encode_phrase_list(
                bef_phrases, self.max_bef_phrases,
            )
            all_text = ", ".join(beur_phrases + bef_phrases)
            concat_enc = self._tok(all_text, self.max_text_len)
            text_fields = {
                "beur_phrase_ids":    beur_ids,
                "beur_phrase_attn":   beur_pattn,
                "beur_phrase_mask":   beur_pmask,
                "bef_phrase_ids":     bef_ids,
                "bef_phrase_attn":    bef_pattn,
                "bef_phrase_mask":    bef_pmask,
                "concat_phrase_ids":  concat_enc["input_ids"].squeeze(0),
                "concat_phrase_mask": concat_enc["attention_mask"].squeeze(0),
                "has_befund":         torch.tensor(bool(bef_phrases), dtype=torch.bool),
            }

        desc = self.descriptor_vectors.get(str(s["image"]), [0] * 21)
        return {
            "global_crop":       full_image,
            "crop_image":        crop_image,
            "patch_labels":      patch_labels,
            "crop_patch_labels": crop_patch_labels,
            "has_mask":          torch.tensor(has_mask, dtype=torch.bool),
            "descriptor_vec":    torch.tensor(desc, dtype=torch.float32),
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
