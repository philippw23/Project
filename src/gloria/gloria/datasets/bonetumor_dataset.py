from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from transformers import AutoTokenizer

from .pretraining_dataset import multimodal_collate_fn  # noqa: F401  (re-exported for data_module)


class BoneTumorPretrainingDataset(data.Dataset):
    """GLoRIA pretraining dataset for bone-tumor X-rays.

    Loads (image_path, report_text) pairs from metadata.xlsx +
    translated_reports.json and applies the same caption/image pipeline
    as MultimodalPretrainingDataset.
    """

    def __init__(self, cfg, split: str = "train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.max_word_num = cfg.data.text.captions_per_image

        self.samples = self._load_samples(split)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_samples(self, split: str) -> list[tuple[Path, str]]:
        """Load pretrain samples from splits.json (same pool as BiomedCLIP/CheXFound).

        Reads the "pretrain" key from splits.json and then splits 90/10 into
        train / monitor-val.  Using splits.json ensures downstream val/test images
        are excluded and all baselines share the same image pool.
        """
        splits_path = Path(self.cfg.data.splits_path)
        seed = int(self.cfg.data.get("seed", 42))

        with open(splits_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        all_samples: list[tuple[Path, str]] = []
        for entry in manifest["pretrain"]:
            img = Path(entry["image"])
            report = (entry.get("report") or "").strip()
            if img.exists() and report:
                all_samples.append((img, report))

        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * 0.9)
        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        return [all_samples[i] for i in indices]

    # ------------------------------------------------------------------
    # Caption / image helpers (identical logic to MultimodalPretrainingDataset)
    # ------------------------------------------------------------------

    def get_caption(self, report_text: str):
        captions = report_text.replace("\n", " ")
        splitter = re.compile(r"[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cnt = 0
        study_sents: list[str] = []
        for cap in captions:
            if not cap:
                continue
            cap = cap.replace("��", " ")
            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())
            if len(tokens) <= 1:
                continue
            included = [
                t.encode("ascii", "ignore").decode("ascii")
                for t in tokens
                if t.encode("ascii", "ignore").decode("ascii")
            ]
            if not included:
                continue
            study_sents.append(" ".join(included))
            cnt += len(included)
            if cnt >= self.max_word_num:
                break

        if not study_sents:
            study_sents = ["normal"]

        if self.cfg.data.text.full_report:
            sent = " ".join(study_sents)
        else:
            sent = study_sents[np.random.randint(0, len(study_sents))]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        return tokens, x_len

    def get_imgs(self, img_path: Path, transform=None):
        x = cv2.imread(str(img_path), 0)
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def _resize_img(self, img: np.ndarray, scale: int) -> np.ndarray:
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        if max_ind == 0:
            wpercent = scale / float(size[0])
            hsize = int(float(size[1]) * float(wpercent))
            desireable_size = (scale, hsize)
        else:
            hpercent = scale / float(size[1])
            wsize = int(float(size[0]) * float(hpercent))
            desireable_size = (wsize, scale)

        resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)

        if max_ind == 0:
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = bottom = 0
        else:
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = right = 0

        return np.pad(resized_img, [(top, bottom), (left, right)], "constant", constant_values=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        img_path, report_text = self.samples[index]
        imgs = self.get_imgs(img_path, self.transform)
        caps, cap_len = self.get_caption(report_text)
        return imgs, caps, cap_len, str(img_path)

    def __len__(self) -> int:
        return len(self.samples)
