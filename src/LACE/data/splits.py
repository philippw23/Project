from __future__ import annotations

import random
from pathlib import Path

from biomedclip.data.splits import build_stratified_splits
from LACE.data.datasets import InternalDatasetV2, InternalTripleDataset


def build_lace_splits(
    args,
    run_dir: Path | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Thin backward-compat wrapper — calls build_stratified_splits and returns 4 values.

    Returns (train, val, test, test) — the first element doubles as the pretrain
    set; the repeated test is a placeholder so existing callers that unpack four
    values still work.  Prefer calling build_stratified_splits directly.
    """
    train, val, test = build_stratified_splits(args, run_dir=run_dir)
    return train, val, test, test


def build_pretrain_datasets_lace(
    pretrain_samples: list[dict],
    preprocess_train,
    preprocess_val,
    tokenizer,
    seed: int,
    monitor_val_frac: float = 0.1,
    max_text_len: int = 128,
    text_mode: str = "full",
    max_bef_phrases: int = 16,
    max_beur_phrases: int = 16,
) -> tuple[InternalTripleDataset, InternalTripleDataset]:
    """90/10 random split of pretrain_samples into train and monitor-val datasets."""
    rng     = random.Random(seed)
    indices = list(range(len(pretrain_samples)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [pretrain_samples[i] for i in indices[:split]]
    val_samp   = [pretrain_samples[i] for i in indices[split:]]

    print(f"Pretrain loop split: {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = InternalTripleDataset(
        train_samp, preprocess_train, tokenizer, max_text_len,
        text_mode, max_bef_phrases, max_beur_phrases,
    )
    val_ds = InternalTripleDataset(
        val_samp, preprocess_val, tokenizer, max_text_len,
        text_mode, max_bef_phrases, max_beur_phrases,
    )
    return train_ds, val_ds


def build_pretrain_datasets_lace_v2(
    pretrain_samples: list[dict],
    preprocess_train,
    preprocess_val,
    tokenizer,
    seed: int,
    monitor_val_frac: float = 0.1,
    max_text_len: int = 128,
    text_mode: str = "phrase_attn",
    max_bef_phrases: int = 16,
    max_beur_phrases: int = 16,
) -> tuple[InternalDatasetV2, InternalDatasetV2]:
    """90/10 random split of pretrain_samples into train and monitor-val datasets (v2)."""
    rng     = random.Random(seed)
    indices = list(range(len(pretrain_samples)))
    rng.shuffle(indices)
    split      = int(len(indices) * (1.0 - monitor_val_frac))
    train_samp = [pretrain_samples[i] for i in indices[:split]]
    val_samp   = [pretrain_samples[i] for i in indices[split:]]

    print(f"Pretrain loop split (v2): {len(train_samp)} train / {len(val_samp)} monitor-val")

    train_ds = InternalDatasetV2(
        train_samp, preprocess_train, tokenizer, max_text_len,
        text_mode, max_bef_phrases, max_beur_phrases,
    )
    val_ds = InternalDatasetV2(
        val_samp, preprocess_val, tokenizer, max_text_len,
        text_mode, max_bef_phrases, max_beur_phrases,
    )
    return train_ds, val_ds
