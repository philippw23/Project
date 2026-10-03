from __future__ import annotations

from pathlib import Path

from biomedclip.data.splits import build_stratified_splits


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

