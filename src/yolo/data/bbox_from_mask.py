"""Extract per-component bounding boxes from a binary segmentation mask.

Returns YOLO-normalised (x_center, y_center, width, height) coordinates,
adjusted for SquarePad geometry (shorter side padded to match the longer side).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LABEL_TO_IDX: dict[str, int] = {"benign": 0, "intermediate": 1, "malignant": 2}


def bboxes_from_mask(
    mask_path: Path,
    class_id: int,
    image_w: int,
    image_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """Return YOLO bboxes for every connected component in *mask_path*.

    Coordinates are normalised to the SquarePad-extended image
    (side = max(image_w, image_h)), matching the geometry applied during
    training by the Ultralytics loader.

    Returns a list of (class_id, x_center, y_center, width, height).
    Returns [] when the mask file is missing or contains no foreground pixels.
    """
    if not mask_path.exists():
        return []

    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary = (mask > 0).astype(np.uint8)

    if not np.any(binary):
        return []

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    side = max(image_w, image_h)
    pad_left = (side - image_w) // 2
    pad_top = (side - image_h) // 2

    boxes: list[tuple[int, float, float, float, float]] = []
    for i in range(1, n_labels):  # label 0 is background
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        x_c = (x + pad_left + w / 2) / side
        y_c = (y + pad_top + h / 2) / side
        w_n = w / side
        h_n = h / side

        boxes.append((class_id, x_c, y_c, w_n, h_n))

    return boxes
