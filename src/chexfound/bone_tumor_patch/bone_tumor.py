from __future__ import annotations

from pathlib import Path

from PIL import Image

from .extended import ExtendedVisionDataset


class BoneTumorDataset(ExtendedVisionDataset):
    """Flat folder of bone-tumour PNGs/JPEGs for iBOT continued pretraining.

    Dataset string: "BoneTumor:split=TRAIN:root=/path/to/images"

    The 'split' argument is accepted but ignored — all images are used for SSL.
    get_target() returns a dummy 0; the iBOT training loop never uses labels.
    """

    def __init__(
        self,
        root: str,
        extra: str | None = None,
        split: str = "TRAIN",
        transforms=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transforms=transforms, target_transform=target_transform)
        extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
        self.image_paths = sorted(
            p for p in Path(root).iterdir() if p.suffix in extensions
        )
        print(f"BoneTumorDataset: {len(self.image_paths)} images found in {root}")

    def get_image_data(self, index: int):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_target(self, index: int) -> int:
        return 0  # required by ExtendedVisionDataset interface; never used by iBOT

    def __len__(self) -> int:
        return len(self.image_paths)
