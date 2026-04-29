from __future__ import annotations

from pathlib import Path

from PIL import Image

from .extended import ExtendedVisionDataset


class BoneTumorDataset(ExtendedVisionDataset):
    """Bone-tumour X-ray dataset for iBOT continued pretraining.

    Receives a pre-filtered list of image paths produced by
    biomedclip.data.splits.build_stratified_splits so that val/test images
    are never seen during SSL pretraining.

    get_target() returns a dummy 0; the iBOT training loop never uses labels.
    """

    def __init__(
        self,
        image_paths: list[Path],
        root: str = "",
        transforms=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transforms=transforms, target_transform=target_transform)
        self.image_paths = list(image_paths)
        print(f"BoneTumorDataset [TRAIN]: {len(self.image_paths)} images")

    def get_image_data(self, index: int):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_target(self, index: int) -> int:
        return 0  # required by ExtendedVisionDataset interface; never used by iBOT

    def __len__(self) -> int:
        return len(self.image_paths)
