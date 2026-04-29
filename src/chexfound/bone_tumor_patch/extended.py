"""Base dataset class replacing CheXFound's ExtendedVisionDataset."""
from __future__ import annotations

from abc import abstractmethod

from torch.utils.data import Dataset


class ExtendedVisionDataset(Dataset):
    """Abstract base matching the CheXFound ExtendedVisionDataset interface.

    Subclasses implement get_image_data() and get_target(); this class wires
    them into the standard PyTorch __getitem__ contract.
    """

    def __init__(self, root: str, transforms=None, target_transform=None) -> None:
        self.root = root
        self.transforms = transforms
        self.target_transform = target_transform

    @abstractmethod
    def get_image_data(self, index: int):
        """Return a PIL Image for the given index."""

    @abstractmethod
    def get_target(self, index: int) -> int:
        """Return the class label for the given index."""

    @abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
