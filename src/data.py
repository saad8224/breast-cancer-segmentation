"""
data.py
-------
Dataset loader for the Kaggle "Breast Cancer Cell Segmentation" collection
(Andrew MVD, CC BY 3.0). The dataset ships 58 image–mask pairs of H&E-stained
histopathology tiles, split 46 train / 12 test in this project.

Expected directory layout after downloading & extracting the Kaggle zip:

    data/
    ├── Images/
    │   ├── 10256_500_f00003_original.tif
    │   └── ...
    └── Masks/
        ├── 10256_500_f00003_mask.tif
        └── ...

Usage from Python:

    from data import BreastCancerDataset, make_loaders
    train_loader, test_loader = make_loaders("data", batch_size=4)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T


IMG_SIZE = 256
SEED = 42


class BreastCancerDataset(Dataset):
    """Pairs each image with its mask by matching filename stems."""

    def __init__(self, root: Path, augment: bool = False) -> None:
        root = Path(root)
        self.images = sorted((root / "Images").glob("*"))
        self.masks_dir = root / "Masks"
        if not self.images:
            raise FileNotFoundError(
                f"No images found under {root/'Images'}. Did you download the dataset?"
            )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def _mask_for(self, image_path: Path) -> Path:
        # Kaggle set uses <stem>_original.tif → <stem>_mask.tif
        stem = image_path.stem.replace("_original", "")
        candidates = list(self.masks_dir.glob(f"{stem}*"))
        if not candidates:
            raise FileNotFoundError(f"No mask matching {image_path.name}")
        return candidates[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[i]
        mask_path = self._mask_for(img_path)

        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))

        img_t = T.ToTensor()(img)                     # [3, H, W] in [0,1]
        mask_arr = np.array(mask, dtype=np.float32)   # [H, W]
        mask_t = torch.from_numpy((mask_arr > 127).astype(np.float32)).unsqueeze(0)

        if self.augment:
            # cheap augmentation to help with a 46-sample training set
            if torch.rand(1).item() < 0.5:
                img_t = torch.flip(img_t, dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if torch.rand(1).item() < 0.5:
                img_t = torch.flip(img_t, dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img_t = torch.rot90(img_t, k, dims=(-2, -1))
                mask_t = torch.rot90(mask_t, k, dims=(-2, -1))

        return img_t, mask_t


def make_loaders(root: str, batch_size: int = 4, train_frac: float = 46 / 58):
    """Split 46/12 the way the paper does, deterministically."""
    full = BreastCancerDataset(root, augment=False)
    n = len(full)
    n_train = int(round(n * train_frac))
    n_test = n - n_train
    gen = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(full, [n_train, n_test], generator=gen)
    # Turn augmentation on only for the training subset.
    train_ds.dataset = BreastCancerDataset(root, augment=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )
