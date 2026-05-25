from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms

from .feature_engineering import feature_engineering_pipeline


class GalaxyDataset(Dataset):
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        fits_dir: str,
        target_shape: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        self.df = catalog_df.reset_index(drop=True)
        self.fits_dir = Path(fits_dir)
        self.target_shape = target_shape
        self.augment = augment
        self.asinh = None

        self.augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 360), fill=0.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        galaxy_id = row["name"]
        label = int(row["label"])

        base_path = self.fits_dir / f"label_{label}"
        bands_data = []

        for band in ("g", "r", "z"):
            file_path = base_path / band / f"{galaxy_id}_{band}.fits"
            with fits.open(file_path) as hdul:
                raw_data = hdul[0].data.astype(np.float64)

            processed_band = feature_engineering_pipeline(raw_data, self.target_shape)
            bands_data.append(processed_band)

        galaxy_tensor = torch.tensor(np.stack(bands_data, axis=0), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.augment:
            galaxy_tensor = self.augmentation_transforms(galaxy_tensor)

        return galaxy_tensor, label_tensor
