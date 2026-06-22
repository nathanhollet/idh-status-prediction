import os
import sys
import torch
from pathlib import Path

from src.utils.paths import project_root, cfg
from dataset import BrainAgeDataset


class UCSFDataset(BrainAgeDataset):
    def __init__(self, csv_path, root_dir, sequence, transform=None):
        super().__init__(csv_path, root_dir, transform)
        self.sequence = sequence

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']

        base_id = "_".join(pat_id.split("_")[:-1])

        # Try both folder conventions
        nifti_folder = Path(self.root_dir) / f"{base_id}_nifti" / f"{pat_id}.nii.gz"
        flat_folder = Path(self.root_dir) / base_id / f"{pat_id}.nii.gz"

        img_path = nifti_folder if nifti_folder.exists() else flat_folder

        if not img_path.exists():
            raise FileNotFoundError(f"File not found: {img_path}")

        sample = self.transform({"image": img_path})
        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id
        }