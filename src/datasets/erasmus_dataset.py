import os
import torch

from dataset import BrainAgeDataset

class ERASMUSDataset(BrainAgeDataset):
    def __init__(self, csv_path, root_dir, sequence, transform=None):
        super().__init__(csv_path, root_dir, transform)
        self.sequence = sequence

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']

        base_id = pat_id.split("_")[0]
        img_path = os.path.join(self.root_dir, base_id, f"{pat_id}.nii.gz")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        sample = self.transform({"image": img_path})
        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id
        }