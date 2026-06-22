from pathlib import Path

import torch

from dataset import BrainAgeDataset


class UPENNDataset(BrainAgeDataset):
    """Per-patient NIfTI loader for UPENN-GBM.

    Patient folders are named ``UPENN-GBM-00001`` but the files inside
    use a timepoint-suffixed prefix, e.g.::

        UPENN-GBM-00001/UPENN-GBM-00001_11_FLAIR.nii.gz

    The per-modality CSVs (and the MAIN CSV) store the timepoint-suffixed
    id as ``pat_id`` (e.g. ``UPENN-GBM-00001_11_FLAIR``), so we strip the
    trailing modality token to recover the file prefix and the trailing
    timepoint token to recover the folder name.
    """

    def __init__(self, csv_path, root_dir, sequence, transform=None):
        super().__init__(csv_path, root_dir, transform)
        self.sequence = sequence

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])  # e.g. UPENN-GBM-00001_11_FLAIR
        label = self.dataframe.loc[idx, "label"]

        file_prefix = "_".join(pat_id.split("_")[:-1])  # UPENN-GBM-00001_11
        dir_name = file_prefix.rsplit("_", 1)[0]         # UPENN-GBM-00001

        img_path = Path(self.root_dir) / dir_name / f"{pat_id}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"File not found: {img_path}")

        sample = self.transform({"image": img_path})
        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id,
        }
