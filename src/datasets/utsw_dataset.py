from pathlib import Path

import torch

from dataset import BrainAgeDataset


class UTSWDataset(BrainAgeDataset):
    """Per-patient NIfTI loader for UTSW-Glioma.

    Patient folders are named ``BT0001`` and images ship with fixed
    (non-patient-prefixed) filenames::

        BT0001/brain_fl_ants.nii.gz
        BT0001/brain_t1ce_ants.nii.gz

    The per-modality CSVs store ``pat_id`` with a sequence suffix
    (``BT0001_FLAIR`` / ``BT0001_T1c``); we split it to recover the
    patient folder and look up the hard-coded per-sequence filename.
    Images are already skull-stripped (ANTs-registered), so no extra
    masking step is needed — matches UCSF-PDGM in that respect.
    """

    SEQUENCE_FILENAMES: dict[str, str] = {
        "FLAIR": "brain_fl_ants.nii.gz",
        "T1c": "brain_t1ce_ants.nii.gz",
        "T1": "brain_t1_ants.nii.gz",
        "T2": "brain_t2_ants.nii.gz",
    }

    def __init__(self, csv_path, root_dir, sequence, transform=None):
        super().__init__(csv_path, root_dir, transform)
        self.sequence = sequence

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])  # e.g. BT0001_FLAIR
        label = self.dataframe.loc[idx, "label"]

        base_id = pat_id.rsplit("_", 1)[0]  # BT0001
        try:
            fname = self.SEQUENCE_FILENAMES[self.sequence]
        except KeyError as e:
            raise ValueError(
                f"Unknown UTSW sequence '{self.sequence}'. "
                f"Available: {list(self.SEQUENCE_FILENAMES)}"
            ) from e

        img_path = Path(self.root_dir) / base_id / fname
        if not img_path.exists():
            raise FileNotFoundError(f"File not found: {img_path}")

        sample = self.transform({"image": img_path})
        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id,
        }
