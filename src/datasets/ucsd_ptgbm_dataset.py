from pathlib import Path

import torch

from dataset import BrainAgeDataset


class UCSDPTGBMDataset(BrainAgeDataset):
    """Per-patient NIfTI loader for UCSD-PTGBM (external IDH test cohort).

    Patient folders are named ``UCSD-PTGBM-XXXX_NN`` (one timepoint per
    patient after ``ucsd_csv_assembly.py``). Files inside the folder are
    prefixed with the folder name::

        UCSD-PTGBM-0001_01/
            UCSD-PTGBM-0001_01_FLAIR.nii.gz
            UCSD-PTGBM-0001_01_T1post.nii.gz           ← codebase's ``T1c``
            UCSD-PTGBM-0001_01_BraTS_tumor_seg.nii.gz

    The per-modality CSVs carry ``pat_id`` with a sequence suffix
    (``UCSD-PTGBM-0001_01_FLAIR`` / ``..._T1c``); we strip the suffix to
    recover the folder and use ``SEQUENCE_FILENAMES`` to map the logical
    ``T1c`` sequence to the on-disk ``_T1post`` filename. Images are
    already BraTS-preprocessed (skull-stripped, co-registered), so no
    extra masking is needed.
    """

    SEQUENCE_SUFFIXES: dict[str, str] = {
        "FLAIR": "FLAIR",
        "T1c": "T1post",
        "T1": "T1pre",
        "T2": "T2",
    }

    def __init__(self, csv_path, root_dir, sequence, transform=None):
        super().__init__(csv_path, root_dir, transform)
        self.sequence = sequence

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, "pat_id"])  # e.g. UCSD-PTGBM-0001_01_FLAIR
        label = self.dataframe.loc[idx, "label"]

        base_id = pat_id.rsplit("_", 1)[0]  # UCSD-PTGBM-0001_01
        try:
            disk_suffix = self.SEQUENCE_SUFFIXES[self.sequence]
        except KeyError as e:
            raise ValueError(
                f"Unknown UCSD-PTGBM sequence '{self.sequence}'. "
                f"Available: {list(self.SEQUENCE_SUFFIXES)}"
            ) from e

        img_path = Path(self.root_dir) / base_id / f"{base_id}_{disk_suffix}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"File not found: {img_path}")

        sample = self.transform({"image": img_path})
        return {
            "image": sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id,
        }
