"""Registry of per-cohort BrainIAC ``Dataset`` classes.

Each class maps one row of its cohort's MAIN CSV to
``(image, label, pat_id)`` where ``image`` is a 3D MRI volume.
``DATASET_CLASSES`` is the canonical name → loader-class map,
keyed by the dataset names in ``configs.yaml``.
"""

from src.datasets.erasmus_dataset import ERASMUSDataset
from src.datasets.ucsd_ptgbm_dataset import UCSDPTGBMDataset
from src.datasets.ucsf_dataset import UCSFDataset
from src.datasets.upenn_dataset import UPENNDataset
from src.datasets.utsw_dataset import UTSWDataset

DATASET_CLASSES = {
    "UCSF-PDGM": UCSFDataset,
    "ERASMUS-GBM": ERASMUSDataset,
    "UPENN-GBM": UPENNDataset,
    "UTSW-Glioma": UTSWDataset,
    "UCSD-PTGBM": UCSDPTGBMDataset,
}

__all__ = [
    "DATASET_CLASSES",
    "ERASMUSDataset",
    "UCSDPTGBMDataset",
    "UCSFDataset",
    "UPENNDataset",
    "UTSWDataset",
]
