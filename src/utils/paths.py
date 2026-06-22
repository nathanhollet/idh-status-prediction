from pathlib import Path
import sys
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "reference" / "BrainIAC" / "src"))

cfg = OmegaConf.load(project_root / "configs" / "configs.yaml")


def get_dataset_cfg(dataset: str):
    """Return the dataset-specific config block from cfg.datasets."""
    if dataset not in cfg.datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(cfg.datasets.keys())}")
    return cfg.datasets[dataset]


DEFAULT_TASK = "IDH"


def task_main_key(dataset: str, task: str = DEFAULT_TASK) -> str:
    """Config-key for the MAIN (pat_id, age, grade, label) CSV of a (dataset, task)."""
    return f"{dataset}-MAIN" if task == DEFAULT_TASK else f"{dataset}-{task}-MAIN"


def task_splits_key(ds_prefix: str, task: str = DEFAULT_TASK) -> str:
    """Config-key for the splits directory of a (dataset-prefix, task)."""
    return f"{ds_prefix}-SPLITS" if task == DEFAULT_TASK else f"{ds_prefix}-{task}-SPLITS"


def task_radiomics_key(ds_prefix: str, task: str = DEFAULT_TASK) -> str:
    """Config-key for the *merged* radiomics CSV of a (dataset-prefix, task).

    The raw PyRadiomics CSV is task-agnostic, so ``RADIOMICS-<prefix>-RAW``
    is shared across tasks — only the merged/labeled table varies.
    """
    return f"RADIOMICS-{ds_prefix}" if task == DEFAULT_TASK else f"RADIOMICS-{ds_prefix}-{task}"


def task_results_subdir(base: "Path", task: str = DEFAULT_TASK) -> "Path":
    """Return the results dir for a given task.

    IDH (default) writes to results/models/<model>/<dataset>/. 
    Non-default tasks get nested under .../<task>/ to avoid collisions.
    """
    return base if task == DEFAULT_TASK else base / task


def canonical_patient_id(dir_name: str, ds_cfg) -> str:
    """Map a raw patient folder name to the canonical patient id.

    The canonical id is the prefix used for all per-patient files
    (e.g. ``<id>_FLAIR.nii.gz``) and matches the ``pat_id`` column in
    the labels CSVs. It is computed as::

        canonical = dir_name.removesuffix(id_suffix) + file_id_suffix

    Examples (given the cfgs in ``configs/configs.yaml``):
        UCSF "UCSF-PDGM-0004_nifti" -> "UCSF-PDGM-0004"
        ERASMUS "EGD-0004"          -> "EGD-0004"
        UPENN "UPENN-GBM-00001"     -> "UPENN-GBM-00001_11"
    """
    name = dir_name
    id_suffix = ds_cfg.get("id_suffix", "") or ""
    if id_suffix and name.endswith(id_suffix):
        name = name[: -len(id_suffix)]
    file_id_suffix = ds_cfg.get("file_id_suffix", "") or ""
    if file_id_suffix:
        name = name + file_id_suffix
    return name