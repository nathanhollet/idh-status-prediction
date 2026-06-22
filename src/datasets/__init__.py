"""Dataset package.

Intentionally lightweight: the per-cohort BrainIAC loaders
(``{ucsf,upenn,erasmus,utsw,ucsd_ptgbm}_dataset.py``) import
``BrainAgeDataset`` from the BrainIAC reference tree, which is only on
``sys.path`` when the BrainIAC embedding script runs. Eager-importing
them from this package would break any other caller that just wants
``embedding_dataset`` (e.g. ``train_lin_probe.py``).

If you need the registry of per-cohort classes, import
``src.datasets.brainiac_registry.DATASET_CLASSES`` — that module sits
behind the BrainIAC-path setup in ``get_emb_brainiac.py``.
"""
