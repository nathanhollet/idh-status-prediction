#!/usr/bin/env python3
"""Skull-strip raw NIfTI volumes with HD-BET.

The Erasmus dataset is *not* distributed skull-stripped, so this step has
to be run once before any of the downstream pipelines (radiomics
preprocessing, embedding extraction) can consume it. UCSF-PDGM already
ships with brain segmentations and does not need this script.

File-naming convention (matches what `rfe_preprocessing.py` and the
embedding scripts expect):

    data/<DATASET>/<patient>/<patient>_<sequence>.nii.gz             # skull-stripped (output)
    data/<DATASET>/<patient>/<patient>_<sequence>_with_skull.nii.gz  # original, renamed
    data/<DATASET>/<patient>/<patient>_<sequence>_brain_mask.nii.gz  # HD-BET mask

For each patient, the script:
  1. If a brain mask already exists, skips the patient (idempotent).
  2. Otherwise renames the original ``<pid>_<seq>.nii.gz`` to
     ``<pid>_<seq>_with_skull.nii.gz`` (if not already renamed),
  3. Runs HD-BET on the with-skull volume to produce the brain mask,
  4. Applies the mask to write the skull-stripped ``<pid>_<seq>.nii.gz``.

Usage::

    python src/preprocessing/skull_strip.py --dataset ERASMUS-GBM --sequence FLAIR
    python src/preprocessing/skull_strip.py --dataset ERASMUS-GBM --sequence T1GD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor

from src.utils.paths import cfg, get_dataset_cfg, project_root


def skull_strip_patient(
    patient_dir: Path,
    sequence: str,
    predictor,
) -> str:
    """Skull-strip one patient's volume in place.

    Returns a status string: ``"done"``, ``"skipped"`` (already
    processed), or ``"missing"`` (no input file).
    """
    pid = patient_dir.name
    stripped = patient_dir / f"{pid}_{sequence}.nii.gz"
    with_skull = patient_dir / f"{pid}_{sequence}_with_skull.nii.gz"
    mask = patient_dir / f"{pid}_{sequence}_brain_mask.nii.gz"

    if mask.exists() and with_skull.exists():
        return "skipped"

    # If we have neither the canonical file nor the renamed original,
    # there's nothing to strip.
    if not stripped.exists() and not with_skull.exists():
        return "missing"

    # Move the original out of the way so we can write the stripped
    # output back to the canonical filename.
    if not with_skull.exists():
        stripped.rename(with_skull)

    predictor.predict_from_files(
        [[str(with_skull)]],
        [str(mask)],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    apply_bet(str(with_skull), str(mask), str(stripped))
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skull-strip raw NIfTI volumes with HD-BET (Erasmus only)."
    )
    parser.add_argument(
        "--dataset", type=str, default="ERASMUS-GBM",
        help="Dataset name from cfg.datasets (default: ERASMUS-GBM).",
    )
    parser.add_argument(
        "--sequence", type=str, required=True,
        help="MRI sequence to process (e.g. FLAIR, T1GD).",
    )
    args = parser.parse_args()

    # Validates the dataset name; future-proofing if more datasets
    # ever need skull-stripping.
    get_dataset_cfg(args.dataset)
    data_root = project_root / cfg.paths.data[args.dataset]
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = get_hdbet_predictor(use_tta=False, device=device, verbose=False)

    patients = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name != "preprocessed")
    print(f"Found {len(patients)} patients in {data_root}")

    counts = {"done": 0, "skipped": 0, "missing": 0}
    for patient_dir in patients:
        status = skull_strip_patient(patient_dir, args.sequence, predictor)
        counts[status] += 1
        print(f"  {patient_dir.name} [{args.sequence}]: {status}")

    print(
        f"\nFinished: {counts['done']} stripped, "
        f"{counts['skipped']} already done, {counts['missing']} missing input."
    )


if __name__ == "__main__":
    main()