#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from radiomics import featureextractor

from src.utils.paths import project_root, cfg, get_dataset_cfg, canonical_patient_id


def get_dataset_dirs(dataset: str) -> tuple[Path, Path]:
    """Return (base_dir, preprocessed_dir) for the given dataset name."""
    base_dir = project_root / cfg.paths.data[dataset]
    preprocessed_dir = base_dir / "preprocessed"
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}. "
                                f"Run rfe_preprocessing.py first.")
    return base_dir, preprocessed_dir


def resolve_patient_paths(
    patient: str,
    patient_dir: Path,
    base_dir: Path,
    ds_cfg,
    sequences: list[str],
) -> tuple[dict, dict, str]:
    """Resolve per-modality mask + image paths for a patient.

    Returns ``(masks, images, canonical_id)`` where ``masks`` and
    ``images`` are both ``{modality: Path}`` dicts keyed by the names in
    ``sequences``. For datasets with a single patient-level mask
    (UCSF/UPENN/ERASMUS) every modality points at the same file; for
    datasets with per-sequence masks each modality has its own file
    written by ``rfe_preprocessing.py``.
    """
    base_id = canonical_patient_id(patient, ds_cfg)

    per_sequence = bool(ds_cfg.get("per_sequence_mask", False))

    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for modality in sequences:
        if per_sequence:
            mask_fname = f"{base_id}_{modality}_tumor_mask.nii.gz"
        else:
            mask_fname = f"{base_id}_tumor_mask.nii.gz"

        if ds_cfg.mask_in_preprocessed:
            masks[modality] = patient_dir / mask_fname
        else:
            masks[modality] = base_dir / patient / mask_fname

        images[modality] = patient_dir / f"{base_id}_{modality}.nii.gz"

    return masks, images, base_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features")
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name,
                        help="Dataset name (e.g. UCSF-PDGM, ERASMUS-GBM)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV file. Defaults to cfg.paths.data.csvs[RADIOMICS-<prefix>-RAW].")
    parser.add_argument("--t1_name", type=str, default=None,
                        help="T1 modality name override (defaults to dataset config)")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                        help="Modalities to extract features from "
                             "(e.g. 'FLAIR T1c T2'). Defaults to ['FLAIR', "
                             "<t1_name>] to preserve current behavior. Each "
                             "modality must have a "
                             "``{preprocessed}/{patient}/{patient}_{modality}.nii.gz`` "
                             "image alongside the tumor mask.")
    parser.add_argument("--append", action="store_true",
                        help="Append new rows to the output CSV instead of "
                             "overwriting it. Skips (patient, modality) pairs "
                             "already present so you can extract only the new "
                             "sequences without redoing previously computed ones.")
    args = parser.parse_args()

    ds_cfg = get_dataset_cfg(args.dataset)
    t1_name = args.t1_name or ds_cfg.t1_name
    sequences = args.sequences or ["FLAIR", t1_name]

    base_dir, preprocessed_dir = get_dataset_dirs(args.dataset)
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        ds_prefix = args.dataset.split("-")[0]
        output_csv = project_root / cfg.paths.data.csvs[f"RADIOMICS-{ds_prefix}-RAW"]

    extractor = featureextractor.RadiomicsFeatureExtractor()
    all_results = []

    existing_pairs: set[tuple[str, str]] = set()
    if args.append and output_csv.exists():
        existing = pd.read_csv(output_csv, usecols=["PatientID", "Modality"])
        existing_pairs = set(zip(existing["PatientID"].astype(str), existing["Modality"]))
        print(f"Append mode: {len(existing_pairs)} (patient, modality) rows already present.")

    patients = sorted(p for p in os.listdir(preprocessed_dir) if (preprocessed_dir / p).is_dir())

    for patient in patients:
        patient_dir = preprocessed_dir / patient
        masks, images, canonical_id = resolve_patient_paths(
            patient, patient_dir, base_dir, ds_cfg, sequences
        )

        for modality in sequences:
            if (canonical_id, modality) in existing_pairs:
                continue
            image_path = images[modality]
            mask_path = masks[modality]
            if not mask_path.exists():
                print(f"Skipping {patient} {modality}: mask missing at {mask_path}")
                continue
            if not image_path.exists():
                print(f"Skipping {patient} {modality}: image missing at {image_path}")
                continue

            print(f"Processing {patient} ({modality})")
            try:
                result = extractor.execute(str(image_path), str(mask_path))
            except ValueError as e:
                # Empty / unreadable tumor mask (e.g. a few UTSW patients
                # ship an all-zero FeTS fallback). Skip rather than abort
                # the whole extraction.
                print(f"Skipping {patient} {modality}: {e}")
                continue
            result["PatientID"] = canonical_id
            result["Modality"] = modality
            all_results.append(result)

    if not all_results:
        print("No features extracted. Check dataset name and preprocessed data.")
        return

    df = pd.DataFrame(all_results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and output_csv.exists():
        existing = pd.read_csv(output_csv)
        df = pd.concat([existing, df], ignore_index=True, sort=False)
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to: {output_csv}")


if __name__ == "__main__":
    main()
