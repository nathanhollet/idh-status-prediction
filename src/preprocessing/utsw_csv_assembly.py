#!/usr/bin/env python3
"""Assemble UTSW-Glioma ``pat_id, age, grade, label`` MAIN csvs.

Reads:
  * ``data/csvs/UTSW_Glioma_Metadata-2-1.tsv``

Writes:
  * ``data/csvs/UTSW_IDH_labels.csv``              (pat_id, label)
  * ``data/csvs/UTSW_IDH_age_grade_labels.csv``    (pat_id, age, grade, label)
  * ``data/csvs/UTSW_IDH_labels_FLAIR.csv``        (pat_id, label) — BrainIAC loader
  * ``data/csvs/UTSW_IDH_labels_T1c.csv``          (pat_id, label) — BrainIAC loader

Inclusion criteria:
  * Patient folder exists on disk and contains FLAIR (``brain_fl_ants.nii.gz``),
    T1c (``brain_t1ce_ants.nii.gz``), and at least one usable tumor seg
    (``rtumorseg_manual_correction.nii.gz`` or ``tumorseg_FeTS.nii.gz``).
  * ``IDH`` ∈ {"wild type", "mutated"} (drops "NA").

Label encoding: IDH 0 = wildtype, 1 = mutated.

Notes
-----
* Patient folders are named ``BT0001`` (no prefix/suffix), so the canonical
  ``pat_id`` is just the folder name. Per-sequence tables append ``_FLAIR`` /
  ``_T1c`` to match the BrainIAC loader convention (``<id>_<seq>.nii.gz``).
* Grade values are kept as-is (2/3/4); rows with grade NA are coerced to -1
  so ``stratified_split`` treats them as an "unknown" bucket rather than
  dropping them.
* Age comes straight from "Age at Imaging".
* The metadata TSV has CRLF line endings; ``pd.read_csv`` handles that
  transparently, but we still strip whitespace from string columns
  defensively.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils.paths import project_root


# Preferred tumor mask file (matches the image voxel grid) and fallback.
PREFERRED_SEG = "rtumorseg_manual_correction.nii.gz"
FALLBACK_SEG = "tumorseg_FeTS.nii.gz"
REQUIRED_IMAGES = ["brain_fl_ants.nii.gz", "brain_t1ce_ants.nii.gz"]


def _seg_has_tumor(seg_path: Path) -> bool:
    """True iff the seg nifti has ≥1 non-zero voxel. A few UTSW patients
    ship an all-zero FeTS fallback, which breaks slice-based embedders
    (BiomedCLIP / MRICore) and radiomics — drop them at assembly time."""
    try:
        return bool(np.any(nib.load(str(seg_path)).get_fdata() > 0))
    except Exception:
        return False


def has_complete_imaging(patient_dir: Path) -> bool:
    if not all((patient_dir / f).exists() for f in REQUIRED_IMAGES):
        return False
    for seg_name in (PREFERRED_SEG, FALLBACK_SEG):
        seg = patient_dir / seg_name
        if seg.exists() and _seg_has_tumor(seg):
            return True
    return False


def _write_task_csvs(df: pd.DataFrame, csvs_dir: Path, task: str) -> None:
    """Write the four per-task CSVs (MAIN + IDH-style + 2x per-sequence)."""
    main_df = df[["pat_id", "age", "grade", "label"]]
    labels_df = df[["pat_id", "label"]]

    out_main = csvs_dir / f"UTSW_{task}_age_grade_labels.csv"
    out_labels = csvs_dir / f"UTSW_{task}_labels.csv"
    out_flair = csvs_dir / f"UTSW_{task}_labels_FLAIR.csv"
    out_t1c = csvs_dir / f"UTSW_{task}_labels_T1c.csv"

    main_df.to_csv(out_main, index=False)
    labels_df.to_csv(out_labels, index=False)

    flair_df = labels_df.copy()
    flair_df["pat_id"] = flair_df["pat_id"] + "_FLAIR"
    flair_df.to_csv(out_flair, index=False)

    t1c_df = labels_df.copy()
    t1c_df["pat_id"] = t1c_df["pat_id"] + "_T1c"
    t1c_df.to_csv(out_t1c, index=False)

    pos = int(main_df.label.sum())
    print(f"  {task}: total={len(main_df)}, positive={pos}, negative={len(main_df) - pos}")
    for p in (out_main, out_labels, out_flair, out_t1c):
        print(f"    {p}")


def main() -> None:
    csvs_dir = project_root / "data" / "csvs"
    data_root = project_root / "data" / "UTSW-Glioma"

    tsv_path = csvs_dir / "UTSW_Glioma_Metadata-2-1.tsv"
    df = pd.read_csv(tsv_path, sep="\t")

    # Defensive whitespace strip on string columns (CRLF line endings in the
    # source file are handled by pandas, but string values may still carry
    # trailing spaces in some cells).
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["pat_id"] = df["Subject ID"]

    # ---- Age / grade -------------------------------------------------------
    df["age"] = pd.to_numeric(df["Age at Imaging"], errors="coerce")
    df = df.dropna(subset=["age"]).copy()
    df["age"] = df["age"].round().astype(int)

    df["grade"] = pd.to_numeric(df["Tumor Grade"], errors="coerce").fillna(-1).astype(int)

    # ---- Imaging completeness on disk -------------------------------------
    if data_root.exists():
        keep = []
        missing = []
        for pid in df["pat_id"]:
            pdir = data_root / pid
            if pdir.exists() and has_complete_imaging(pdir):
                keep.append(pid)
            else:
                missing.append(pid)
        df = df[df["pat_id"].isin(keep)].copy()
        if missing:
            print(f"Dropped {len(missing)} patients with missing imaging or seg "
                  f"(first 5: {missing[:5]})")
    else:
        print(f"[WARN] {data_root} not found — skipping imaging completeness check.")

    # ---- IDH ---------------------------------------------------------------
    print("\n=== IDH MAIN csvs ===")
    idh_map = {"wild type": 0, "wildtype": 0, "mutated": 1}
    idh_df = df.copy()
    idh_df["idh_norm"] = idh_df["IDH"].str.lower()
    idh_df = idh_df[idh_df["idh_norm"].isin(idh_map.keys())].copy()
    idh_df["label"] = idh_df["idh_norm"].map(idh_map).astype(int)
    idh_df = idh_df.sort_values("pat_id").reset_index(drop=True)
    _write_task_csvs(idh_df, csvs_dir, task="IDH")


if __name__ == "__main__":
    main()
