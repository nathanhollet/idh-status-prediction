#!/usr/bin/env python3
"""Assemble UCSD-PTGBM IDH test-cohort CSVs from the raw TCIA metadata.

Reads:
  * ``data/csvs/ucsd_ptgbm.csv``

Writes:
  * ``data/csvs/UCSD_PTGBM_IDH_labels.csv``            (pat_id, label)
  * ``data/csvs/UCSD_PTGBM_IDH_age_grade_labels.csv``  (pat_id, age, grade, label)
  * ``data/csvs/UCSD_PTGBM_IDH_labels_FLAIR.csv``      (pat_id, label) — BrainIAC loader
  * ``data/csvs/UCSD_PTGBM_IDH_labels_T1c.csv``        (pat_id, label) — BrainIAC loader

Scope
-----
UCSD-PTGBM is post-operative GBM and will only be used as an external IDH
test cohort. To avoid patient leakage, we keep exactly one timepoint per
patient (the earliest, i.e. ``_01`` when available). Rows with unknown
IDH are dropped; label encoding matches the rest of the project
(0 = wildtype, 1 = mutated).

Patient IDs in the source CSV look like ``UCSD-PTGBM-0001_01``; the
``_NN`` suffix is the timepoint. We keep the full timepoint-level ID as
``pat_id`` so it still matches the folder naming from TCIA downloads.
Per-sequence CSVs append ``_FLAIR`` / ``_T1c`` to match the BrainIAC
loader convention (``<id>_<seq>.nii.gz``).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.utils.paths import project_root


IDH_MAP = {"wild type": 0, "wildtype": 0, "mutant": 1, "mutated": 1}

REQUIRED_SUFFIXES = ("_FLAIR.nii.gz", "_T1post.nii.gz", "_BraTS_tumor_seg.nii.gz")


def has_complete_imaging(patient_dir: Path, base_id: str) -> bool:
    return all((patient_dir / f"{base_id}{suf}").exists() for suf in REQUIRED_SUFFIXES)


def _write_per_sequence_csvs(labels_df: pd.DataFrame, csvs_dir: Path) -> None:
    out_flair = csvs_dir / "UCSD_PTGBM_IDH_labels_FLAIR.csv"
    out_t1c = csvs_dir / "UCSD_PTGBM_IDH_labels_T1c.csv"

    flair_df = labels_df.copy()
    flair_df["pat_id"] = flair_df["pat_id"] + "_FLAIR"
    flair_df.to_csv(out_flair, index=False)

    t1c_df = labels_df.copy()
    t1c_df["pat_id"] = t1c_df["pat_id"] + "_T1c"
    t1c_df.to_csv(out_t1c, index=False)

    print(f"    {out_flair}")
    print(f"    {out_t1c}")


def main() -> None:
    csvs_dir = project_root / "data" / "csvs"
    data_root = project_root / "data" / "UCSD-PTGBM"
    src_csv = csvs_dir / "ucsd_ptgbm.csv"

    df = pd.read_csv(src_csv)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ---- IDH label -----------------------------------------------------
    df["idh_norm"] = df["IDH"].str.lower()
    df = df[df["idh_norm"].isin(IDH_MAP.keys())].copy()
    df["label"] = df["idh_norm"].map(IDH_MAP).astype(int)

    # ---- One timepoint per patient (earliest) --------------------------
    df["pat_id"] = df["ID"]
    df["base_id"] = df["pat_id"].str.rsplit("_", n=1).str[0]
    df["timepoint"] = pd.to_numeric(
        df["pat_id"].str.rsplit("_", n=1).str[1], errors="coerce"
    ).fillna(1).astype(int)

    df = df.sort_values(["base_id", "timepoint"]).drop_duplicates(
        subset="base_id", keep="first"
    )

    # ---- Age / grade ---------------------------------------------------
    df["age"] = pd.to_numeric(df["Patient's Age"], errors="coerce")
    df = df.dropna(subset=["age"]).copy()
    df["age"] = df["age"].round().astype(int)
    df["grade"] = pd.to_numeric(df["Grade"], errors="coerce").fillna(-1).astype(int)

    # ---- Imaging completeness on disk (required files: FLAIR, T1post,
    #      BraTS seg). Skip the check if the data dir isn't present so
    #      the CSV can still be built on a machine without the cohort.
    if data_root.exists():
        missing = []
        keep = []
        for _, row in df.iterrows():
            pat_dir = data_root / row["pat_id"]
            if pat_dir.exists() and has_complete_imaging(pat_dir, row["pat_id"]):
                keep.append(row["pat_id"])
            else:
                missing.append(row["pat_id"])
        df = df[df["pat_id"].isin(keep)].copy()
        if missing:
            print(f"Dropped {len(missing)} patients with missing imaging "
                  f"(first 5: {missing[:5]})")
    else:
        print(f"[WARN] {data_root} not found — skipping imaging completeness check.")

    df = df.sort_values("pat_id").reset_index(drop=True)

    main_df = df[["pat_id", "age", "grade", "label"]]
    labels_df = df[["pat_id", "label"]]

    out_main = csvs_dir / "UCSD_PTGBM_IDH_age_grade_labels.csv"
    out_labels = csvs_dir / "UCSD_PTGBM_IDH_labels.csv"
    main_df.to_csv(out_main, index=False)
    labels_df.to_csv(out_labels, index=False)

    pos = int(main_df.label.sum())
    print(f"IDH: total={len(main_df)}, mutant={pos}, wildtype={len(main_df) - pos}")
    print(f"    {out_main}")
    print(f"    {out_labels}")
    _write_per_sequence_csvs(labels_df, csvs_dir)


if __name__ == "__main__":
    main()
