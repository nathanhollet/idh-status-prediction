#!/usr/bin/env python3
"""Assemble UPENN-GBM ``pat_id, age, grade, label`` MAIN CSV.

Reads:
  * ``data/csvs/UPENN_IDH_labels.csv``  (pat_id, label=IDH)
  * ``data/csvs/UPENN-GBM_clinical_info_v2.1_original.csv`` (age + ...)

Writes:
  * ``data/csvs/UPENN_IDH_age_grade_labels.csv``

Notes
-----
* Grade is set to 4 for every patient: UPENN-GBM is a glioblastoma cohort.
  This makes the grade columns constant in ``stratified_split`` (no
  contribution to stratification), which is the desired behavior.
* Patients in the labels CSV but missing from the clinical info (no age)
  are dropped with a warning.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.utils.paths import project_root


def build_main_csv(labels_csv: Path, clinical: pd.DataFrame, out_csv: Path) -> None:
    """Merge per-task labels with clinical info and write the MAIN csv."""
    labels = pd.read_csv(labels_csv)

    df = labels.merge(
        clinical[["ID", "Age_at_scan_years"]],
        left_on="pat_id", right_on="ID", how="left",
    ).drop(columns=["ID"])

    df = df.rename(columns={"Age_at_scan_years": "age"})
    df["grade"] = 4  # all UPENN-GBM cases are glioblastoma

    before = len(df)
    df = df.dropna(subset=["age"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} patients without age in clinical info.")

    df["age"] = df["age"].astype(float).round().astype(int)
    df["label"] = df["label"].astype(int)
    df = df[["pat_id", "age", "grade", "label"]]

    df.to_csv(out_csv, index=False)
    print(df.head())
    print(
        f"  Total: {len(df)}, Positive: {df.label.sum()}, "
        f"Negative: {(df.label == 0).sum()} -> {out_csv}\n"
    )


def main() -> None:
    csvs = project_root / "data" / "csvs"
    clinical = pd.read_csv(csvs / "UPENN-GBM_clinical_info_v2.1_original.csv")

    print("=== IDH MAIN csv ===")
    build_main_csv(
        labels_csv=csvs / "UPENN_IDH_labels.csv",
        clinical=clinical,
        out_csv=csvs / "UPENN_IDH_age_grade_labels.csv",
    )


if __name__ == "__main__":
    main()
