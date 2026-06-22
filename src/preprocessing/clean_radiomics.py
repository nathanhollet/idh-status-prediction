#!/usr/bin/env python3

from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.utils.paths import (
    project_root, cfg, get_dataset_cfg,
    task_main_key, task_radiomics_key, DEFAULT_TASK,
)


def remove_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PyRadiomics diagnostics_* columns."""
    return df.loc[:, ~df.columns.str.startswith("diagnostics")]  # type: ignore[no-any-return]


def split_modality(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return rows whose ``Modality`` column matches ``modality`` exactly."""
    return df[df["Modality"] == modality].copy()


def add_modality_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Append modality suffix to feature columns and drop Modality column."""
    id_cols = ["PatientID", "Modality"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    rename_dict = {c: f"{c}_{suffix}" for c in feature_cols}
    df = df.rename(columns=rename_dict)
    return df.drop(columns=["Modality"])


def merge_modalities(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Inner-join per-modality feature tables on PatientID."""
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge(out, f, on="PatientID", how="inner")
    return out


def radiomics_t1_suffix(dataset: str) -> str:
    """Suffix used for T1 radiomics columns of ``dataset``.

    Datasets name their post-contrast T1 differently on disk
    (``T1c`` / ``T1GD``), and radiomics feature columns inherit that
    suffix. A dataset cfg can override via ``t1_radiomics_suffix`` to
    force an alias so columns align across cohorts at external-test
    time. Otherwise we fall back to the dataset's ``t1_name``.
    """
    ds_cfg = get_dataset_cfg(dataset)
    return ds_cfg.get("t1_radiomics_suffix") or ds_cfg.t1_name


def align_t1_suffix(
    df: pd.DataFrame,
    from_suffix: str,
    to_suffix: str,
) -> pd.DataFrame:
    """Rename an external cohort's T1-suffixed radiomics columns to match
    the training cohort's suffix. Cohorts name their post-contrast T1
    sequences differently on disk (T1c / T1GD / CT1), and radiomics
    columns inherit that suffix — so a model trained on UCSF
    (``_T1c``) can't be applied to ERASMUS (``_T1GD``) without first
    renaming. FLAIR columns are untouched because every cohort uses
    ``_FLAIR``.
    """
    if from_suffix == to_suffix:
        return df
    src = f"_{from_suffix}"
    dst = f"_{to_suffix}"
    rename = {c: c[: -len(src)] + dst for c in df.columns if c.endswith(src)}
    return df.rename(columns=rename)


def fit_feature_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.9,
) -> list[str]:
    """Fit feature selection on training data only and return kept columns.

    Drops:
      1. Features with <= 1 unique (non-NaN) value in the training set.
      2. One of each pair with |corr| > threshold in the training set,
         keeping the feature with higher |corr| to the training label.
    """
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    constant = [c for c in numeric_cols if X_train[c].nunique(dropna=True) <= 1]
    remaining = [c for c in numeric_cols if c not in constant]
    print(f"Constant features removed: {len(constant)}")

    corr_matrix = X_train[remaining].corr().abs()
    label_corr = X_train[remaining].corrwith(y_train).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop: set[str] = set()
    for col in upper.columns:
        for row in upper.index[upper[col] > corr_threshold]:
            if label_corr[col] < label_corr[row]:
                drop.add(col)
            else:
                drop.add(row)

    print(f"Highly correlated features removed: {len(drop)}")
    return [c for c in remaining if c not in drop]


def merge_radiomics_table(
    input_csv: Path,
    label_csv: Path,
    t1_suffix: str,
    modalities: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Merge modalities and labels into a single feature table.

    ``modalities`` is an ordered list of ``(modality_in_csv, column_suffix)``
    pairs. Defaults to ``[("FLAIR", "FLAIR"), (t1_suffix, t1_suffix)]`` so
    existing 2-modality calls keep their column names unchanged. For
    multi-modality runs the caller passes the full list
    (e.g. ``[("FLAIR", "FLAIR"), ("T1c", "T1c"), ("T1", "T1"), ("T2", "T2")]``).

    This performs only split-agnostic operations (diagnostics removal,
    modality merge, label join). Feature selection (constant /
    correlated feature removal) must be done per-fold on training data
    only — see ``fit_feature_selector``.
    """
    if modalities is None:
        modalities = [("FLAIR", "FLAIR"), (t1_suffix, t1_suffix)]

    print(f"Loading radiomics features from: {input_csv}")
    df = pd.read_csv(input_csv)
    print("Initial shape:", df.shape)

    df = remove_diagnostics(df)
    print("After removing diagnostics:", df.shape)

    frames = []
    for src_modality, suffix in modalities:
        sub = split_modality(df, src_modality)
        print(f"{src_modality} rows: {sub.shape[0]}")
        if sub.empty:
            raise ValueError(
                f"No rows with Modality == {src_modality!r} in {input_csv}. "
                f"Did you run extract_radiomics.py with --sequences {src_modality}?"
            )
        frames.append(add_modality_suffix(sub, suffix))
    combined = merge_modalities(frames)
    print("After merging modalities:", combined.shape)

    print(f"Loading labels from: {label_csv}")
    labels_df = pd.read_csv(label_csv)[["pat_id", "label"]]

    # PatientID written by extract_radiomics is already the canonical id
    # (matching the labels CSV's pat_id), so no string surgery is needed.
    combined["PatientID"] = combined["PatientID"].astype(str)
    combined = combined.merge(labels_df, left_on="PatientID", right_on="pat_id", how="inner")
    combined = combined.drop(columns=["pat_id"])
    print("After adding labels:", combined.shape)

    # Reorder: PatientID, label, then features
    rest = [c for c in combined.columns if c not in ["PatientID", "label"]]
    combined = combined[["PatientID", "label"] + rest]
    print("Final shape:", combined.shape)
    return combined


def resolve_paths(
    dataset: str,
    input_csv: str | None,
    label_csv: str | None,
    output_csv: str | None,
    task: str = DEFAULT_TASK,
) -> tuple[Path, Path, Path]:
    """Resolve input, label, and output paths from config / arguments."""
    ds_prefix = dataset.split("-")[0]

    input_path = (
        Path(input_csv) if input_csv
        else project_root / cfg.paths.data.csvs[f"RADIOMICS-{ds_prefix}-RAW"]
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Input radiomics CSV not found: {input_path}")

    # Labels come from the task-specific MAIN csv (pat_id, label, ...),
    # so tasks pull from their own label tables while sharing the same
    # raw radiomics features.
    label_path = (
        Path(label_csv) if label_csv
        else project_root / cfg.paths.data.csvs[task_main_key(dataset, task)]
    )
    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")

    output_path = (
        Path(output_csv) if output_csv
        else project_root / cfg.paths.data.csvs[task_radiomics_key(ds_prefix, task)]
    )
    return input_path, label_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PyRadiomics CSVs (FLAIR + T1 + labels). "
                    "Feature selection is intentionally NOT done here — it must "
                    "be applied per-fold on training data only."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. UCSF-PDGM, ERASMUS-GBM)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Prediction target (default: IDH). Raw radiomics "
                             "features are task-agnostic; only the label join "
                             "and output path differ.")
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Override raw radiomics CSV path "
                             "(defaults to cfg.paths.data.csvs[RADIOMICS-<prefix>-RAW])")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Override label CSV path "
                             "(defaults to the task's MAIN csv)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Override merged CSV output path "
                             "(defaults to cfg.paths.data.csvs[RADIOMICS-<prefix>(-<task>)])")
    parser.add_argument("--extra-sequences", type=str, default="",
                        help="Comma-separated extra modalities to include "
                             "alongside FLAIR + T1 (e.g. 'T1,T2'). Each "
                             "extra must already be present in the raw "
                             "radiomics CSV's Modality column (extract with "
                             "extract_radiomics.py --sequences T1 T2). The "
                             "default output path is replaced with a "
                             "``_merged_full.csv`` variant so existing "
                             "``_merged.csv`` pipelines stay untouched.")
    args = parser.parse_args()

    ds_cfg = get_dataset_cfg(args.dataset)
    extra_sequences: list[str] = [s for s in args.extra_sequences.split(",") if s]
    input_path, label_path, output_path = resolve_paths(
        args.dataset, args.input_csv, args.label_csv, args.output_csv, args.task
    )

    # ``t1_radiomics_suffix`` lets a dataset alias its post-contrast T1
    # column name in the merged feature table (e.g. force a shared
    # suffix across cohorts so columns align for cross-cohort scoring).
    # Defaults to ``t1_name`` so existing configs keep their column names.
    t1_suffix = ds_cfg.get("t1_radiomics_suffix") or ds_cfg.t1_name

    modalities = [("FLAIR", "FLAIR"), (t1_suffix, t1_suffix)]
    modalities += [(seq, seq) for seq in extra_sequences]

    if extra_sequences and args.output_csv is None:
        # Don't clobber the existing 2-modality _merged.csv — write a
        # parallel _merged_full.csv so the existing TabPFN / LogReg
        # pipelines keep their inputs unchanged.
        output_path = output_path.with_name(
            output_path.stem.replace("_merged", "_merged_full") + output_path.suffix
        )

    merged = merge_radiomics_table(
        input_csv=input_path,
        label_csv=label_path,
        t1_suffix=t1_suffix,
        modalities=modalities,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nMerged CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
