from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def load_split(
    split_path: Path,
    df: pd.DataFrame,
    id_col: str = "pat_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a canonical split CSV and partition ``df`` into (train, val, test).

    The split CSV is expected to have at least the columns ``pat_id`` and
    ``split``. Patients in ``df`` but missing from the split file (or vice
    versa) are silently dropped via inner join — caller code that needs to
    detect this should check the returned sizes against ``len(df)``.

    Args:
        split_path: Path to a ``splits_run{i}.csv`` file.
        df: DataFrame to partition. Must contain ``id_col``.
        id_col: Name of the patient-id column in ``df``. Defaults to
            ``pat_id``; pass ``"PatientID"`` for radiomics tables.
    """
    split_df = pd.read_csv(split_path)[["pat_id", "split"]]
    merged = df.merge(split_df, left_on=id_col, right_on="pat_id", how="inner")
    if id_col != "pat_id":
        merged = merged.drop(columns=["pat_id"])

    return (
        merged[merged.split == "train"].drop(columns="split"),
        merged[merged.split == "val"].drop(columns="split"),
        merged[merged.split == "test"].drop(columns="split"),
    )


def stratified_split(df: pd.DataFrame, seed: int, train_size=0.7, val_size=0.15, test_size=0.15):
    df = df.copy()

    df["gradeU"] = (df.grade == -1).astype(int) # If no -1 then all 0s, so it won't affect the stratification
    df["grade2"] = (df.grade == 2).astype(int)
    df["grade3"] = (df.grade == 3).astype(int)
    df["grade4"] = (df.grade == 4).astype(int)

    # MultilabelStratifiedShuffleSplit expects BINARY multilabels — a
    # continuous ``age_norm`` column would be treated as one distinct
    # label per unique value (useless). Bin age into tertiles so each
    # fold has a similar young/middle/old mix.
    age_bins = pd.qcut(df.age, q=3, labels=False, duplicates="drop")
    df["age_young"] = (age_bins == 0).astype(int)
    df["age_mid"] = (age_bins == 1).astype(int)
    df["age_old"] = (age_bins == 2).astype(int)

    Y = df[["label", "gradeU", "grade2", "grade3", "grade4",
            "age_young", "age_mid", "age_old"]].values
 
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - train_size),
        random_state=seed
    )
    train_idx, temp_idx = next(msss.split(df, Y))
 
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    Y_temp = temp_df[["label", "gradeU", "grade2", "grade3", "grade4",
                      "age_young", "age_mid", "age_old"]].values
 
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size / (test_size + val_size),
        random_state=seed
    )
    val_idx, test_idx = next(msss2.split(temp_df, Y_temp))
 
    return train_df, temp_df.iloc[val_idx], temp_df.iloc[test_idx]