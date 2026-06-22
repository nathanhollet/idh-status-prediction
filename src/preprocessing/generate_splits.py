#!/usr/bin/env python3
"""Generate canonical train/val/test split CSVs for a dataset.

Reads the MAIN labels CSV and writes one ``splits_run{i}.csv`` per seed
in ``cfg.training.seeds``. These files are the single source of truth
for splits across all training scripts (linear probe, logistic
regression, TabPFN).

Re-run this script whenever:
  * the patient cohort in the MAIN CSV changes,
  * ``cfg.training.seeds`` changes,
  * ``stratified_split()`` logic changes.
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.utils.paths import (
    project_root, cfg, task_main_key, task_splits_key, DEFAULT_TASK,
)
from src.utils.splits import stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. UCSF-PDGM, ERASMUS-GBM)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Prediction target (default: IDH).")
    args = parser.parse_args()

    main_csv = project_root / cfg.paths.data.csvs[task_main_key(args.dataset, args.task)]
    df = pd.read_csv(main_csv)
    print(f"Loaded {len(df)} patients from {main_csv}")

    ds_prefix = args.dataset.split("-")[0]
    out_dir = project_root / cfg.paths.data.csvs[task_splits_key(ds_prefix, args.task)]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_cols = ["pat_id", "label", "grade", "age", "split"]
    seeds = cfg.training.seeds

    for run, seed in enumerate(seeds):
        train_df, val_df, test_df = stratified_split(df, seed)
        split_df = pd.concat([
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ])[out_cols]
        out_path = out_dir / f"splits_run{run}.csv"
        split_df.to_csv(out_path, index=False)
        print(
            f"  run {run} (seed {seed}): "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()