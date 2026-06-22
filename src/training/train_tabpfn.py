#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, balanced_accuracy_score,
)

import tabpfn_client
from tabpfn_client import TabPFNClassifier

from src.utils.paths import (
    project_root, cfg,
    task_radiomics_key, task_splits_key, task_results_subdir, DEFAULT_TASK,
)
from src.utils.reproducibility import set_seed
from src.utils.splits import load_split
from src.utils.external_test import ExternalTestRunner
from src.utils.metrics import recalibrated_threshold_metrics
from src.preprocessing.clean_radiomics import (
    align_t1_suffix, fit_feature_selector, radiomics_t1_suffix,
)


METRIC_NAMES = ["AUROC", "AUPRC", "F1", "BalancedAccuracy"]


def evaluate(clf, X, y) -> tuple[dict, list]:
    """Compute classification metrics for a fitted classifier and return the
    per-row positive-class probabilities alongside the metric dict, so callers
    can persist predictions (e.g. for calibration analysis) without paying for
    a second cloud-inference call."""
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    metrics = {
        "AUROC": roc_auc_score(y, y_proba),
        "AUPRC": average_precision_score(y, y_proba),
        "F1": f1_score(y, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y, y_pred),
    }
    return metrics, y_proba


def load_split_data(
    df_base: pd.DataFrame,
    split_path: Path,
    corr_threshold: float,
) -> tuple[dict, dict, list[str]]:
    """Load a canonical split file and return ``(data, ids, selected)``.

    ``data[split] = (X, y)`` after feature selection; ``ids[split]`` is the
    matching PatientID array (preserved so the caller can write per-patient
    predictions). Feature selection is fitted on the training split only and
    then applied to val/test, to avoid leakage.
    """
    train_df, val_df, test_df = load_split(split_path, df_base, id_col="PatientID")

    data, ids = {}, {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = split_df.drop(columns=["PatientID", "label"])
        y = split_df["label"]
        data[name] = (X, y)
        ids[name] = split_df["PatientID"].values

    selected = fit_feature_selector(*data["train"], corr_threshold=corr_threshold)
    data = {name: (X[selected], y) for name, (X, y) in data.items()}
    return data, ids, selected


def save_and_print_results(metrics: dict, output_dir: Path) -> None:
    """Save per-split and summary CSVs, print mean +/- std."""
    for split_name, metrics_list in metrics.items():
        df = pd.DataFrame(metrics_list)
        df["split"] = [f"run{i}" for i in range(len(df))]
        df.to_csv(output_dir / f"tabpfn_{split_name}_metrics_per_split.csv", index=False)

        summary = pd.DataFrame({
            "metric": METRIC_NAMES,
            "mean": [df[c].mean() for c in METRIC_NAMES],
            "std": [df[c].std() for c in METRIC_NAMES],
        })
        summary.to_csv(output_dir / f"tabpfn_{split_name}_metrics_summary.csv", index=False)

        print(f"\n{'=' * 30}")
        print(f"{split_name.upper()} RESULTS (mean ± std)")
        print("=" * 30)
        for c in METRIC_NAMES:
            print(f"{c}: {df[c].mean():.3f} ± {df[c].std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN on radiomics features")
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name,
                        help="Dataset name (e.g. UCSF-PDGM, ERASMUS-GBM)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Prediction target (default: IDH).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to radiomics CSV (defaults to config)")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory with split CSVs (defaults to config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to config)")
    parser.add_argument("--external-test", type=str, nargs="+", default=None,
                        help="One or more external radiomics cohorts to "
                             "evaluate each run's fitted classifier on "
                             "(e.g. 'UCSD-PTGBM'). Each external "
                             "gets its own <out_dir>/external_<DATASET>/ subdir.")
    parser.add_argument("--external_data_paths", type=str, nargs="+", default=None,
                        help="Optional explicit radiomics CSV path per external "
                             "cohort (must match --external-test in length and "
                             "order). Use this when piloting multi-modality "
                             "merged_full.csv files so the external lookup "
                             "doesn't silently fall back to the config-default "
                             "2-modality merged.csv.")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to the cohort segment "
                             "of the results directory (e.g. '_T1_T2'). Lets "
                             "pilot runs coexist with existing results.")
    args = parser.parse_args()

    ds_prefix = args.dataset.split("-")[0]
    data_path = (
        Path(args.data_path) if args.data_path
        else project_root / cfg.paths.data.csvs[task_radiomics_key(ds_prefix, args.task)]
    )
    split_dir = (
        Path(args.split_dir) if args.split_dir
        else project_root / cfg.paths.data.csvs[task_splits_key(ds_prefix, args.task)]
    )
    cohort_dirname = args.dataset + args.output_suffix
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else task_results_subdir(
            project_root / cfg.paths.results / "tabpfn" / cohort_dirname, args.task
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        raise RuntimeError("Set the TABPFN_TOKEN environment variable")
    tabpfn_client.set_access_token(token)

    df_base = pd.read_csv(data_path)

    ext_datasets = args.external_test or []
    if args.external_data_paths is not None and len(args.external_data_paths) != len(ext_datasets):
        raise ValueError(
            f"--external_data_paths has {len(args.external_data_paths)} entries "
            f"but --external-test has {len(ext_datasets)}; they must match."
        )

    # Optional external test cohorts.
    externals: list[dict] = []
    for i, ext_dataset in enumerate(ext_datasets):
        ext_prefix = ext_dataset.split("-")[0]
        if args.external_data_paths is not None:
            ext_path = Path(args.external_data_paths[i])
        else:
            ext_path = project_root / cfg.paths.data.csvs[task_radiomics_key(ext_prefix, args.task)]
        ext_df = pd.read_csv(ext_path)
        ext_df = align_t1_suffix(
            ext_df,
            from_suffix=radiomics_t1_suffix(ext_dataset),
            to_suffix=radiomics_t1_suffix(args.dataset),
        )
        ext_runner = ExternalTestRunner(
            dataset=ext_dataset,
            out_dir=output_dir / f"external_{ext_dataset}",
            metrics_filename="tabpfn_test_metrics_per_split.csv",
            predictions_filename="predictions_per_patient.csv",
        )
        externals.append({"dataset": ext_dataset, "df": ext_df, "runner": ext_runner})
        print(f"External test set: {ext_dataset} (N={len(ext_df)}) → {ext_runner.out_dir}")

    split_paths = sorted(split_dir.glob("splits_run*.csv"))
    if not split_paths:
        raise FileNotFoundError(
            f"No split files found in {split_dir}. "
            f"Run: python src/preprocessing/generate_splits.py --dataset {args.dataset}"
        )
    print(f"Found {len(split_paths)} split files")

    seeds = cfg.training.seeds
    assert len(seeds) == len(split_paths), \
        f"seeds ({len(seeds)}) and split files ({len(split_paths)}) must match"

    metrics = {"test": [], "val": []}
    all_test_predictions: list[pd.DataFrame] = []

    for run, (seed, split_path) in enumerate(zip(seeds, split_paths)):
        print(f"\n===== RUN {run} | seed {seed} =====")
        set_seed(seed)
        data, ids, selected = load_split_data(
            df_base, split_path, corr_threshold=cfg.training.corr_threshold
        )

        clf = TabPFNClassifier()
        clf.fit(*data["train"])

        for split_name in ["test", "val"]:
            result, proba = evaluate(clf, *data[split_name])
            metrics[split_name].append(result)
            if split_name == "test":
                all_test_predictions.append(pd.DataFrame({
                    "PatientID": ids["test"],
                    "run": run,
                    "true_label": data["test"][1].values,
                    "pred_prob": proba,
                }))

        print(f"Test ROC-AUC: {metrics['test'][-1]['AUROC']:.4f}")
        print(f"Test AUPRC: {metrics['test'][-1]['AUPRC']:.4f}")

        for external in externals:
            ext_X = external["df"].drop(columns=["PatientID", "label"])
            missing = [f for f in selected if f not in ext_X.columns]
            if missing:
                raise KeyError(
                    f"External cohort {external['dataset']} is missing "
                    f"{len(missing)} features selected by run {run}. "
                    f"First few: {missing[:5]}"
                )
            ext_X = ext_X[selected]
            ext_y = external["df"]["label"]
            ext_result, ext_proba = evaluate(clf, ext_X, ext_y)
            ext_pred = clf.predict(ext_X)
            ext_result.update(recalibrated_threshold_metrics(ext_y, ext_proba))
            external["runner"].record(
                run, ext_result,
                pd.DataFrame({
                    "PatientID": external["df"]["PatientID"].values,
                    "run": run,
                    "true_label": ext_y.values,
                    "pred_prob": ext_proba,
                    "pred_label": ext_pred,
                }),
            )

    save_and_print_results(metrics, output_dir)
    if all_test_predictions:
        pd.concat(all_test_predictions, ignore_index=True).to_csv(
            output_dir / "predictions_per_patient.csv", index=False
        )

    for external in externals:
        external["runner"].finalize(METRIC_NAMES + ["F1_recal", "BalancedAccuracy_recal"])


if __name__ == "__main__":
    main()
