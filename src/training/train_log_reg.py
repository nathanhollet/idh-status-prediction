import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

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


C_VALUES = np.logspace(-4, 2, 7)
PENALTIES = ["l1", "l2"]


def train_eval_model(X_train, y_train, X_val, y_val, C, penalty, seed):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty=penalty, C=C, solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=seed
        ))
    ])
    pipeline.fit(X_train, y_train)
    score = average_precision_score(y_val, pipeline.predict_proba(X_val)[:, 1])
    return score, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Prediction target (default: IDH).")
    parser.add_argument("--external-test", type=str, nargs="+", default=None,
                        help="One or more external radiomics cohorts to evaluate "
                             "each run's fitted pipeline on (e.g. 'UCSD-PTGBM'). "
                             "Each external gets its own "
                             "<out_dir>/external_<DATASET>/ subdir. External "
                             "CSVs must share the same feature columns (post "
                             "T1-suffix alignment).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to radiomics CSV (defaults to config). "
                             "Point at ``*_merged_full.csv`` for multi-modality "
                             "pilots.")
    parser.add_argument("--external_data_paths", type=str, nargs="+", default=None,
                        help="Optional explicit radiomics CSV path per external "
                             "cohort (must match --external-test in length and "
                             "order). Use this when piloting multi-modality "
                             "merged_full.csv files so the external lookup "
                             "doesn't fall back to the config-default "
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
    split_dir = project_root / cfg.paths.data.csvs[task_splits_key(ds_prefix, args.task)]
    cohort_dirname = args.dataset + args.output_suffix
    out_dir = task_results_subdir(
        project_root / cfg.paths.results / "radiomics" / cohort_dirname, args.task
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_csv(data_path)

    ext_datasets = args.external_test or []
    if args.external_data_paths is not None and len(args.external_data_paths) != len(ext_datasets):
        raise ValueError(
            f"--external_data_paths has {len(args.external_data_paths)} entries "
            f"but --external-test has {len(ext_datasets)}; they must match."
        )

    # Optional external test cohorts (e.g. UCSD-PTGBM for IDH).
    externals: list[dict] = []
    for i, ext_dataset in enumerate(ext_datasets):
        ext_prefix = ext_dataset.split("-")[0]
        if args.external_data_paths is not None:
            ext_path = Path(args.external_data_paths[i])
        else:
            ext_path = project_root / cfg.paths.data.csvs[task_radiomics_key(ext_prefix, args.task)]
        ext_df = pd.read_csv(ext_path)
        # Align the external cohort's T1 suffix to the training cohort's
        # (e.g. ERASMUS ``_T1GD`` → UCSF ``_T1c``) so selected_features
        # lookups hit the same column names.
        ext_df = align_t1_suffix(
            ext_df,
            from_suffix=radiomics_t1_suffix(ext_dataset),
            to_suffix=radiomics_t1_suffix(args.dataset),
        )
        ext_runner = ExternalTestRunner(
            dataset=ext_dataset,
            out_dir=out_dir / f"external_{ext_dataset}",
            metrics_filename="results_per_run.csv",
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
    print(f"Found {len(split_paths)} split files.")

    seeds = cfg.training.seeds
    assert len(seeds) == len(split_paths), \
        f"seeds ({len(seeds)}) and split files ({len(split_paths)}) must match"

    all_results, all_predictions, all_selected_features = [], [], []

    for run, (seed, split_path) in enumerate(zip(seeds, split_paths)):
        print(f"\n===== RUN {run} | seed {seed} =====")
        set_seed(seed)

        train_df, val_df, test_df = load_split(split_path, df_base, id_col="PatientID")

        X_train = train_df.drop(columns=["PatientID", "label"])
        y_train = train_df["label"]
        X_val   = val_df.drop(columns=["PatientID", "label"])
        y_val   = val_df["label"]
        X_test  = test_df.drop(columns=["PatientID", "label"])
        y_test  = test_df["label"]

        # Fit feature selector on training data ONLY, then apply to all splits
        selected_features = fit_feature_selector(X_train, y_train, corr_threshold=cfg.training.corr_threshold)
        X_train = X_train[selected_features]
        X_val   = X_val[selected_features]
        X_test  = X_test[selected_features]

        feature_names = selected_features

        best_score, best_model, best_params = -np.inf, None, None

        for C in C_VALUES:
            for penalty in PENALTIES:
                try:
                    score, model = train_eval_model(X_train, y_train, X_val, y_val, C, penalty, seed)
                    if score > best_score:
                        best_score, best_model, best_params = score, model, (C, penalty)
                except Exception as e:
                    print(f"  Failed C={C}, penalty={penalty}: {e}")
                    continue

        print(f"Best params: {best_params} | Val AUPRC: {best_score:.4f}")

        coef = best_model.named_steps["logreg"].coef_[0]
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
        selected = coef_df[coef_df["coefficient"] != 0].sort_values("coefficient", key=abs, ascending=False)
        selected["run"] = run
        all_selected_features.append(selected)

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        all_predictions.append(pd.DataFrame({
            "PatientID": test_df["PatientID"].values,
            "run": run, "true_label": y_test.values,
            "pred_prob": y_prob, "pred_label": y_pred,
            "best_C": float(best_params[0]), "best_penalty": best_params[1]
        }))

        all_results.append({
            "run": run, "best_C": float(best_params[0]), "best_penalty": best_params[1],
            "AUROC": roc_auc_score(y_test, y_prob),
            "AUPRC": average_precision_score(y_test, y_prob),
            "F1": f1_score(y_test, y_pred),
            "BalancedAccuracy": balanced_accuracy_score(y_test, y_pred)
        })

        # ---- external test: apply the same feature selection then score
        for external in externals:
            ext_X = external["df"].drop(columns=["PatientID", "label"])
            missing = [f for f in selected_features if f not in ext_X.columns]
            if missing:
                raise KeyError(
                    f"External cohort {external['dataset']} is missing "
                    f"{len(missing)} features selected by run {run}. "
                    f"First few: {missing[:5]}"
                )
            ext_X = ext_X[selected_features]
            ext_y = external["df"]["label"]
            ext_prob = best_model.predict_proba(ext_X)[:, 1]
            ext_pred = best_model.predict(ext_X)

            external["runner"].record(
                run,
                {
                    "AUROC": roc_auc_score(ext_y, ext_prob),
                    "AUPRC": average_precision_score(ext_y, ext_prob),
                    "F1": f1_score(ext_y, ext_pred),
                    "BalancedAccuracy": balanced_accuracy_score(ext_y, ext_pred),
                    **recalibrated_threshold_metrics(ext_y, ext_prob),
                },
                pd.DataFrame({
                    "PatientID": external["df"]["PatientID"].values,
                    "run": run,
                    "true_label": ext_y.values,
                    "pred_prob": ext_prob,
                    "pred_label": ext_pred,
                }),
            )

    pd.concat(all_predictions).to_csv(out_dir / "predictions_per_patient.csv", index=False)
    pd.concat(all_selected_features).to_csv(out_dir / "selected_features.csv", index=False)
    pd.concat(all_selected_features)["feature"].value_counts().to_csv(out_dir / "feature_selection_frequency.csv")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "results_per_run.csv", index=False)
    results_df[["AUROC","AUPRC","F1","BalancedAccuracy"]].agg(["mean","std"]).to_csv(out_dir / "results_summary.csv")

    print("\nFinal Results (mean ± std):")
    print(results_df[["AUROC","AUPRC","F1","BalancedAccuracy"]].agg(["mean","std"]))

    for external in externals:
        external["runner"].finalize([
            "AUROC", "AUPRC", "F1", "BalancedAccuracy",
            "F1_recal", "BalancedAccuracy_recal",
        ])


if __name__ == "__main__":
    main()