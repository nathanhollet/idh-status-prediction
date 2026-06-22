import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils.reproducibility import set_seed
from src.utils.splits import load_split
from src.utils.metrics import evaluate, recalibrated_threshold_metrics
from src.utils.external_test import ExternalTestRunner
from src.datasets.embedding_dataset import preload_embeddings, EmbeddingDataset
from src.models.linear_probe import LinearProbe
from src.utils.paths import (
    project_root, cfg, get_dataset_cfg,
    task_main_key, task_splits_key, task_results_subdir, DEFAULT_TASK,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(train_df, val_df, lr, args, embedding_cache, run):

    train_ds = EmbeddingDataset(train_df, embedding_cache)
    val_ds = EmbeddingDataset(val_df, embedding_cache)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)

    sample_x, _ = train_ds[0]
    model = LinearProbe(sample_x.numel()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = train_df.label.sum()
    neg = len(train_df) - pos
    # Clamp so tiny positive folds (e.g. UPENN IDH, ~10 positives / 360
    # negatives) don't blow up gradient magnitude by 30-50x and swamp
    # the signal from everything else.
    pos_weight_val = min(neg / (pos + 1e-6), 10.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()

    best_val_auprc, best_epoch, best_state = 0, 0, None
    patience_counter = 0
    history = []

    for epoch in range(args.num_epochs):

        model.train()
        train_probs, train_labels, train_loss = [], [], 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = train_criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

            probs = torch.sigmoid(out).detach().cpu().numpy()
            train_probs.extend(probs.flatten())
            train_labels.extend(y.cpu().numpy().flatten())

        train_loss = train_loss / len(train_labels)

        try:
            train_auprc = average_precision_score(train_labels, train_probs)
            train_auroc = roc_auc_score(train_labels, train_probs)
        except Exception:
            train_auprc = float("nan")
            train_auroc = float("nan")

        val_metrics = evaluate(model, val_loader, eval_criterion, device)

        history.append({
            "run": run,
            "learning_rate": lr,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auroc": train_auroc,
            "train_auprc": train_auprc,
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"]
        })

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.use_early_stopping and patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_epoch, history, eval_criterion


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--model", type=str, default=cfg.model.name)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Prediction target (default: IDH). Embeddings are "
                             "task-agnostic and reused across tasks; only the "
                             "label CSV + splits differ per task.")
    parser.add_argument("--external-test", type=str, nargs="+", default=None,
                        help="One or more external datasets to evaluate each "
                             "run's best model on (e.g. 'UCSD-PTGBM'). "
                             "Each external gets its own <out_dir>/external_<DATASET>/ "
                             "subdir. Every external must have its own MAIN csv "
                             "for the given task and pre-computed embeddings.")
    parser.add_argument("--extra-sequences", type=str, default="",
                        help="Comma-separated extra modalities to concatenate "
                             "onto the FLAIR + T1c embedding vector (e.g. 'T1,T2'). "
                             "Each must have pre-computed embeddings under "
                             "embeddings/<model>/pt/<dataset>/<sequence>/. Empty "
                             "(default) preserves the current 2-modality behavior.")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to the cohort segment "
                             "of the results directory (e.g. '_T1_T2' produces "
                             "results/models/<model>/<cohort>_T1_T2/...). Lets "
                             "pilot runs coexist with the existing 2-modality "
                             "results instead of overwriting them.")
    args = parser.parse_args()
    extra_sequences: list[str] = [s for s in args.extra_sequences.split(",") if s]

    args.num_epochs = cfg.training.num_epochs
    args.patience = cfg.training.patience
    args.use_early_stopping = cfg.training.early_stopping

    dataset_csv = project_root / cfg.paths.data.csvs[task_main_key(args.dataset, args.task)]

    emb_root = project_root / cfg.paths.embeddings[args.model] / "pt" / args.dataset
    flair_dir = emb_root / "FLAIR"
    t1c_seq = get_dataset_cfg(args.dataset).t1_name
    t1c_dir = emb_root / t1c_seq
    extra_dirs = [(seq, emb_root / seq) for seq in extra_sequences]

    df = pd.read_csv(dataset_csv)

    cohort_dirname = args.dataset + args.output_suffix
    out_dir = task_results_subdir(
        project_root / cfg.paths.results / args.model / cohort_dirname, args.task
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_cache = preload_embeddings(
        df, flair_dir, t1c_dir, t1c_seq=t1c_seq, extra_dirs=extra_dirs,
    )

    # -----------------------------------------------------------------------
    # Optional external test cohorts (e.g. UCSD-PTGBM). Each external is
    # loaded once up-front; after every run, the best model is scored
    # against each, producing per-run predictions + aggregate metrics
    # under external_<DATASET>/.
    # -----------------------------------------------------------------------
    externals: list[dict] = []
    for ext_dataset in args.external_test or []:
        ext_t1c_seq = get_dataset_cfg(ext_dataset).t1_name
        ext_emb_root = project_root / cfg.paths.embeddings[args.model] / "pt" / ext_dataset
        ext_flair_dir = ext_emb_root / "FLAIR"
        ext_t1c_dir = ext_emb_root / ext_t1c_seq
        ext_extra_dirs = [(seq, ext_emb_root / seq) for seq in extra_sequences]

        ext_main_csv = project_root / cfg.paths.data.csvs[task_main_key(ext_dataset, args.task)]
        ext_df = pd.read_csv(ext_main_csv)
        ext_cache = preload_embeddings(
            ext_df, ext_flair_dir, ext_t1c_dir, t1c_seq=ext_t1c_seq,
            extra_dirs=ext_extra_dirs,
        )

        ext_runner = ExternalTestRunner(
            dataset=ext_dataset,
            out_dir=out_dir / f"external_{ext_dataset}",
            metrics_filename="test_metrics_runs.csv",
            predictions_filename="test_predictions_external.csv",
            per_run_prediction_pattern="test_predictions_run{run}.csv",
        )
        externals.append({"dataset": ext_dataset, "df": ext_df, "cache": ext_cache, "runner": ext_runner})
        print(f"External test set: {ext_dataset} (N={len(ext_df)}) → {ext_runner.out_dir}")

    ds_prefix = args.dataset.split("-")[0]
    split_dir = project_root / cfg.paths.data.csvs[task_splits_key(ds_prefix, args.task)]
    split_paths = sorted(split_dir.glob("splits_run*.csv"))
    if not split_paths:
        raise FileNotFoundError(
            f"No split files found in {split_dir}. "
            f"Run: python src/preprocessing/generate_splits.py --dataset {args.dataset}"
        )

    seeds = cfg.training.seeds
    assert len(seeds) == len(split_paths), \
        f"seeds ({len(seeds)}) and split files ({len(split_paths)}) must match"

    lr_grid = cfg.training.lr_grid

    all_test_metrics, all_histories, split_stats_records = [], [], []

    for run, (seed, split_path) in enumerate(zip(seeds, split_paths)):

        print(f"\n===== RUN {run} | seed {seed} =====")
        set_seed(seed)

        train_df, val_df, test_df = load_split(split_path, df, id_col="pat_id")

        for split_name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
            grades = subset.grade.value_counts().reindex([2, 3, 4], fill_value=0)
            split_stats_records.append({
                "run": run,
                "split": split_name,
                "n": len(subset),
                "mutation_ratio": subset.label.mean(),
                "age_mean": subset.age.mean(),
                "grade2": grades[2],
                "grade3": grades[3],
                "grade4": grades[4]
            })

        best_lr, best_val_score, best_model, best_epoch, best_history, best_criterion = None, -np.inf, None, None, None, None

        for lr in lr_grid:
            set_seed(seed)  # Reset RNG so every LR sees the same init / shuffle order
            print(f"Testing LR = {lr}")
            model, epoch, history, criterion = train_model(train_df, val_df, lr, args, embedding_cache, run)
            val_score = max(h["val_auprc"] for h in history)

            if val_score > best_val_score:
                best_val_score = val_score
                best_lr = lr
                best_model = model
                best_epoch = epoch
                best_history = history
                best_criterion = criterion

        all_histories.extend(best_history)

        print(f"Best LR: {best_lr} | Best epoch: {best_epoch}")

        test_ds = EmbeddingDataset(test_df, embedding_cache)
        test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

        metrics = evaluate(best_model, test_loader, best_criterion, device)

        labels = metrics.pop("labels")
        probs = metrics.pop("probs")

        pd.DataFrame({
            "run": run,
            "pat_id": test_df["pat_id"].values,
            "label": labels,
            "prob": probs
        }).to_csv(out_dir / f"test_predictions_run{run}.csv", index=False)

        metrics.update({"run": run, "learning_rate": best_lr, "best_epoch": best_epoch})
        all_test_metrics.append(metrics)

        # ---- external test evaluation (frozen best model, untouched cohort)
        for external in externals:
            ext_ds = EmbeddingDataset(external["df"], external["cache"])
            ext_loader = DataLoader(ext_ds, batch_size=cfg.training.batch_size)
            ext_metrics = evaluate(best_model, ext_loader, best_criterion, device)

            ext_labels = ext_metrics.pop("labels")
            ext_probs = ext_metrics.pop("probs")
            ext_metrics.update(recalibrated_threshold_metrics(ext_labels, ext_probs))
            external["runner"].record(
                run, ext_metrics,
                pd.DataFrame({
                    "run": run,
                    "pat_id": external["df"]["pat_id"].values,
                    "label": ext_labels,
                    "prob": ext_probs,
                }),
            )

    pd.DataFrame(all_histories).to_csv(out_dir / "training_history_all_runs.csv", index=False)
    pd.DataFrame(all_test_metrics).to_csv(out_dir / "test_metrics_runs.csv", index=False)
    pd.DataFrame(split_stats_records).to_csv(out_dir / "split_statistics.csv", index=False)

    summary = pd.DataFrame(all_test_metrics).drop(columns=["run", "best_epoch", "learning_rate"]).agg(["mean", "std"]).T
    summary.to_csv(out_dir / "test_metrics_summary.csv")

    print("\nFinal Results (mean ± std):")
    print(summary)

    for external in externals:
        external["runner"].finalize([
            "loss", "acc", "bal_acc", "f1", "auroc", "auprc",
            "F1_recal", "BalancedAccuracy_recal",
        ])


if __name__ == "__main__":
    main()