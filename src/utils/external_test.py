from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ExternalTestRunner:
    """Collects per-run external-test metrics + predictions and writes them.

    Usage::

        ext = ExternalTestRunner(
            dataset="UCSD-PTGBM",
            out_dir=out_dir / "external_UCSD-PTGBM",
            metrics_filename="results_per_run.csv",
            predictions_filename="predictions_per_patient.csv",
        )
        for run in range(n_runs):
            ...
            ext.record(run, metric_dict, predictions_df)
        ext.finalize(["AUROC", "AUPRC", "F1", "BalancedAccuracy"])
    """

    def __init__(
        self,
        dataset: str,
        out_dir: Path,
        metrics_filename: str,
        predictions_filename: Optional[str] = None,
        summary_filename: Optional[str] = None,
        per_run_prediction_pattern: Optional[str] = None,
    ):
        self.dataset = dataset
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_filename = metrics_filename
        self.predictions_filename = predictions_filename
        self.per_run_prediction_pattern = per_run_prediction_pattern
        if summary_filename is None:
            # Derive a sensible default: "_runs" / "_per_run" / "_per_split"
            # → "_summary" so caller only has to supply one filename.
            summary_filename = (
                metrics_filename
                .replace("_runs.csv", "_summary.csv")
                .replace("_per_run.csv", "_summary.csv")
                .replace("_per_split.csv", "_summary.csv")
            )
        self.summary_filename = summary_filename
        self._metrics: list[dict] = []
        self._predictions: list[pd.DataFrame] = []

    def record(self, run: int, metrics: dict, predictions: pd.DataFrame) -> None:
        """Append one run's metric dict and prediction DataFrame.

        If ``per_run_prediction_pattern`` was configured, also writes
        a per-run CSV (e.g. ``test_predictions_run3.csv``) immediately.
        """
        self._metrics.append({**metrics, "run": run})
        self._predictions.append(predictions)
        if self.per_run_prediction_pattern is not None:
            predictions.to_csv(
                self.out_dir / self.per_run_prediction_pattern.format(run=run),
                index=False,
            )

    def finalize(self, metric_cols: list[str]) -> pd.DataFrame:
        """Write metrics_per_run, metrics_summary, and aggregated predictions.

        ``metric_cols`` is the subset of columns in each recorded
        metrics dict to report mean±std over (e.g.
        ``["AUROC","AUPRC","F1","BalancedAccuracy"]``). Returns the
        per-run metrics DataFrame so the caller can print / inspect it.
        """
        metrics_df = pd.DataFrame(self._metrics)
        metrics_df.to_csv(self.out_dir / self.metrics_filename, index=False)

        summary = pd.DataFrame({
            "metric": metric_cols,
            "mean": [metrics_df[c].mean() for c in metric_cols],
            "std": [metrics_df[c].std() for c in metric_cols],
        })
        summary.to_csv(self.out_dir / self.summary_filename, index=False)

        if self.predictions_filename is not None and self._predictions:
            pd.concat(self._predictions, ignore_index=True).to_csv(
                self.out_dir / self.predictions_filename, index=False,
            )

        print(f"\nExternal ({self.dataset}) Results (mean ± std):")
        for c in metric_cols:
            print(f"{c}: {metrics_df[c].mean():.3f} ± {metrics_df[c].std():.3f}")
        return metrics_df
