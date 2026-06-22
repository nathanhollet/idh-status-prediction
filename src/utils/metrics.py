import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score
)


def recalibrated_threshold_metrics(y_true, y_prob) -> dict:
    """Best-threshold F1 / balanced accuracy for an external cohort.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = np.array([f1_score(y_true, y_prob > t, zero_division=0) for t in thresholds])
    bas = np.array([balanced_accuracy_score(y_true, y_prob > t) for t in thresholds])
    best_f1, best_ba = int(f1s.argmax()), int(bas.argmax())
    return {
        "F1_recal": float(f1s[best_f1]),
        "BalancedAccuracy_recal": float(bas[best_ba]),
        "threshold_f1": float(thresholds[best_f1]),
        "threshold_ba": float(thresholds[best_ba]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> dict:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)

        probs = torch.sigmoid(out).view(-1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.view(-1).cpu().numpy())

    preds = (np.array(all_probs) > 0.5).astype(int)
    labels = np.array(all_labels)

    try:
        auroc = roc_auc_score(labels, all_probs)
        auprc = average_precision_score(labels, all_probs)
    except Exception:
        auroc, auprc = float("nan"), float("nan")

    return {
        "loss": total_loss / len(labels),
        "acc": accuracy_score(labels, preds),
        "bal_acc": balanced_accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auroc": auroc,
        "auprc": auprc,
        "labels": labels,
        "probs": np.array(all_probs)
    }