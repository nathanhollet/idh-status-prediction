import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import argparse
import tempfile
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.paths import project_root, cfg
from dataset import get_validation_transform
from load_brainiac import load_brainiac
from src.datasets.brainiac_registry import DATASET_CLASSES
from src.utils.csv_to_pt_converter import convert_csv_to_pt
from src.utils.reproducibility import set_seed

set_seed(cfg.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"


def infer(model, test_loader):
    model.eval()
    features_df = None

    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting features", unit="batch"):
            inputs = sample['image'].to(device)
            class_labels = sample['label'].float().to(device)

            features_numpy = model(inputs).cpu().numpy()
            batch_features = pd.DataFrame(features_numpy, columns=[f'Feature_{i}' for i in range(features_numpy.shape[1])])
            batch_features["pat_id"] = sample["pat_id"]
            batch_features['GroundTruthClassLabel'] = class_labels.cpu().numpy().flatten()

            features_df = batch_features if features_df is None else pd.concat([features_df, batch_features], ignore_index=True)

    return features_df


def collect_task_csvs(dataset: str, sequence: str) -> list[Path]:
    """Return every per-sequence label CSV across tasks for (dataset, sequence).

    Matches both the default-task key ``{dataset}-{sequence}`` and any
    ``{dataset}-<TASK>-{sequence}`` variants for non-default tasks. A
    patient may be labeled for only one task, so we extract embeddings
    for the union of labeled patients — otherwise patients only labeled
    for a non-default task would have no ``.pt`` cache.
    """
    prefix = f"{dataset}-"
    suffix = f"-{sequence}"
    paths: list[Path] = []
    for key, rel in cfg.paths.data.csvs.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        if key.endswith(f"-MAIN{suffix}"):
            continue
        paths.append(project_root / rel)
    return paths


def union_csv(csv_paths: list[Path]) -> Path:
    """Concat CSVs, drop duplicate pat_ids, write to a tempfile, return path.

    Labels from different tasks are incommensurable, but ``convert_csv_to_pt``
    drops ``GroundTruthClassLabel`` before writing the .pt cache, so keeping
    the first-seen label is harmless bookkeeping.
    """
    combined = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    combined = combined.drop_duplicates(subset="pat_id", keep="first").reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    combined.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--model", type=str, default=cfg.model.name)
    parser.add_argument("--sequence", type=str, default="FLAIR")
    args = parser.parse_args()

    csv_key = f"{args.dataset}-{args.sequence}"

    if args.dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(DATASET_CLASSES)}")

    task_csvs = collect_task_csvs(args.dataset, args.sequence)
    if not task_csvs:
        raise ValueError(
            f"No per-sequence label CSV found for {args.dataset}-{args.sequence} "
            f"in cfg.paths.data.csvs."
        )
    if len(task_csvs) == 1:
        csv_path = task_csvs[0]
    else:
        csv_path = union_csv(task_csvs)
        print(f"Unioned {len(task_csvs)} task CSVs → {csv_path} "
              f"(sources: {[p.name for p in task_csvs]})")

    dataset_class = DATASET_CLASSES[args.dataset]
    test_dataset = dataset_class(
        csv_path=csv_path,
        root_dir=project_root / cfg.paths.data[args.dataset],
        sequence=args.sequence,
        transform=get_validation_transform()
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = load_brainiac(project_root / cfg.paths.checkpoints.brainiac, device)
    features_df = infer(model, test_loader)

    output_path = project_root / cfg.paths.embeddings[args.model] / f"{csv_key}_embeddings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path} — shape: {features_df.shape}, dims: {features_df.shape[1] - 1}")

    pt_output_dir = project_root / cfg.paths.embeddings[args.model] / "pt" / args.dataset / args.sequence
    convert_csv_to_pt(str(output_path), str(pt_output_dir))


if __name__ == "__main__":
    main()