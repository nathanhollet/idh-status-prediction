from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset


def preload_embeddings(
    df: pd.DataFrame,
    flair_dir: Path,
    t1c_dir: Path,
    t1c_seq: str = "T1c",
    extra_dirs: list[tuple[str, Path]] = (),
) -> dict:
    """Preload FLAIR + T1 (+ optional extras) embeddings into a dict keyed by pat_id.

    ``t1c_seq`` is the T1 sequence name on disk (e.g. ``"T1c"`` for UCSF,
    ``"T1GD"`` for ERASMUS / UPENN). ``extra_dirs`` is an ordered list of
    ``(sequence_name, embedding_dir)`` pairs for additional modalities
    (e.g. ``[("T1", path/to/T1), ("T2", path/to/T2)]``); each modality's
    vector is appended to the per-patient concat in the order given so the
    linear-probe input dimensionality is deterministic across runs.
    """
    seq_dirs: list[tuple[str, Path]] = [
        ("FLAIR", Path(flair_dir)),
        (t1c_seq, Path(t1c_dir)),
        *[(name, Path(d)) for name, d in extra_dirs],
    ]
    cache = {}

    for pid in df["pat_id"].unique():
        vectors = []
        for seq, d in seq_dirs:
            v = torch.load(d / f"{pid}_{seq}.pt", map_location="cpu", weights_only=True)
            if isinstance(v, dict):
                v = v["embedding"]
            # MRICore embeddings have spatial dimensions — mean pool down to 1D
            if v.ndim == 4:
                v = v.mean(dim=[2, 3]).squeeze(0)
            vectors.append(v.view(-1))
        cache[pid] = torch.cat(vectors, dim=0)

    return cache


class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embedding_cache: dict):
        self.df = df.reset_index(drop=True)
        self.cache = embedding_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.df.iloc[idx]["pat_id"]
        label = int(self.df.iloc[idx]["label"])
        return self.cache[pid], label