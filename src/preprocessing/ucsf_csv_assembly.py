#!/usr/bin/env python3
"""Assemble the UCSF-PDGM ``pat_id, age, grade, label`` MAIN CSV from the
v5 metadata table.

Reads:
  * ``data/csvs/UCSF-PDGM-metadata_v5_original.csv``

Writes:
  * ``data/csvs/UCSF_IDH_age_grade_labels.csv``

Notes
-----
* The metadata ``ID`` column uses 3-digit padding (``UCSF-PDGM-004``) and
  also contains a handful of follow-up scan rows suffixed ``_FU<n>d``
  (e.g. ``UCSF-PDGM-0391_FU016d``). We normalize to the canonical
  4-digit-padded ``UCSF-PDGM-0004`` form used by the rest of the
  pipeline, then drop the follow-up scans (keeping the baseline).
* Label encoding: ``IDH`` is a string column with values ``wildtype`` or
  one of several mutation variants (``IDH1 p.R132H``, ``mutated (NOS)``,
  ...). We collapse to binary ``label`` = 0 (wildtype) / 1 (any mutant),
  matching the polarity used by every other cohort in this project.
"""

import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.utils.paths import project_root


def normalize_id(raw: str) -> str:
    """Convert ``UCSF-PDGM-004`` / ``UCSF-PDGM-0391_FU016d`` -> ``UCSF-PDGM-0004``."""
    m = re.match(r"^(UCSF-PDGM)-(\d+)", raw)
    if not m:
        raise ValueError(f"Unexpected UCSF ID format: {raw!r}")
    return f"{m.group(1)}-{int(m.group(2)):04d}"


def main() -> None:
    csvs = project_root / "data" / "csvs"
    meta = pd.read_csv(csvs / "UCSF-PDGM-metadata_v5_original.csv")

    meta["pat_id"] = meta["ID"].apply(normalize_id)

    before = len(meta)
    meta = meta.sort_values("ID").drop_duplicates("pat_id", keep="first")
    dropped = before - len(meta)
    if dropped:
        print(f"Dropped {dropped} follow-up scan rows (kept baseline).")

    meta["label"] = (meta["IDH"] != "wildtype").astype(int)
    meta = meta.rename(columns={"Age at MRI": "age", "WHO CNS Grade": "grade"})
    df = meta[["pat_id", "age", "grade", "label"]].sort_values("pat_id").reset_index(drop=True)

    out = csvs / "UCSF_IDH_age_grade_labels.csv"
    df.to_csv(out, index=False)
    print(df.head())
    print(
        f"Total: {len(df)}, Mutant: {df.label.sum()}, "
        f"Wildtype: {(df.label == 0).sum()} -> {out}"
    )


if __name__ == "__main__":
    main()
