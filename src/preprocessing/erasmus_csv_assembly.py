import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.utils.paths import project_root

def read_table(folder: Path, stem: str) -> pd.DataFrame:
    for ext, reader in [(".csv", pd.read_csv), (".xlsx", pd.read_excel)]:
        path = folder / (stem + ext)
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f"No .csv or .xlsx found for '{stem}' in {folder}")

csvs = project_root / "data" / "csvs"
demographics = read_table(csvs, "Erasmus_Clinical_data_original")
genomics = read_table(csvs, "Erasmus_Genetic_and_Histological_labels_original")

df = demographics.merge(genomics, on="Subject")

df = df[df["IDH"] != -1]

df = df.rename(columns={
    "Subject": "pat_id",
    "Age": "age",
    "Grade": "grade",
    "IDH": "label"
})

df["label"] = df["label"].astype(int)
df = df[["pat_id", "age", "grade", "label"]]

df.to_csv("data/csvs/ERASMUS_IDH_age_grade_labels.csv", index=False)
print(df.head())
print(f"Total patients: {len(df)}, Mutant: {df.label.sum()}, Wildtype: {(df.label==0).sum()}")