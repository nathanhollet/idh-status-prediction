# Data layout

Raw imaging is not redistributed here. Download each cohort from its source and place the patient folders directly under this directory, matching the layouts below.

[`csvs/`](csvs) ships only the per-cohort source metadata that `*_csv_assembly.py` consumes (TCIA-distributed metadata files for UCSF / UPENN / UTSW / UCSD), plus the precomputed `radiomics/*_merged.csv` features used by the TabPFN and LogReg baselines. After downloading the raw imaging, the IDH-label CSVs and train/val/test splits are regenerated via the assembly + split scripts (a few seconds per cohort).

**ERASMUS-GBM (EGD):** the data-use agreement does not permit redistribution of the original clinical/genomic tables. Users must request the EGD data from the authors and place `Erasmus_Clinical_data_original.xlsx` + `Erasmus_Genetic_and_Histological_labels_original.xlsx` under `data/csvs/` before running `erasmus_csv_assembly.py`. No EGD-derived CSVs are shipped in this repo.

## Sources

| Cohort       | Source                                                                                         |
|--------------|------------------------------------------------------------------------------------------------|
| UCSF-PDGM    | https://www.cancerimagingarchive.net/collection/ucsf-pdgm/                                     |
| UPENN-GBM    | https://www.cancerimagingarchive.net/collection/upenn-gbm/                                     |
| ERASMUS-GBM  | https://www.healthinformationportal.eu/health-information-sources/erasmus-glioma-database (email request) |
| UTSW-Glioma  | https://doi.org/10.7937/DFAE-1B86 (TCIA)                                                       |
| UCSD-PTGBM   | https://doi.org/10.7937/FWV2-DT74 (TCIA; external IDH test cohort only)                        |

## Expected per-cohort layouts

Only the files our pipeline actually reads are listed. Downloads typically include additional sequences and derived volumes, these can be deleted to save disk space.

### UCSF-PDGM

```
data/UCSF-PDGM/<UCSF-PDGM-XXXX>/
├── UCSF-PDGM-XXXX_FLAIR.nii.gz
├── UCSF-PDGM-XXXX_T1c.nii.gz
├── UCSF-PDGM-XXXX_brain_segmentation.nii.gz       (used for z-score mask)
└── UCSF-PDGM-XXXX_tumor_segmentation.nii.gz       (multi-label, binarized at preprocessing)
```

### UPENN-GBM

```
data/UPENN-GBM/<UPENN-GBM-XXXXX>/
├── UPENN-GBM-XXXXX_11_FLAIR.nii.gz                (timepoint "_11" suffix; one TP per patient)
├── UPENN-GBM-XXXXX_11_T1GD.nii.gz
└── UPENN-GBM-XXXXX_11_automated_approx_segm.nii.gz
```

### ERASMUS-GBM (EGD)

```
data/ERASMUS-GBM/<EGD-XXXX>/
├── EGD-XXXX_FLAIR.nii.gz                          (not skull-stripped; HD-BET runs at preprocessing)
├── EGD-XXXX_T1GD.nii.gz
└── EGD-XXXX_tumor_mask.nii.gz
```

### UTSW-Glioma

Files are non-prefixed and pre-skull-stripped (ANTs-registered):

```
data/UTSW-Glioma/<BTXXXX>/
├── brain_fl_ants.nii.gz                           (FLAIR)
├── brain_t1ce_ants.nii.gz                         (T1c)
├── rtumorseg_manual_correction.nii.gz             (preferred; image-grid mask)
└── tumorseg_FeTS.nii.gz                           (FeTS auto-seg fallback if the above is missing)
```

### UCSD-PTGBM

```
data/UCSD-PTGBM/<UCSD-PTGBM-XXXX_NN>/              (NN = timepoint; keep earliest only per patient)
├── UCSD-PTGBM-XXXX_NN_FLAIR.nii.gz
├── UCSD-PTGBM-XXXX_NN_T1post.nii.gz               (maps to the codebase's "T1c")
└── UCSD-PTGBM-XXXX_NN_BraTS_tumor_seg.nii.gz      (BraTS multi-label seg, binarized at preprocessing)
```

## After download

For each cohort, run the per-cohort assembly script to regenerate the label CSVs from the source metadata, then generate the train/val/test splits:

```bash
python -m src.preprocessing.ucsf_csv_assembly        # likewise for upenn / utsw / ucsd / erasmus
python -m src.preprocessing.generate_splits --dataset UCSF-PDGM   # likewise for the other training cohorts
```

The one-liner that ingests a downloaded cohort into the canonical preprocessed layout is:

```bash
python -m src.preprocessing.rfe_preprocessing --dataset <COHORT>
```

The precomputed `radiomics/*_merged.csv` features are already provided to spare users a ~30-minute PyRadiomics extraction per cohort. To regenerate them from scratch (after preprocessing), run `extract_radiomics.py` then `clean_radiomics.py` — both are wrapped by `run_all.sh`.
