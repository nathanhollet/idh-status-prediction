import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import argparse
import numpy as np
import nibabel as nib

from src.utils.paths import project_root, cfg, get_dataset_cfg, canonical_patient_id


def zscore_normalize(image, mask):
    mask = mask > 0
    values = image[mask]
    mean, std = values.mean(), values.std()
    if std == 0:
        std = 1
    norm = (image - mean) / std
    norm = np.clip(norm, -7, 7)
    norm = norm - norm.min()
    norm[mask == 0] = 0
    return norm


def binarize_mask(mask_path, output_path):
    mask_img = nib.load(mask_path)
    binary = (mask_img.get_fdata() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(binary, mask_img.affine, mask_img.header), output_path)
    print(f"Saved: {output_path}")


def process_ucsf(patient_dir, out_dir):
    patient = Path(patient_dir).name
    patient_id = patient.replace("_nifti", "")

    # Step 1: binarize tumor mask
    tumor_mask = Path(patient_dir) / f"{patient_id}_tumor_segmentation.nii.gz"
    out_patient_dir = Path(out_dir) / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_patient_dir / f"{patient_id}_tumor_mask.nii.gz"

    if not tumor_mask.exists():
        print(f"Skipping {patient_id}: tumor mask not found")
        return

    if not out_mask.exists():
        binarize_mask(str(tumor_mask), str(out_mask))

    # Step 2: normalize FLAIR and T1c using brain segmentation mask
    brain_mask_path = Path(patient_dir) / f"{patient_id}_brain_segmentation.nii.gz"
    if not brain_mask_path.exists():
        print(f"Skipping {patient_id}: brain mask not found")
        return

    brain_mask = nib.load(str(brain_mask_path)).get_fdata()

    for seq in ["FLAIR", "T1c", "T1", "T2"]:
        out_path = out_patient_dir / f"{patient_id}_{seq}.nii.gz"
        if out_path.exists():
            continue
        img_path = Path(patient_dir) / f"{patient_id}_{seq}.nii.gz"
        if not img_path.exists():
            continue
        img = nib.load(str(img_path))
        norm = zscore_normalize(img.get_fdata(), brain_mask)
        nib.save(nib.Nifti1Image(norm, img.affine, img.header), out_path)
        print(f"Saved: {out_path}")


def process_erasmus(patient_dir, out_dir):
    patient = Path(patient_dir).name
    out_patient_dir = Path(out_dir) / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    for seq in ["FLAIR", "T1GD"]:
        img_path = Path(patient_dir) / f"{patient}_{seq}.nii.gz"
        mask_path = Path(patient_dir) / f"{patient}_{seq}_brain_mask.nii.gz"

        if not img_path.exists() or not mask_path.exists():
            continue

        img = nib.load(str(img_path))
        mask = nib.load(str(mask_path)).get_fdata()
        norm = zscore_normalize(img.get_fdata(), mask)
        nib.save(nib.Nifti1Image(norm, img.affine, img.header), out_patient_dir / img_path.name)
        print(f"Saved: {out_patient_dir / img_path.name}")


def process_upenn(patient_dir, out_dir):
    """Preprocess one UPENN-GBM patient.

    UPENN images are already skull-stripped (background = 0), so we use
    ``image > 0`` as the brain mask for z-score normalization. The
    segmentation file is ``<id>_automated_approx_segm.nii.gz`` and gets
    binarized into ``<id>_tumor_mask.nii.gz`` to match the convention
    used by ``extract_radiomics`` for the other datasets.
    """
    ds_cfg = get_dataset_cfg("UPENN-GBM")
    patient = Path(patient_dir).name
    base_id = canonical_patient_id(patient, ds_cfg)
    out_patient_dir = Path(out_dir) / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    raw_seg = Path(patient_dir) / f"{base_id}_automated_approx_segm.nii.gz"
    if not raw_seg.exists():
        print(f"Skipping {base_id}: segmentation not found at {raw_seg}")
        return
    out_mask = out_patient_dir / f"{base_id}_tumor_mask.nii.gz"
    if not out_mask.exists():
        binarize_mask(str(raw_seg), str(out_mask))

    for seq in ["FLAIR", "T1GD"]:
        img_path = Path(patient_dir) / f"{base_id}_{seq}.nii.gz"
        if not img_path.exists():
            continue
        img = nib.load(str(img_path))
        data = img.get_fdata()
        norm = zscore_normalize(data, data > 0)
        out_path = out_patient_dir / f"{base_id}_{seq}.nii.gz"
        nib.save(nib.Nifti1Image(norm, img.affine, img.header), out_path)
        print(f"Saved: {out_path}")


def process_utsw(patient_dir, out_dir):
    """Preprocess one UTSW-Glioma patient.

    Raw UTSW layout::

        BT0001/brain_fl_ants.nii.gz
        BT0001/brain_t1ce_ants.nii.gz
        BT0001/rtumorseg_manual_correction.nii.gz  (preferred, image-grid)
        BT0001/tumorseg_FeTS.nii.gz                 (auto-seg fallback)

    Images are already ANTs-registered and skull-stripped (background = 0),
    so we z-score normalize with ``image > 0`` as the brain mask (same as
    UPENN). Writes canonical-named files under the preprocessed
    dir so ``extract_radiomics`` can consume UTSW with the same resolver
    it uses for UCSF/UPENN::

        <out>/BT0001/BT0001_FLAIR.nii.gz
        <out>/BT0001/BT0001_T1c.nii.gz
        <out>/BT0001/BT0001_tumor_mask.nii.gz
    """
    ds_cfg = get_dataset_cfg("UTSW-Glioma")
    patient = Path(patient_dir).name
    base_id = canonical_patient_id(patient, ds_cfg)  # "BT0001"

    # Prefer manually-refined mask (resampled to image grid); fall back to
    # FeTS auto-seg. The non-``r`` manual variant has a different voxel
    # grid than the image and can't be used with PyRadiomics.
    seg_candidates = [
        Path(patient_dir) / "rtumorseg_manual_correction.nii.gz",
        Path(patient_dir) / "tumorseg_FeTS.nii.gz",
    ]
    seg_path = next((p for p in seg_candidates if p.exists()), None)
    if seg_path is None:
        print(f"Skipping {base_id}: no usable tumor segmentation in {patient_dir}")
        return

    out_patient_dir = Path(out_dir) / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    out_mask = out_patient_dir / f"{base_id}_tumor_mask.nii.gz"
    if not out_mask.exists():
        binarize_mask(str(seg_path), str(out_mask))

    for seq, fname in [("FLAIR", "brain_fl_ants.nii.gz"),
                       ("T1c", "brain_t1ce_ants.nii.gz"),
                       ("T1", "brain_t1_ants.nii.gz"),
                       ("T2", "brain_t2_ants.nii.gz")]:
        out_path = out_patient_dir / f"{base_id}_{seq}.nii.gz"
        if out_path.exists():
            continue
        img_path = Path(patient_dir) / fname
        if not img_path.exists():
            print(f"Skipping {base_id} {seq}: image missing at {img_path}")
            continue
        img = nib.load(str(img_path))
        data = img.get_fdata()
        norm = zscore_normalize(data, data > 0)
        nib.save(nib.Nifti1Image(norm, img.affine, img.header), out_path)
        print(f"Saved: {out_path}")


def process_ucsd_ptgbm(patient_dir, out_dir):
    """Preprocess one UCSD-PTGBM patient (external IDH test cohort).

    Raw UCSD-PTGBM layout (one timepoint folder per patient)::

        UCSD-PTGBM-0001_01/
            UCSD-PTGBM-0001_01_FLAIR.nii.gz
            UCSD-PTGBM-0001_01_T1post.nii.gz            (post-contrast T1)
            UCSD-PTGBM-0001_01_BraTS_tumor_seg.nii.gz   (BraTS multi-label)

    Images are BraTS-preprocessed (already skull-stripped and
    co-registered), background = 0, so we z-score with ``image > 0`` as
    the brain mask — same as UPENN / UTSW. Writes canonical
    names under the preprocessed dir so ``extract_radiomics`` can
    consume UCSD with the same resolver it uses for everyone else.
    Renames ``_T1post`` → ``_T1c`` so the column name matches the
    codebase's convention::

        <out>/UCSD-PTGBM-0001_01/UCSD-PTGBM-0001_01_FLAIR.nii.gz
        <out>/UCSD-PTGBM-0001_01/UCSD-PTGBM-0001_01_T1c.nii.gz
        <out>/UCSD-PTGBM-0001_01/UCSD-PTGBM-0001_01_tumor_mask.nii.gz
    """
    patient = Path(patient_dir).name
    base_id = patient  # UCSD folders = pat_id (no suffix / prefix surgery)

    raw_seg = Path(patient_dir) / f"{base_id}_BraTS_tumor_seg.nii.gz"
    if not raw_seg.exists():
        print(f"Skipping {base_id}: BraTS tumor seg not found at {raw_seg}")
        return

    out_patient_dir = Path(out_dir) / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    out_mask = out_patient_dir / f"{base_id}_tumor_mask.nii.gz"
    if not out_mask.exists():
        binarize_mask(str(raw_seg), str(out_mask))

    for seq, raw_suffix in [("FLAIR", "FLAIR"),
                            ("T1c", "T1post"),
                            ("T1", "T1pre"),
                            ("T2", "T2")]:
        out_path = out_patient_dir / f"{base_id}_{seq}.nii.gz"
        if out_path.exists():
            continue
        img_path = Path(patient_dir) / f"{base_id}_{raw_suffix}.nii.gz"
        if not img_path.exists():
            print(f"Skipping {base_id} {seq}: image missing at {img_path}")
            continue
        img = nib.load(str(img_path))
        data = img.get_fdata()
        norm = zscore_normalize(data, data > 0)
        nib.save(nib.Nifti1Image(norm, img.affine, img.header), out_path)
        print(f"Saved: {out_path}")


PROCESS_FNS = {
    "UCSF-PDGM": process_ucsf,
    "ERASMUS-GBM": process_erasmus,
    "UPENN-GBM": process_upenn,
    "UTSW-Glioma": process_utsw,
    "UCSD-PTGBM": process_ucsd_ptgbm,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    args = parser.parse_args()

    input_dir = project_root / cfg.paths.data[args.dataset]
    output_dir = project_root / cfg.paths.data[args.dataset] / "preprocessed"

    if args.dataset not in PROCESS_FNS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(PROCESS_FNS)}")
    process_fn = PROCESS_FNS[args.dataset]

    patients = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name != "preprocessed"])
    print(f"Found {len(patients)} patients.")

    for patient_dir in patients:
        process_fn(str(patient_dir), str(output_dir))


if __name__ == "__main__":
    main()