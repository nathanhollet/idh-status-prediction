from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from src.utils.paths import canonical_patient_id, get_dataset_cfg


def get_slice_indices(seg_data):
    """Get 25th, 50th, 75th percentile tumor slices from segmentation mask."""
    tumor_voxels = np.where(seg_data > 0)
    z_indices = tumor_voxels[2]
    if len(z_indices) == 0:
        raise ValueError("Tumor mask is empty.")
    return [int(np.percentile(z_indices, p)) for p in [25, 50, 75]]


def extract_slice_embeddings(
    input_dir: Path,
    output_dir: Path,
    sequence: str,
    dataset: str,
    embed_fn: Callable,
    desc: str = "Extracting embeddings",
) -> None:
    """Shared patient-loop for slice-based embedding extraction (BiomedCLIP, MRICore).

    Args:
        input_dir: Root data directory containing patient folders.
        output_dir: Where to save .pt files.
        sequence: MRI sequence name (e.g. "FLAIR", "T1c").
        dataset: Dataset name, used to pick segmentation pattern.
        embed_fn: Callable(mri_data, seg_data, slice_idx) -> Tensor [1, D].
        desc: tqdm progress bar description.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_cfg = get_dataset_cfg(dataset)

    def _resolve(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _patterns_for_sequence(value, kind: str) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            entry = value.get(sequence)
            if entry is None:
                raise ValueError(
                    f"No {kind} pattern configured for sequence '{sequence}' "
                    f"on dataset '{dataset}'. Available: {list(value.keys())}"
                )
            return [entry] if isinstance(entry, str) else list(entry)
        raise TypeError(f"Unexpected {kind} pattern type: {type(value).__name__}")

    seg_patterns = _patterns_for_sequence(
        _resolve(ds_cfg.raw_tumor_seg_pattern), "segmentation"
    )

    # ``raw_image_pattern`` is an optional escape hatch for datasets whose
    # raw files don't follow the ``<id>_<seq>.nii.gz`` convention
    # (e.g. UTSW ships ``brain_fl_ants.nii.gz`` / ``brain_t1ce_ants.nii.gz``).
    # Defaults to a sequence-suffix glob so existing datasets keep working
    # unchanged.
    raw_img_cfg = _resolve(ds_cfg.get("raw_image_pattern", None))
    if raw_img_cfg is None:
        img_patterns = [f"*{sequence}.nii.gz"]
    else:
        img_patterns = _patterns_for_sequence(raw_img_cfg, "image")

    patient_folders = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name != "preprocessed"])
    print(f"Found {len(patient_folders)} patients.\n")

    for patient_folder in tqdm(patient_folders, desc=desc):
        patient_id = canonical_patient_id(patient_folder.name, ds_cfg)
        out_path = output_dir / f"{patient_id}_{sequence}.pt"

        if out_path.exists():
            tqdm.write(f"Skipping {patient_id} (already processed)")
            continue

        seg_files: list[Path] = []
        for pat in seg_patterns:
            seg_files = list(patient_folder.rglob(pat))
            if seg_files:
                break

        mri_files: list[Path] = []
        for pat in img_patterns:
            mri_files = list(patient_folder.rglob(pat))
            if mri_files:
                break

        if not seg_files or not mri_files:
            tqdm.write(f"Skipping {patient_id} (missing {sequence} or segmentation)")
            continue

        try:
            # Reorient both to canonical RAS so the axial axis is always -1
            # and the image/seg voxel grids line up. No-op for
            # already-canonical datasets.
            mri_img = nib.as_closest_canonical(nib.load(str(mri_files[0])))
            seg_img = nib.as_closest_canonical(nib.load(str(seg_files[0])))
            mri_data = mri_img.get_fdata()
            seg_data = seg_img.get_fdata()
            if seg_data.shape != mri_data.shape:
                raise ValueError(
                    f"Image/seg shape mismatch after canonical reorientation: "
                    f"image {mri_data.shape} vs seg {seg_data.shape}"
                )
            slice_list = get_slice_indices(seg_data)

            embeddings = [embed_fn(mri_data, seg_data, idx) for idx in slice_list]

            final_emb = torch.cat(embeddings, dim=1)
            torch.save({
                "embedding": final_emb,
                "slices": {"p25": slice_list[0], "p50": slice_list[1], "p75": slice_list[2]},
            }, out_path)
            tqdm.write(f"Saved {out_path.name} | slices {slice_list} | shape {tuple(final_emb.shape)}")

        except Exception as e:
            tqdm.write(f"Error processing {patient_id}: {e}")


def load_mri_slice_mricore(mri_data, seg_data, slice_idx, target_size):
    """Load and preprocess a single slice for MRICore — returns [1,3,H,W] tensor."""
    img_slice = mri_data[..., slice_idx]
    img_slice = img_slice - np.min(img_slice)
    if np.max(img_slice) > 0:
        img_slice = img_slice / np.max(img_slice)
    img_resized = resize(img_slice, (target_size, target_size), preserve_range=True)
    img_3ch = np.stack([img_resized] * 3, axis=0)
    return torch.tensor(img_3ch, dtype=torch.float32).unsqueeze(0)


def load_mri_slice_biomedclip(mri_data, slice_idx, preprocess, device):
    """Load and preprocess a single slice for BiomedCLIP — returns preprocessed tensor."""
    img_slice = mri_data[..., slice_idx]
    img_slice = img_slice - np.min(img_slice)
    if np.max(img_slice) > 0:
        img_slice = img_slice / np.max(img_slice)
    img_uint8 = (img_slice * 255).astype(np.uint8)
    pil_img = Image.fromarray(np.stack([img_uint8] * 3, axis=-1))
    return preprocess(pil_img).unsqueeze(0).to(device)