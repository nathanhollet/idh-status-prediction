"""BrainDINO (Med-DINOv3) per-slice embedding extraction.

Loads the frozen teacher backbone of the DINOv3-style ViT-B/16 from
``checkpoints/pretrained/braindino/model.pth`` and extracts post-norm
CLS-token features from three tumor-centric axial slices (25th / 50th
/ 75th percentile of the tumor mask). The subject embedding is the
concatenation of the three 768-d CLS tokens (final dim = 2304).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import torch

from src.utils.paths import project_root, cfg
from src.utils.reproducibility import set_seed
from src.utils.inference import load_mri_slice_mricore, extract_slice_embeddings

# The BrainDINO encoder code imports as ``from dinov3.layers import ...`` and
# ``from dinov3.utils import ...``, so we add its parent dir (the one that
# contains the ``dinov3/`` package) to sys.path before importing.
braindino_src = project_root / "reference" / "BrainDINO" / "BrainDINO_seg"
sys.path.append(str(braindino_src))

from dinov3.models.vision_transformer import vit_base  # noqa: E402


IMG_SIZE = 256
N_STORAGE_TOKENS = 4

set_seed(cfg.seed)


def _load_teacher_backbone(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Instantiate ViT-B and load the teacher backbone weights only.

    The checkpoint's ``teacher`` sub-dict contains both backbone weights
    (``backbone.*``) and the DINO/iBOT projection heads, which we don't
    need for downstream embedding extraction. We strip the ``backbone.``
    prefix and load with strict=False so RoPE buffers etc. don't trip
    the strict matcher.
    """
    # qkv_bias=False and layerscale_init=1e-5 are required to match the
    # checkpoint shape: the trained backbone uses no QKV bias and a
    # small-init LayerScale on each block (ls1.gamma / ls2.gamma).
    model = vit_base(
        patch_size=16,
        img_size=IMG_SIZE,
        in_chans=3,
        n_storage_tokens=N_STORAGE_TOKENS,
        qkv_bias=False,
        layerscale_init=1e-5,
        pos_embed_rope_dtype="fp32",  # safer on CPU/older GPUs than bf16
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_sd = ckpt["teacher"] if "teacher" in ckpt else ckpt
    backbone_sd = {
        k[len("backbone."):]: v
        for k, v in teacher_sd.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("rope_embed.")]
    real_unexpected = [k for k in unexpected if not k.startswith("rope_embed.")]
    if real_missing:
        print(f"[braindino] missing keys (first 5): {real_missing[:5]}  (+{max(0, len(real_missing)-5)} more)")
    if real_unexpected:
        print(f"[braindino] unexpected keys (first 5): {real_unexpected[:5]}  (+{max(0, len(real_unexpected)-5)} more)")

    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--sequence", type=str, default="FLAIR")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = project_root / cfg.paths.checkpoints.braindino
    model = _load_teacher_backbone(checkpoint_path, device)

    def embed_fn(mri_data, seg_data, slice_idx):
        # Use the shared MRI-CORE / BiomedCLIP slice loader (min-max
        # normalize to [0, 1], resize to IMG_SIZE, replicate 1→3
        # channels).
        img_tensor = load_mri_slice_mricore(mri_data, seg_data, slice_idx, IMG_SIZE).to(device)
        with torch.no_grad():
            out = model.forward_features(img_tensor)
        # ``forward_features`` returns a dict; ``x_norm_clstoken`` is the
        # post-norm CLS embedding (shape [1, 768]). This matches DINO's
        # standard linear-probe convention.
        return out["x_norm_clstoken"].cpu()

    extract_slice_embeddings(
        input_dir=project_root / cfg.paths.data[args.dataset],
        output_dir=project_root / cfg.paths.embeddings.braindino / "pt" / args.dataset / args.sequence,
        sequence=args.sequence,
        dataset=args.dataset,
        embed_fn=embed_fn,
        desc=f"Extracting BrainDINO {args.sequence} embeddings",
    )


if __name__ == "__main__":
    main()
