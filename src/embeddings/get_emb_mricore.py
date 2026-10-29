import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import torch

from src.utils.paths import project_root, cfg
from src.utils.reproducibility import set_seed
from src.utils.inference import load_mri_slice_mricore, extract_slice_embeddings

mricore_src = project_root / "reference" / "mri_foundation"
sys.path.append(str(mricore_src))

from models.sam import sam_model_registry
import cfg as mricore_cfg

set_seed(cfg.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--sequence", type=str, default="FLAIR")
    args = parser.parse_args()

    sys_argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    mricore_args = mricore_cfg.parse_args()
    mricore_args.if_encoder_adapter = False
    mricore_args.if_mask_decoder_adapter = False
    sys.argv = sys_argv_backup

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = project_root / cfg.paths.checkpoints.mricore
    model = sam_model_registry["vit_b"](
        mricore_args,
        checkpoint=str(checkpoint_path),
        num_classes=mricore_args.num_cls,
        image_size=mricore_args.image_size,
        pretrained_sam=False,
    ).to(device).eval()

    def embed_fn(mri_data, seg_data, slice_idx):
        img_tensor = load_mri_slice_mricore(mri_data, seg_data, slice_idx, mricore_args.image_size)
        with torch.no_grad():
            return model.image_encoder(img_tensor.to(device)).cpu()

    extract_slice_embeddings(
        input_dir=project_root / cfg.paths.data[args.dataset],
        output_dir=project_root / cfg.paths.embeddings.mricore / "pt" / args.dataset / args.sequence,
        sequence=args.sequence,
        dataset=args.dataset,
        embed_fn=embed_fn,
        desc=f"Extracting MRICore {args.sequence} embeddings",
    )


if __name__ == "__main__":
    main()
