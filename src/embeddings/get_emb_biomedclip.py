import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from open_clip import create_model_from_pretrained
import argparse

from src.utils.paths import project_root, cfg
from src.utils.reproducibility import set_seed
from src.utils.inference import load_mri_slice_biomedclip, extract_slice_embeddings

set_seed(cfg.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.dataset.name)
    parser.add_argument("--sequence", type=str, default="FLAIR")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading BioMedCLIP...")
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(device).eval()

    def embed_fn(mri_data, seg_data, slice_idx):
        img_tensor = load_mri_slice_biomedclip(mri_data, slice_idx, preprocess, device)
        with torch.no_grad():
            return model.encode_image(img_tensor).cpu()

    extract_slice_embeddings(
        input_dir=project_root / cfg.paths.data[args.dataset],
        output_dir=project_root / cfg.paths.embeddings.biomedclip / "pt" / args.dataset / args.sequence,
        sequence=args.sequence,
        dataset=args.dataset,
        embed_fn=embed_fn,
        desc=f"Extracting BiomedCLIP {args.sequence} embeddings",
    )


if __name__ == "__main__":
    main()
