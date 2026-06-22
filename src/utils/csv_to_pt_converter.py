import os, argparse, torch, pandas as pd

# Example run: Change input and output paths as necessary
# python csv_to_pt_converter.py \
#     --input_csv ./csvs/extracted_features_UCSF_IDH_FLAIR.csv \
#     --output_dir ./.pt_files/UCSF_IDH_FLAIR/

def convert_csv_to_pt(csv_path, output_root):
    # Create output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    print("CSV shape:", df.shape)

    # Columns to drop
    columns_to_drop = ["pat_id", "GroundTruthClassLabel"]

    # Check embedding dimensions
    example_embedding = df.drop(columns=columns_to_drop).iloc[0].values.astype("float32")
    print("Example tensor shape:", example_embedding.shape)

    # Loop over rows
    for idx, row in df.iterrows():

        pat_id = row["pat_id"]

        # Drop unwanted columns
        embedding_values = row.drop(columns_to_drop).values.astype("float32")

        # Convert to torch tensor
        embedding_tensor = torch.tensor(embedding_values)

        # Output path
        output_path = os.path.join(output_root, f"{pat_id}.pt")

        # Save tensor
        torch.save(embedding_tensor, output_path)

    print("All embeddings saved as .pt files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV embeddings to .pt tensors.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, help="Directory to save .pt files")

    args = parser.parse_args()

    convert_csv_to_pt(args.input_csv, args.output_dir)