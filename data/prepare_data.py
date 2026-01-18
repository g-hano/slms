import os
import yaml
from glob import glob
from datasets import load_dataset

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

prepare_config = config.get("data_pipeline", {}).get("prepare", {})

# 1. Paths
input_dir = prepare_config.get("input_dir", "D:/fineweb2/data/data/CC-MAIN-2025-26")
output_dir = prepare_config.get("output_dir", "D:/fineweb2-train/CC-MAIN-2025-26")

os.makedirs(output_dir, exist_ok=True)

# 2. Files
files = glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
print(f"Found {len(files)} files. Starting process...")

cols_to_remove = ['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count']

# 3. Process Loop
for file_path in files:
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    # File already exists, skip (You need to delete the old directory to recreate it!)
    if os.path.exists(output_path):
        print(f"File already exists, skipping: {file_name}")
        continue

    try:
        # Load the dataset
        ds = load_dataset("parquet", data_files=file_path, split="train")
        
        # Remove unnecessary columns
        existing_cols = [col for col in cols_to_remove if col in ds.column_names]
        ds_clean = ds.remove_columns(existing_cols)
        
        # Clean broken data (NEWLY ADDED)
        # Filter out rows where text is None or empty string
        if "text" in ds_clean.column_names:
            initial_count = len(ds_clean)
            
            ds_clean = ds_clean.filter(
                lambda x: x["text"] is not None and len(x["text"].strip()) > 0
            )
            
            final_count = len(ds_clean)
            dropped = initial_count - final_count
            if dropped > 0:
                print(f"  -> {file_name}: {dropped} broken/empty rows dropped.")
        
        # Save the cleaned dataset
        ds_clean.to_parquet(output_path)        
        print(f"Completed: {file_name}")
        
    except Exception as e:
        print(f"Error occurred ({file_name}): {e}")

print("All processes completed.")