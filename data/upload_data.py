import yaml
from huggingface_hub import create_repo, upload_folder, login

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HF_WRITE_TOKEN = config.get("HF_WRITE_TOKEN", "")
upload_config = config.get("data_pipeline", {}).get("upload", {})

# Login to HuggingFace
if HF_WRITE_TOKEN:
    login(token=HF_WRITE_TOKEN)
else:
    print("Warning: HF_WRITE_TOKEN not found in config.yaml")

# Upload all phases from config
for phase_name, phase_config in upload_config.items():
    repo_id = phase_config.get("repo_id")
    folder_path = phase_config.get("folder_path")
    private = phase_config.get("private", True)
    
    if not repo_id or not folder_path:
        print(f"Warning: Skipping {phase_name} - missing repo_id or folder_path")
        continue
    
    print(f"Processing {phase_name}...")
    
    # Extract repo name from full repo_id (e.g., "Chan-Y/phase1-256" -> "phase1-256")
    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=private
    )
    
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder_path,
    )
    
    print(f"✅ {phase_name} uploaded successfully!")

print("\n🎉 All phases uploaded!")
