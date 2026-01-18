import yaml
from huggingface_hub import snapshot_download

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HF_READ_TOKEN = config.get("HF_READ_TOKEN", "")
install_config = config.get("data_pipeline", {}).get("install", {})

folder = snapshot_download(
    install_config.get("source_repo", "HuggingFaceFW/fineweb"), 
    repo_type=install_config.get("repo_type", "dataset"),
    local_dir=install_config.get("local_dir", "data/fineweb"),
    allow_patterns=install_config.get("allow_patterns", []),
    token=HF_READ_TOKEN
)
