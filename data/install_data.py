from huggingface_hub import snapshot_download
folder = snapshot_download(
    "HuggingFaceFW/fineweb", 
    repo_type="dataset",
    local_dir="data/fineweb",
    allow_patterns=[
        "data/CC-MAIN-2025-26/000_0000*", # english
        "data/CC-MAIN-2025-26/000_0001*", # english
    ],
    token=""
)
