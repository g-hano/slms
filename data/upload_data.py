from huggingface_hub import create_repo, upload_folder, login

#create_repo(
#    repo_id="phase1-256",
#    repo_type="dataset",
#    private=True
#)
#upload_folder(
#    repo_id="Chan-Y/phase1-256",
#    repo_type="dataset",
#    folder_path="D:/fineweb2-packed/phase1_256",
#)
#create_repo(
#    repo_id="phase2-1024",
#    repo_type="dataset",
#    private=True
#)
#upload_folder(
#    repo_id="Chan-Y/phase2-1024",
#    repo_type="dataset",
#    folder_path="D:/fineweb2-packed/phase2_1024",
#)
#create_repo(
#    repo_id="phase3-2048",
#    repo_type="dataset",
#    private=True
#)
upload_folder(
    repo_id="ChanY2/phase3-2048",
    repo_type="dataset",
    folder_path="D:/fineweb2-packed/phase3_2048",
)
