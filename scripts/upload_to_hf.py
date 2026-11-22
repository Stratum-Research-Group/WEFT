"""
Hugging Face Hub Release Deployment Script for WEFT-2 Sub-Bit Fleet.
Publishes .weft model containers, GGUF files, and mathematical certificates to the Hub.
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_fleet(repo_id: str, local_dir: str = "checkpoints"):
    api = HfApi()
    print(f"Ensuring repository {repo_id} exists on Hugging Face Hub...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Files to upload
    target_extensions = (".weft", ".gguf", "_certificate.json")
    files_to_upload = [
        f for f in os.listdir(local_dir)
        if any(f.endswith(ext) for ext in target_extensions)
    ]
    
    print(f"Found {len(files_to_upload)} release artifacts in {local_dir}:")
    for f in sorted(files_to_upload):
        size_mb = os.path.getsize(os.path.join(local_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")

    for f in sorted(files_to_upload):
        local_path = os.path.join(local_dir, f)
        print(f"Uploading {f} to {repo_id}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded {f}")
        except Exception as err:
            print(f"Upload of {f} skipped / deferred: {err}")

    # Also upload the model card README.md if present
    readme_path = os.path.join(local_dir, "README.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print("Uploaded model card README.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload WEFT-2 models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="Stratum-Research-Group/weft-2-subbit-fleet", help="Hugging Face repo ID")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Local directory containing artifacts")
    args = parser.parse_args()
    upload_fleet(args.repo_id, args.dir)
