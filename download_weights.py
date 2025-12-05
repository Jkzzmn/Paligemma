from huggingface_hub import snapshot_download
import os

def main():
    weights_dir = os.path.expanduser("~/Document/Project/Paligemma/paligemma-weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 모델 다운로드
    model_id = "google/paligemma-3b-pt-224"
    local_dir = os.path.join(weights_dir, "paligemma-3b-pt-224")
    
    print(f"Downloading {model_id} to {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete!")

if __name__ == "__main__":
    main() 