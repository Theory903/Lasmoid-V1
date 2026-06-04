import torch
from huggingface_hub import hf_hub_download
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS

def main():
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    filename = "lasmoid_step_00000400.pt"
    
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    sd = ckpt["model_state_dict"]
    
    print("--- Checkpoint Parameter Statistics ---")
    for name, t in sd.items():
        if "encoder" in name or "emb" in name or "memory" in name or "head" in name:
            print(f"  {name:50s} | shape: {str(list(t.shape)):15s} | mean: {t.float().mean().item():.6f} | std: {t.float().std().item():.6f} | max: {t.float().abs().max().item():.6f}")

if __name__ == "__main__":
    main()
