import torch
from huggingface_hub import hf_hub_download
import os, sys
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS

def main():
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = 1024
    cfg["max_batch_size"] = 4
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    filename = "lasmoid_step_00000400.pt"
    
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    model.train()
    
    # Batch size 4, sequence length 1024
    xb = torch.randint(0, model_args.vocab_size - 1, (4, 1024), dtype=torch.long)
    yb = torch.randint(0, model_args.vocab_size - 1, (4, 1024), dtype=torch.long)
    
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_nxt, logits_nxt2, _, _ = model(xb, yb)
        z_loss = model.last_z_loss
        ce = F.cross_entropy(logits_nxt.view(-1, model_args.vocab_size), yb.view(-1))
        
    print("\n--- Forward Pass with Real Params (B=4, S=1024) ---")
    print(f"Logits shape:      {logits_nxt.shape}")
    print(f"Logits min/max:    {logits_nxt.min().item():.4f} / {logits_nxt.max().item():.4f}")
    print(f"Cross Entropy (CE): {ce.item():.4f}")
    print(f"Router Z-loss:      {z_loss.item():.4f}")

if __name__ == "__main__":
    main()
