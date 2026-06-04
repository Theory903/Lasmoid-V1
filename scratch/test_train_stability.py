import torch
from huggingface_hub import hf_hub_download
import os, sys
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS, Muon

def main():
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = 256
    cfg["max_batch_size"] = 2
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    # Download checkpoint
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    filename = "lasmoid_step_00000400.pt"
    
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    model.train()
    xb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    yb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_nxt, logits_nxt2, _, _ = model(xb, yb)
        z_loss = model.last_z_loss
        ce = F.cross_entropy(logits_nxt.view(-1, model_args.vocab_size), yb.view(-1))
        loss = ce + 0.3 * logits_nxt2.mean() * 0.0 + model_args.router_z_loss_coeff * z_loss
        
    loss.backward()
    
    print("\n--- Individual Parameter Gradient Norms (Top 20) ---")
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            norms.append((name, norm, p.shape))
            
    norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm, shape in norms[:30]:
        print(f"  {name:50s} | shape: {str(list(shape)):15s} | grad norm: {norm:.4f}")

if __name__ == "__main__":
    main()
