import torch
from huggingface_hub import hf_hub_download
import os, sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS, Muon

def main():
    print("Initializing model...")
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = 256
    cfg["max_batch_size"] = 1
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    # Set up optimizers with new parameter partition (excluding gate/hc)
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 2
            and "emb" not in name
            and "head" not in name
            and "adj" not in name
            and "gate" not in name
            and "hc" not in name
        ):
            muon_params.append(param)
        else:
            adamw_params.append(param)
            
    print(f"Current Muon params: {len(muon_params)}")
    print(f"Current AdamW params: {len(adamw_params)}")
    
    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=3e-4)
    
    # Download checkpoint from HF Hub
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    filename = "lasmoid_step_00000400.pt"
    
    print(f"Downloading checkpoint {filename}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    print("Loading model state dict...")
    model.load_state_dict(ckpt["model_state_dict"])
    
    print("Running robust optimizer loading...")
    # Reconstruct original parameter list partitioning
    saved_muon_params = []
    saved_adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim == 2
            and "emb" not in name
            and "head" not in name
            and "adj" not in name
        ):
            saved_muon_params.append(p)
        else:
            saved_adamw_params.append(p)
            
    # Load Muon states
    muon_param_ids = []
    for group in ckpt["opt_muon_state"]["param_groups"]:
        muon_param_ids.extend(group["params"])
    muon_state_map = {}
    for idx, p in enumerate(saved_muon_params):
        if idx < len(muon_param_ids):
            p_id = muon_param_ids[idx]
            if p_id in ckpt["opt_muon_state"]["state"]:
                muon_state_map[p] = ckpt["opt_muon_state"]["state"][p_id]
                
    loaded_muon_count = 0
    for group in opt_muon.param_groups:
        for p in group["params"]:
            if p in muon_state_map:
                opt_muon.state[p] = muon_state_map[p]
                loaded_muon_count += 1
                
    # Load AdamW states
    adamw_param_ids = []
    for group in ckpt["opt_adamw_state"]["param_groups"]:
        adamw_param_ids.extend(group["params"])
    adamw_state_map = {}
    for idx, p in enumerate(saved_adamw_params):
        if idx < len(adamw_param_ids):
            p_id = adamw_param_ids[idx]
            if p_id in ckpt["opt_adamw_state"]["state"]:
                adamw_state_map[p] = ckpt["opt_adamw_state"]["state"][p_id]
                
    loaded_adamw_count = 0
    for group in opt_adamw.param_groups:
        for p in group["params"]:
            if p in adamw_state_map:
                opt_adamw.state[p] = adamw_state_map[p]
                loaded_adamw_count += 1
                
    print(f"✅ Robust optimizer state loaded successfully!")
    print(f"   Muon states loaded: {loaded_muon_count} / {len(muon_params)}")
    print(f"   AdamW states loaded: {loaded_adamw_count} / {len(adamw_params)}")
    
if __name__ == "__main__":
    main()
