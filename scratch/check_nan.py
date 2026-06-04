import torch
from huggingface_hub import hf_hub_download
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

token = os.getenv("HF_TOKEN")
repo_id = "Theory903/lasmoid-10m"
filename = "lasmoid_step_00000400.pt"

path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
ckpt = torch.load(path, map_location="cpu", weights_only=False)

def check_structure(obj, path=""):
    nans = []
    infs = []
    max_val = 0.0
    
    if isinstance(obj, torch.Tensor):
        if torch.isnan(obj).any():
            nans.append(path)
        if torch.isinf(obj).any():
            infs.append(path)
        if obj.numel() > 0:
            m = obj.abs().max().item()
            if m > max_val:
                max_val = m
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sub_nans, sub_infs, sub_max = check_structure(v, f"{path}.{k}" if path else str(k))
            nans.extend(sub_nans)
            infs.extend(sub_infs)
            if sub_max > max_val:
                max_val = sub_max
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sub_nans, sub_infs, sub_max = check_structure(v, f"{path}[{i}]" if path else f"[{i}]")
            nans.extend(sub_nans)
            infs.extend(sub_infs)
            if sub_max > max_val:
                max_val = sub_max
                
    return nans, infs, max_val

for state_name in ["opt_muon_state", "opt_adamw_state"]:
    state = ckpt.get(state_name)
    if state is None:
        print(f"\nNo {state_name} found in checkpoint.")
        continue
    nans, infs, max_val = check_structure(state)
    print(f"\n--- Checking {state_name} ---")
    print(f"Max absolute value in state: {max_val}")
    if nans:
        print(f"❌ Found NaN in state ({len(nans)} keys):")
        for k in nans[:10]:
            print(f"  - {k}")
    else:
        print("✅ No NaNs found in state.")
    if infs:
        print(f"❌ Found Inf in state ({len(infs)} keys):")
        for k in infs[:10]:
            print(f"  - {k}")
    else:
        print("✅ No Infs found in state.")
