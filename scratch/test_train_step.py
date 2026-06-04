import torch
from huggingface_hub import hf_hub_download
import os, sys
import torch.nn.functional as F
import tiktoken
from contextlib import nullcontext

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS, MultiTaskLoader, stream_packed_tokens, Muon, load_checkpoint

def main():
    print("Initializing model...")
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = 1024
    cfg["max_batch_size"] = 4
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    # Setup optimizers exactly like train_kaggle.py
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
            
    opt_muon = Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    
    # Reconstruct load_checkpoint to run locally
    from train_kaggle import load_checkpoint
    # Let's call load_checkpoint
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    
    start_step, tokens_seen = load_checkpoint(model, opt_muon, opt_adamw, api, repo_id, token, "cpu")
    print(f"Resumed from step {start_step}, tokens seen: {tokens_seen}")
    
    # Initialize real data streams
    tok = tiktoken.get_encoding("gpt2")
    
    GEN_EDU = stream_packed_tokens(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        tok,
        model_args.max_seq_len,
        label="FineWeb-Edu",
        hf_token=token,
    )
    
    loader = MultiTaskLoader([GEN_EDU], [1.0], 2, "cpu", eot_id=tok.eot_token)
    
    # Run 5 training updates
    model.train()
    for step in range(start_step, start_step + 5):
        x, y, cu_seqlens = loader.next_batch()
        
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits_nxt, logits_nxt2, _, _ = model(x, x, cu_seqlens=cu_seqlens)
            z_loss = model.last_z_loss
            
            ce = F.cross_entropy(logits_nxt.view(-1, model_args.vocab_size), y.view(-1))
            
            mtp = torch.tensor(0.0)
            if logits_nxt2 is not None:
                mtp = F.cross_entropy(
                    logits_nxt2[:, :-1].contiguous().view(-1, model_args.vocab_size),
                    y[:, 1:].contiguous().view(-1),
                    ignore_index=-1
                )
                
            loss = ce + 0.1 * mtp + model_args.router_z_loss_coeff * z_loss
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        opt_muon.step()
        opt_adamw.step()
        
        print(f"Step {step:3d} | loss: {loss.item():.4f} | ce: {ce.item():.4f} | mtp: {mtp.item():.4f} | z: {z_loss.item():.4f}")

if __name__ == "__main__":
    main()
