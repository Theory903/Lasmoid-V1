import torch
import torch.nn.functional as F
import os
import sys

# Add root folder to sys.path
sys.path.append(os.path.abspath("."))

from inference.model import LasmoidV1, ModelArgs
import tiktoken

def main():
    print("Loading model config and args...")
    # Initialize 10M config
    cfg = dict(
        dim=128,
        n_layers=4,
        n_heads=4,
        head_dim=48,
        q_lora_rank=32,
        o_lora_rank=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=256,
        max_seq_len=256,
    )
    model_args = ModelArgs(**cfg)
    model_args.dtype = "bf16" # Same as training run
    
    device = "cpu"
    model = LasmoidV1(model_args).to(device)
    
    ckpt_path = "checkpoints/lasmoid_step_00000400.pt"
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Model state dict loaded successfully!")
    
    # Run a mock forward pass to see the logits and loss
    # Using real tiktoken vocabulary tokens
    enc = tiktoken.get_encoding("gpt2")
    # Let's create dummy input tokens (B=2, T=256)
    xb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    yb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    
    model.train()
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_nxt, logits_nxt2, _, _ = model(xb, xb)
        z_loss = model.last_z_loss
        
        ce = F.cross_entropy(
            logits_nxt.view(-1, model_args.vocab_size),
            yb.view(-1),
            ignore_index=-1,
        )
        
        mtp = torch.tensor(0.0)
        if logits_nxt2 is not None:
            mtp = F.cross_entropy(
                logits_nxt2[:, :-1].contiguous().view(-1, model_args.vocab_size),
                yb[:, 1:].contiguous().view(-1),
                ignore_index=-1,
            )
            
    print("\n--- Mock Forward Pass Results ---")
    print(f"Logits shape: {logits_nxt.shape}")
    print(f"Logits min/max/mean: {logits_nxt.min().item():.4f} / {logits_nxt.max().item():.4f} / {logits_nxt.mean().item():.4f}")
    print(f"Logits std: {logits_nxt.std().item():.4f}")
    print(f"Cross Entropy (against random targets): {ce.item():.4f}")
    print(f"MTP Loss: {mtp.item():.4f}")
    print(f"Router Z-loss: {z_loss.item():.4f}")
    
    # Let's check embeddings and output weights
    print("\n--- Weight Stats ---")
    print(f"emb.weight max: {model.emb.weight.abs().max().item():.4f}")
    print(f"head.weight max: {model.head.weight.abs().max().item():.4f}")
    print(f"decoder_norm scale max: {model.decoder_norm.weight.abs().max().item():.4f}")

if __name__ == "__main__":
    main()
