import os
import sys
import argparse
import urllib.request
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Add root folder to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.model import LasmoidV1, ModelArgs, Linear

# Muon Optimizer
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                grad, state = p.grad, self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                if len(p.shape) == 2:  # Newton-Schulz for 2D matrices
                    G = buf.clone()
                    a, b, c = 3.4445, -4.7750, 2.0315
                    X = G / (G.norm() + 1e-8)
                    for _ in range(5):
                        A = X @ X.T
                        B = A @ X
                        X = a * X + b * B + c * A @ B
                    update = X * (G.norm() + 1e-8)
                else:
                    update = buf
                p.add_(update, alpha=-lr)

def get_lr_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=500)
    args_cli = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Setup Checkpoint Directory
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # 1. Dataset Loading
    dataset_path = "input.txt"
    if not os.path.exists(dataset_path):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dataset_path)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text, allowed_special={"<|endoftext|>"}), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model_args = ModelArgs()
    max_seq_len = model_args.max_seq_len

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - max_seq_len - 1, (args_cli.batch_size,))
        x = torch.stack([d[i:i+max_seq_len] for i in ix]).to(device)
        y = torch.stack([d[i+1:i+max_seq_len+1] for i in ix]).to(device)
        return x, y

    # 2. Model Initialization
    model = LasmoidV1(model_args).to(device)
    print(f"Model initialized with dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")

    # Partition Parameters
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        # 2D weights updated by Muon (excluding embedding layers and gate routers)
        if len(p.shape) == 2 and "emb" not in name and "adj" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate)

    print("Training initiated...")
    model.train()
    
    for step in range(args_cli.max_iters):
        # LR Scheduling (Warmup + Cosine Decay)
        lr_mult = get_lr_multiplier(step, args_cli.max_iters, args_cli.warmup_steps)
        for g in opt_muon.param_groups:
            g['lr'] = 2e-3 * lr_mult
        for g in opt_adamw.param_groups:
            g['lr'] = args_cli.learning_rate * lr_mult

        opt_muon.zero_grad()
        opt_adamw.zero_grad()

        total_step_loss = 0.0
        total_step_ce = 0.0
        total_step_mtp = 0.0

        # Gradient Accumulation Loop
        for micro in range(args_cli.grad_accum):
            xb, yb = get_batch('train')
            
            # Encoder gets prompt context, Decoder gets full sequence to prevent leakage
            logits_next, logits_next_next, _, _ = model(xb, xb)
            
            # Next-token prediction loss
            ce_loss_next = F.cross_entropy(logits_next.view(-1, model_args.vocab_size), yb.view(-1))
            
            # MTP (next-next token) loss
            ce_loss_mtp = torch.tensor(0.0, device=device)
            if logits_next_next is not None:
                # yb shifted for t+2 prediction
                ce_loss_mtp = F.cross_entropy(logits_next_next.view(-1, model_args.vocab_size), yb[:, 1:].contiguous().view(-1))
            
            loss = ce_loss_next + 0.3 * ce_loss_mtp
            loss = loss / args_cli.grad_accum
            loss.backward()

            total_step_loss += loss.item() * args_cli.grad_accum
            total_step_ce += ce_loss_next.item()
            total_step_mtp += ce_loss_mtp.item()

        # Step optimizers
        opt_muon.step()
        opt_adamw.step()

        if step % 50 == 0:
            print(f"Step {step:4d} | Total Loss: {total_step_loss:.4f} | CE (t+1): {total_step_ce:.4f} | MTP (t+2): {total_step_mtp:.4f} | LR Scale: {lr_mult:.4f}", flush=True)

        # Checkpoint Saving
        if step > 0 and step % args_cli.save_interval == 0:
            ckpt_path = os.path.join(args_cli.checkpoint_dir, f"lasmoid_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'opt_muon_state': opt_muon.state_dict(),
                'opt_adamw_state': opt_adamw.state_dict(),
                'args': model_args
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Save final model
    final_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model parameters saved to {final_path}")

if __name__ == "__main__":
    train()
