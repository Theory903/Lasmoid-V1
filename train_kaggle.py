import os
import sys
import time
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Add root folder to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.model import LasmoidV1, ModelArgs

# Muon Optimizer Verbatim
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


# ── MULTI-TASK DATASET STREAMER ───────────────────────────────────────
def stream_packed_tokens(dataset_name, config_name, tokenizer, seq_len):
    """
    Streams samples from Hugging Face dataset, formats them,
    tokenizes on-the-fly, and packs them into chunks of size `seq_len`.
    """
    from datasets import load_dataset
    print(f"Initializing streaming generator for {dataset_name} (config: {config_name})...")
    
    buffer = []
    eot_token = tokenizer.eot_token
    
    while True:
        try:
            # Load split in streaming mode
            ds = load_dataset(dataset_name, config_name, streaming=True, split="train")
            for row in ds:
                # 1. Format text content based on dataset schema
                if "messages" in row: # WithinUsAI/claude_mythos_distilled_25k
                    text = ""
                    for msg in row["messages"]:
                        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                elif "query" in row and "thinking" in row: # HelioAI/Claude-Opus-4.8-DeepThink-462x-105M
                    text = f"<|im_start|>user\n{row['query']}<|im_end|>\n<|im_start|>thought\n{row['thinking']}<|im_end|>\n"
                elif "content" in row: # openbmb/Ultra-FineWeb-L3
                    text = row["content"]
                elif "text" in row:
                    text = row["text"]
                else:
                    text = str(row)
                
                # 2. Tokenize text
                tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                buffer.extend(tokens)
                buffer.append(eot_token)
                
                # 3. Yield full sequence blocks
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len:]
                    yield chunk
        except Exception as e:
            print(f"Generator {dataset_name} encountered exception: {e}. Restarting stream...", flush=True)
            time.sleep(2)


class MixedMultiTaskLoader:
    """
    Stochastically mixes batches from multiple streaming generators.
    """
    def __init__(self, generators, weights, batch_size, device):
        self.generators = generators
        self.weights = weights
        self.batch_size = batch_size
        self.device = device
        
        # Normalize weights to probabilities
        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]
        
    def next_batch(self):
        xb, yb = [], []
        for _ in range(self.batch_size):
            # Select generator stochastically
            gen = random.choices(self.generators, weights=self.probs, k=1)[0]
            chunk = next(gen)
            xb.append(torch.tensor(chunk[:-1], dtype=torch.long))
            yb.append(torch.tensor(chunk[1:], dtype=torch.long))
            
        return torch.stack(xb).to(self.device), torch.stack(yb).to(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick local test on fake data")
    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device.upper()}")

    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # 1. Initialize 100M Parameter Config (verified via param sweep)
    model_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        head_dim=64,
        q_lora_rank=64,
        o_lora_rank=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=1024,
        max_seq_len=512,  # Fit comfortably in memory
        max_batch_size=args_cli.batch_size,
    )
    
    model = LasmoidV1(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LasmoidV1 initialized with: dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")
    print(f"Total trainable parameters: {total_params:,}")

    # 2. Setup Optimizers
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if len(p.shape) == 2 and "emb" not in name and "adj" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate, weight_decay=0.01)

    start_step = 0
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # 3. Setup Hugging Face Hub Integration
    hf_token = os.getenv("HF_TOKEN")
    api = None
    repo_id = None
    
    if hf_token:
        try:
            from huggingface_hub import HfApi, login, hf_hub_download
            login(token=hf_token)
            api = HfApi()
            username = api.whoami(token=hf_token)["name"]
            repo_id = f"{username}/lasmoid-100m"
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"Hugging Face sync active. Repository: {repo_id}")
            
            # Auto-Resume logic: Find the latest checkpoint on HF Hub
            try:
                files = api.list_repo_files(repo_id=repo_id)
                pt_files = [f for f in files if f.startswith("lasmoid_checkpoint_step_") and f.endswith(".pt")]
                if pt_files:
                    # Find maximum step number
                    latest_file = max(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    print(f"Found latest remote checkpoint: {latest_file}. Downloading...")
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename=latest_file)
                    
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt_muon.load_state_dict(checkpoint['opt_muon_state'])
                    opt_adamw.load_state_dict(checkpoint['opt_adamw_state'])
                    start_step = checkpoint['step'] + 1
                    if scaler and 'scaler_state' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler_state'])
                    print(f"Successfully resumed training from step {start_step}!")
                elif "lasmoid_latest.pt" in files:
                    print("Found lasmoid_latest.pt checkpoint. Downloading...")
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename="lasmoid_latest.pt")
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt_muon.load_state_dict(checkpoint['opt_muon_state'])
                    opt_adamw.load_state_dict(checkpoint['opt_adamw_state'])
                    start_step = checkpoint['step'] + 1
                    if scaler and 'scaler_state' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler_state'])
                    print(f"Successfully resumed training from step {start_step}!")
            except Exception as e:
                print(f"No checkpoint found or failed to load. Starting from scratch: {e}")
        except Exception as e:
            print(f"Failed to initialize Hugging Face Hub integration: {e}. Running without sync.")

    # 4. Initialize Data Loader
    if args_cli.dry_run:
        print("Dry run active. Generating mock sequences...")
        # Create a mock generator that yields random integer sequences
        def mock_generator():
            while True:
                yield [random.randint(0, 50256) for _ in range(model_args.max_seq_len + 1)]
        
        generators = [mock_generator()]
        weights = [1.0]
        loader = MixedMultiTaskLoader(generators, weights, args_cli.batch_size, device)
    else:
        # Load Hugging Face streaming loaders
        tokenizer = tiktoken.get_encoding("gpt2")
        
        gen_web = stream_packed_tokens(
            "openbmb/Ultra-FineWeb-L3", 
            "Ultra-FineWeb-L3-en-Multi-Style-Synthetic", 
            tokenizer, 
            model_args.max_seq_len
        )
        gen_mythos = stream_packed_tokens(
            "WithinUsAI/claude_mythos_distilled_25k", 
            None, 
            tokenizer, 
            model_args.max_seq_len
        )
        gen_opus = stream_packed_tokens(
            "HelioAI/Claude-Opus-4.8-DeepThink-462x-105M", 
            None, 
            tokenizer, 
            model_args.max_seq_len
        )
        
        generators = [gen_web, gen_mythos, gen_opus]
        weights = [0.5, 0.35, 0.15] # 50% Web, 35% SFT instructions, 15% long-form reasoning traces
        loader = MixedMultiTaskLoader(generators, weights, args_cli.batch_size, device)

    # 5. Training Loop
    print(f"Starting training loop from step {start_step} to {args_cli.max_iters}...")
    model.train()
    
    for step in range(start_step, args_cli.max_iters):
        t0 = time.time()
        
        # Learning Rate Schedule scale
        lr_mult = get_lr_multiplier(step, args_cli.max_iters, args_cli.warmup_steps)
        for g in opt_muon.param_groups:
            g['lr'] = 2e-3 * lr_mult
        for g in opt_adamw.param_groups:
            g['lr'] = args_cli.learning_rate * lr_mult

        opt_muon.zero_grad()
        opt_adamw.zero_grad()

        total_loss = 0.0
        total_ce = 0.0
        total_mtp = 0.0
        total_z = 0.0

        # Gradient Accumulation
        for micro in range(args_cli.grad_accum):
            x, y = loader.next_batch()
            
            # Autocast context for memory efficiency and speed
            autocast_ctx = torch.amp.autocast(
                device_type="cuda" if device == "cuda" else "cpu", 
                dtype=torch.float16 if device == "cuda" else torch.bfloat16
            )
            
            with autocast_ctx:
                # Encoder-Decoder forward path (preventing prompt future leakage)
                # Next-token logits, next-next-token logits, concept DB, Z-loss
                logits_next, logits_next_next, _, _ = model(x, x)
                z_loss = model.last_z_loss
                
                # Next-token prediction loss
                ce_loss_next = F.cross_entropy(logits_next.view(-1, model_args.vocab_size), y.view(-1))
                
                # MTP (next-next token) loss
                ce_loss_mtp = torch.tensor(0.0, device=device)
                if logits_next_next is not None:
                    ce_loss_mtp = F.cross_entropy(
                        logits_next_next[:, :-1].contiguous().view(-1, model_args.vocab_size), 
                        y[:, 1:].contiguous().view(-1)
                    )
                
                # Composite loss with MoE Z-loss scaling
                loss = ce_loss_next + 0.3 * ce_loss_mtp + model_args.router_z_loss_coeff * z_loss
                loss = loss / args_cli.grad_accum
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * args_cli.grad_accum
            total_ce += ce_loss_next.item()
            total_mtp += ce_loss_mtp.item()
            total_z += z_loss.item()

        # Step optimizers
        if scaler:
            scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adamw)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt_muon)
            scaler.step(opt_adamw)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

        dt = time.time() - t0
        
        # Logging
        if step % 10 == 0:
            print(
                f"Step {step:6d} | Loss: {total_loss:.4f} | CE: {total_ce:.4f} | "
                f"MTP: {total_mtp:.4f} | Z-Loss: {total_z:.4f} | LR Scale: {lr_mult:.4f} | "
                f"Time: {dt*1000:.1f}ms", 
                flush=True
            )

        # Checkpointing and Syncing to Hugging Face
        if step > 0 and step % args_cli.save_interval == 0:
            ckpt_name = f"lasmoid_checkpoint_step_{step}.pt"
            ckpt_path = os.path.join(args_cli.checkpoint_dir, ckpt_name)
            latest_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_latest.pt")
            
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'opt_muon_state': opt_muon.state_dict(),
                'opt_adamw_state': opt_adamw.state_dict(),
                'args': model_args
            }
            if scaler:
                checkpoint['scaler_state'] = scaler.state_dict()
                
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, latest_path)
            print(f"Checkpoint saved locally to {ckpt_path}", flush=True)
            
            if hf_token and api and repo_id:
                try:
                    print(f"Syncing step {step} checkpoint to Hugging Face Hub...", flush=True)
                    api.upload_file(
                        path_or_fileobj=latest_path,
                        path_in_repo="lasmoid_latest.pt",
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=ckpt_name,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    print("Hugging Face upload complete ✓", flush=True)
                    
                    # Clean up old local checkpoints to save disk space
                    local_files = os.listdir(args_cli.checkpoint_dir)
                    for f in local_files:
                        if f.startswith("lasmoid_checkpoint_step_") and f != ckpt_name:
                            try:
                                os.remove(os.path.join(args_cli.checkpoint_dir, f))
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Hugging Face upload failed: {e}", flush=True)

    # Save final model
    final_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model weights saved to {final_path}", flush=True)
    if hf_token and api and repo_id:
        try:
            api.upload_file(
                path_or_fileobj=final_path,
                path_in_repo="lasmoid_final.pt",
                repo_id=repo_id,
                repo_type="model"
            )
            print("Final weights synced to Hugging Face Hub ✓", flush=True)
        except Exception as e:
            print(f"Final weights upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
