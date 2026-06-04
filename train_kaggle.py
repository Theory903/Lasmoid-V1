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

# ── SESSION WATCHDOG ──────────────────────────────────────────────────
SESSION_START_TIME = time.time()
SESSION_MAX_SECONDS = 11 * 3600  # 11 hours — Kaggle limit is 12h; exit gracefully before

def session_time_remaining():
    return SESSION_MAX_SECONDS - (time.time() - SESSION_START_TIME)

def session_expired():
    return time.time() - SESSION_START_TIME >= SESSION_MAX_SECONDS


# ── MUON OPTIMIZER ────────────────────────────────────────────────────
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
    Streams samples from a HF dataset, tokenizes on-the-fly,
    and packs them into chunks of exactly `seq_len + 1` tokens.
    Automatically restarts on any error (network blip, shard EOF, etc.)
    """
    from datasets import load_dataset

    print(f"  Initializing stream: {dataset_name} (config={config_name})...")
    buffer = []
    eot_token = tokenizer.eot_token

    while True:
        try:
            ds = load_dataset(dataset_name, config_name, streaming=True, split="train",
                              trust_remote_code=True)
            for row in ds:
                # ── Normalise schema to plain text ───────────────────
                if "messages" in row:
                    # Chat format (WithinUsAI/claude_mythos_distilled_25k, UltraData-SFT)
                    msgs = row["messages"]
                    if isinstance(msgs, list):
                        text = ""
                        for msg in msgs:
                            if isinstance(msg, dict):
                                role = msg.get("role", "user")
                                content = msg.get("content", "")
                            else:
                                role, content = "user", str(msg)
                            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    else:
                        text = str(msgs)
                elif "conversations" in row:
                    # ShareGPT / UltraData-SFT alternative key
                    text = ""
                    for turn in (row["conversations"] or []):
                        role = turn.get("from", turn.get("role", "user"))
                        value = turn.get("value", turn.get("content", ""))
                        text += f"<|im_start|>{role}\n{value}<|im_end|>\n"
                elif "query" in row and "thinking" in row:
                    # HelioAI/Claude-Opus-4.8-DeepThink-462x-105M
                    text = (f"<|im_start|>user\n{row['query']}<|im_end|>\n"
                            f"<|im_start|>thought\n{row['thinking']}<|im_end|>\n")
                    if row.get("response"):
                        text += f"<|im_start|>assistant\n{row['response']}<|im_end|>\n"
                elif "content" in row:
                    # openbmb/Ultra-FineWeb-L3
                    text = row["content"]
                elif "text" in row:
                    text = row["text"]
                elif "instruction" in row:
                    # Instruction-tuning format
                    text = (f"<|im_start|>user\n{row['instruction']}<|im_end|>\n"
                            f"<|im_start|>assistant\n{row.get('output', row.get('response', ''))}<|im_end|>\n")
                else:
                    text = " ".join(str(v) for v in row.values() if isinstance(v, str))

                if not text.strip():
                    continue

                tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                buffer.extend(tokens)
                buffer.append(eot_token)

                while len(buffer) >= seq_len + 1:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len:]
                    yield chunk

        except Exception as e:
            print(f"  [stream] {dataset_name} error: {e!r} — restarting in 5s...", flush=True)
            time.sleep(5)


class MixedMultiTaskLoader:
    """Stochastically mixes batches from multiple streaming generators."""
    def __init__(self, generators, weights, batch_size, device):
        self.generators = generators
        self.batch_size = batch_size
        self.device = device
        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]

    def next_batch(self):
        xb, yb = [], []
        for _ in range(self.batch_size):
            gen = random.choices(self.generators, weights=self.probs, k=1)[0]
            chunk = next(gen)
            xb.append(torch.tensor(chunk[:-1], dtype=torch.long))
            yb.append(torch.tensor(chunk[1:], dtype=torch.long))
        return torch.stack(xb).to(self.device), torch.stack(yb).to(self.device)


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────
def save_checkpoint(model, opt_muon, opt_adamw, step, checkpoint_dir,
                    model_args, api=None, repo_id=None):
    ckpt_name = f"lasmoid_checkpoint_step_{step}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    latest_path = os.path.join(checkpoint_dir, "lasmoid_latest.pt")

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'opt_muon_state': opt_muon.state_dict(),
        'opt_adamw_state': opt_adamw.state_dict(),
        'args': model_args,
    }
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"  Checkpoint saved → {ckpt_path}", flush=True)

    if api and repo_id:
        try:
            print(f"  Syncing step {step} to HF Hub ({repo_id})...", flush=True)
            api.upload_file(path_or_fileobj=latest_path,
                            path_in_repo="lasmoid_latest.pt",
                            repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=ckpt_path,
                            path_in_repo=ckpt_name,
                            repo_id=repo_id, repo_type="model")
            print("  HF Hub upload complete ✓", flush=True)

            # Keep only the latest numbered checkpoint locally
            for f in os.listdir(checkpoint_dir):
                if f.startswith("lasmoid_checkpoint_step_") and f != ckpt_name:
                    try: os.remove(os.path.join(checkpoint_dir, f))
                    except: pass
        except Exception as e:
            print(f"  HF Hub upload failed: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters",       type=int,   default=50000)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--learning_rate",   type=float, default=6e-4)
    parser.add_argument("--warmup_steps",    type=int,   default=1000)
    parser.add_argument("--save_interval",   type=int,   default=500)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints")
    parser.add_argument("--session_hours",   type=float, default=11.0,
                        help="Exit gracefully after this many hours (for Kaggle auto-restart)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 5 steps on mock data to verify setup")
    args_cli = parser.parse_args()

    global SESSION_MAX_SECONDS
    SESSION_MAX_SECONDS = args_cli.session_hours * 3600

    # ── Device ──────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
    print(f"Training on device: {device.upper()}")

    # bfloat16 is supported on T4/A100 and MPS; avoids the fp16 inplace
    # gradient corruption bug in scaled-dot-product attention.
    amp_dtype = torch.bfloat16
    use_amp   = (device in ("cuda", "mps"))
    print(f"AMP: {'bfloat16' if use_amp else 'disabled'}")

    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # ── 1. Model ─────────────────────────────────────────────────────
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
        max_seq_len=512,
        max_batch_size=args_cli.batch_size,
    )

    model = LasmoidV1(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LasmoidV1: dim={model_args.dim}, layers={model_args.n_layers}, "
          f"heads={model_args.n_heads}")
    print(f"Total trainable parameters: {total_params:,}")

    # ── 2. Optimizers ─────────────────────────────────────────────────
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if len(p.shape) == 2 and "emb" not in name and "adj" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon  = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate, weight_decay=0.01)

    start_step = 0

    # ── 3. Hugging Face Hub ──────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN")
    api      = None
    repo_id  = None

    if hf_token:
        try:
            from huggingface_hub import HfApi, login, hf_hub_download
            login(token=hf_token, add_to_git_credential=False)
            # Bind token explicitly so api uses the right account regardless
            # of any pre-existing env var conflicts on Kaggle
            api      = HfApi(token=hf_token)
            username = api.whoami()["name"]
            repo_id  = f"{username}/lasmoid-100m"
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"HF Hub sync active → {repo_id}")

            # Auto-resume: find latest checkpoint on Hub
            try:
                files    = list(api.list_repo_files(repo_id=repo_id))
                pt_files = [f for f in files
                            if f.startswith("lasmoid_checkpoint_step_") and f.endswith(".pt")]
                if pt_files:
                    latest_file = max(pt_files,
                                      key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    print(f"  Found remote checkpoint: {latest_file}. Downloading...")
                    ckpt_path  = hf_hub_download(repo_id=repo_id, filename=latest_file,
                                                 token=hf_token)
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt_muon.load_state_dict(checkpoint['opt_muon_state'])
                    opt_adamw.load_state_dict(checkpoint['opt_adamw_state'])
                    start_step = checkpoint['step'] + 1
                    print(f"  Resumed from step {start_step} ✓")
                elif "lasmoid_latest.pt" in files:
                    ckpt_path  = hf_hub_download(repo_id=repo_id, filename="lasmoid_latest.pt",
                                                 token=hf_token)
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt_muon.load_state_dict(checkpoint['opt_muon_state'])
                    opt_adamw.load_state_dict(checkpoint['opt_adamw_state'])
                    start_step = checkpoint['step'] + 1
                    print(f"  Resumed from step {start_step} ✓")
            except Exception as e:
                print(f"  No checkpoint found or failed to load — starting fresh: {e}")
        except Exception as e:
            print(f"HF Hub init failed: {e}\n  → Training without sync.")

    # ── 4. Data Loaders ──────────────────────────────────────────────
    if args_cli.dry_run:
        print("Dry run active — using mock data.")
        def mock_gen():
            while True:
                yield [random.randint(0, 50256) for _ in range(model_args.max_seq_len + 1)]
        generators = [mock_gen()]
        weights    = [1.0]
    else:
        tokenizer  = tiktoken.get_encoding("gpt2")
        print("\nInitializing dataset streams:")

        # Dataset weights (must sum to 1.0)
        #   40% Ultra-FineWeb       — broad web text, language grounding
        #   25% UltraData-SFT       — instruction following
        #   20% Claude Mythos SFT   — rich chat / story / reasoning  
        #   15% DeepThink traces    — long-form chain-of-thought
        gen_web    = stream_packed_tokens(
            "openbmb/Ultra-FineWeb-L3",
            "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
            tokenizer, model_args.max_seq_len)

        gen_sft    = stream_packed_tokens(
            "openbmb/UltraData-SFT-2605",
            None,
            tokenizer, model_args.max_seq_len)

        gen_mythos = stream_packed_tokens(
            "WithinUsAI/claude_mythos_distilled_25k",
            None,
            tokenizer, model_args.max_seq_len)

        gen_opus   = stream_packed_tokens(
            "HelioAI/Claude-Opus-4.8-DeepThink-462x-105M",
            None,
            tokenizer, model_args.max_seq_len)

        generators = [gen_web, gen_sft, gen_mythos, gen_opus]
        weights    = [0.40,    0.25,    0.20,        0.15]

        print(f"\nDataset mix:")
        print(f"  40% openbmb/Ultra-FineWeb-L3 (web text)")
        print(f"  25% openbmb/UltraData-SFT-2605 (SFT instructions)")
        print(f"  20% WithinUsAI/claude_mythos_distilled_25k (chat/story)")
        print(f"  15% HelioAI/Claude-Opus-4.8-DeepThink-462x-105M (reasoning traces)")

    loader = MixedMultiTaskLoader(generators, weights, args_cli.batch_size, device)

    # ── 5. Training Loop ──────────────────────────────────────────────
    print(f"\nStarting training: step {start_step} → {args_cli.max_iters}")
    print(f"Session watchdog: will save & exit after {args_cli.session_hours:.1f}h\n")
    model.train()

    for step in range(start_step, args_cli.max_iters):
        # ── Session watchdog — exit gracefully before Kaggle kills us ──
        if session_expired():
            print(f"\n⏰  Session limit reached ({args_cli.session_hours:.1f}h). "
                  f"Saving checkpoint and exiting for auto-restart...", flush=True)
            save_checkpoint(model, opt_muon, opt_adamw, step, args_cli.checkpoint_dir,
                            model_args, api, repo_id)
            print("  Re-run this notebook to resume automatically from this step.", flush=True)
            sys.exit(0)

        t0 = time.time()

        # ── LR schedule ─────────────────────────────────────────────
        lr_mult = get_lr_multiplier(step, args_cli.max_iters, args_cli.warmup_steps)
        for g in opt_muon.param_groups:
            g['lr'] = 2e-3 * lr_mult
        for g in opt_adamw.param_groups:
            g['lr'] = args_cli.learning_rate * lr_mult

        opt_muon.zero_grad()
        opt_adamw.zero_grad()

        total_loss = total_ce = total_mtp = total_z = 0.0

        # ── Gradient accumulation ────────────────────────────────────
        for micro in range(args_cli.grad_accum):
            x, y = loader.next_batch()

            # bfloat16 autocast — no GradScaler needed; avoids fp16 inplace bug
            ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                   if device == "cuda"
                   else torch.amp.autocast(device_type="cpu", dtype=amp_dtype)
                   if use_amp
                   else torch.no_grad().__class__())          # fallback: no autocast

            with ctx:
                logits_next, logits_next_next, _, _ = model(x, x)
                z_loss = model.last_z_loss

                ce_next = F.cross_entropy(
                    logits_next.view(-1, model_args.vocab_size), y.view(-1))

                ce_mtp = torch.tensor(0.0, device=device)
                if logits_next_next is not None:
                    ce_mtp = F.cross_entropy(
                        logits_next_next[:, :-1].contiguous().view(-1, model_args.vocab_size),
                        y[:, 1:].contiguous().view(-1))

                loss = (ce_next
                        + 0.3 * ce_mtp
                        + model_args.router_z_loss_coeff * z_loss)
                loss = loss / args_cli.grad_accum

            loss.backward()   # plain backward — no scaler needed with bf16

            total_loss += loss.item() * args_cli.grad_accum
            total_ce   += ce_next.item()
            total_mtp  += ce_mtp.item()
            total_z    += z_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adamw.step()

        dt = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────
        if step % 10 == 0:
            elapsed_h = (time.time() - SESSION_START_TIME) / 3600
            remaining_h = (SESSION_MAX_SECONDS - (time.time() - SESSION_START_TIME)) / 3600
            print(
                f"Step {step:6d} | Loss: {total_loss:.4f} | CE: {total_ce:.4f} | "
                f"MTP: {total_mtp:.4f} | Z: {total_z:.4f} | "
                f"LR: {lr_mult:.4f} | {dt*1000:.0f}ms | "
                f"Session: {elapsed_h:.1f}h/{remaining_h:.1f}h left",
                flush=True)

        # ── Checkpointing ─────────────────────────────────────────────
        if step > 0 and step % args_cli.save_interval == 0:
            save_checkpoint(model, opt_muon, opt_adamw, step, args_cli.checkpoint_dir,
                            model_args, api, repo_id)

    # ── Final save ───────────────────────────────────────────────────
    final_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final weights → {final_path}", flush=True)

    if api and repo_id:
        try:
            api.upload_file(path_or_fileobj=final_path,
                            path_in_repo="lasmoid_final.pt",
                            repo_id=repo_id, repo_type="model")
            print("Final weights synced to HF Hub ✓", flush=True)
        except Exception as e:
            print(f"Final upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
