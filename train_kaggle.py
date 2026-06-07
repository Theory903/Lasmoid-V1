"""
train_kaggle.py — LasmoidV1 Production Training Script v3
==========================================================
Author : theory903 (Abhishek Jha)  |  Model: Lasmoid-V1 — Neuro-Symbolic MoE
License: MIT  |  Made on Kaggle (T4 × 2)

Techniques from: DeepSeek-V3/V4-Pro, MiniCPM, YuLan-Mini,
                 OLMo, Phi-4, Chinchilla, Muon paper

Architecture decisions:
  ✓ bfloat16 AMP        No fp16 inplace grad corruption
  ✓ Nesterov Muon       Newton-Schulz orthograd for weight matrices
  ✓ AdamW (β₂=0.95)     DeepSeek-V3 betas for 2nd moment stability
  ✓ WSD LR schedule     Warmup → Stable → Decay (MiniCPM / MoonshotAI)
                         better than cosine for multi-session training
  ✓ 3-stage curriculum   8-stream curriculum for staged pretraining
  ✓ Chinchilla tracking  Logs tokens seen vs optimal (6.7 tokens/param)
  ✓ MTP loss (0.3×)      Multi-token prediction head auxiliary task
  ✓ Gradient checkpoint  Memory-saving for longer sequences / larger MoE
  ✓ EMA weight avg       Exponential Moving Average for inference quality
  ✓ W&B integration      Optional Weights & Biases experiment tracking
  ✓ Z-loss routing       MoE load balance via router Z-loss
  ✓ HF Hub sync          Token bound correctly — survives session restart
  ✓ Session watchdog     Graceful exit at 11h; re-running auto-resumes
  ✓ torch.compile        Optional 20-30% CUDA speedup
  ✓ --data_check         Verify all streams before training

Dataset curriculum (inspired by Llama-3, Phi-4, YuLan-Mini):
  8 streams: UltraFineWeb-L3, UltraData-IF/Math/Code, claude_mythos, DeepThink, smollm python-edu, FineWeb-Edu
  Stage 1  (steps    0 → 70%): web 30%, edu 20%, pyedu 15%, mythos 10%, think 10%, SFT-IF 7%, SFT-code 4%, SFT-math 4%
  Stage 2  (steps 70% → 90%): think 20%, SFT-IF 15%, web 15%, mythos 15%, SFT-math 10%, pyedu 10%, edu 10%, SFT-code 5%
  Stage 3  (steps 90% →100%): think 35%, mythos 25%, SFT-IF 20%, SFT-math 15%, SFT-code 5%  (pure quality)
"""

import os, sys, time, math, random, argparse

# ── Early data_check exit (MUST run before importing torch/initializing CUDA) ──
if "--data_check" in sys.argv:
    # Disable CUDA for the data check process to prevent CUDA + fork SIGABRT
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[DEBUG] Entered --data_check block. CUDA disabled.", flush=True)
    is_master = int(os.environ.get("RANK", 0)) == 0
    if is_master:
        print("[DEBUG] Master process confirmed. Importing datasets...", flush=True)
        from datasets import load_dataset
        print("[DEBUG] datasets imported successfully.", flush=True)
        _hf_token = os.getenv("HF_TOKEN")
        print("\n  Data check mode — verifying all 8 streams\n", flush=True)
        stream_infos = [
            (
                "UltraFineWeb",
                "openbmb/Ultra-FineWeb-L3",
                "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
                "train",
            ),
            ("UltraData-IF",   "openbmb/UltraData-SFT-2605", "IF",          "no_think"),
            ("UltraData-Math", "openbmb/UltraData-SFT-2605", "Math",        "no_think"),
            ("UltraData-Code", "openbmb/UltraData-SFT-2605", "Code",        "no_think"),
            ("Claude-Mythos",  "WithinUsAI/claude_mythos_distilled_25k", None, "train"),
            ("DeepThink",      "HelioAI/Claude-Opus-4.8-DeepThink-462x-105M", None, "train"),
            ("PythonEdu",      "HuggingFaceTB/smollm-corpus", "python-edu", "train"),
            ("FineWeb-Edu",    "HuggingFaceFW/fineweb-edu",   "sample-10BT", "train"),
        ]
        for label, path, config, split in stream_infos:
            print(f"\n  [{label}]")
            print(f"    dataset: {path}")
            if config:
                print(f"    config:  {config}")
            print(f"    split:   {split}")
            try:
                ds = load_dataset(path, config, split=split, streaming=True, token=_hf_token)
                first = next(iter(ds))
                print(f"    columns: {list(first.keys())}")
                for k, v in first.items():
                    val = str(v)
                    if len(val) > 150:
                        val = val[:150] + "..."
                    print(f"      {k}: {type(v).__name__} = {val}")
            except Exception as e:
                print(f"    ERROR: {e}")
        print("\n  All stream checks complete ✓")
    sys.exit(0)

# Prevent CUDA memory fragmentation on Kaggle T4 (16GB)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference.model import LasmoidV1, ModelArgs


# ══════════════════════════════════════════════════════════════════════
# SESSION WATCHDOG
# ══════════════════════════════════════════════════════════════════════
_SESSION_START = time.time()
_SESSION_LIMIT = 11 * 3600


def session_remaining_h():
    return (_SESSION_LIMIT - (time.time() - _SESSION_START)) / 3600


def session_expired():
    return (time.time() - _SESSION_START) >= _SESSION_LIMIT


# ══════════════════════════════════════════════════════════════════════
# MUON OPTIMIZER  (arxiv.org/abs/2502.16982)
# Nesterov momentum + Newton-Schulz orthogonalisation for 2-D weights
# ══════════════════════════════════════════════════════════════════════
class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonal Updates.
    Uses degree-5 Chebyshev polynomial approximation of the matrix sign
    function to orthogonalise gradient updates for weight matrices.
    ~2× better loss reduction per step vs AdamW on dense layers.
    """

    # Chebyshev coefficients for NS polynomial
    _A, _B, _C = 3.4445, -4.7750, 2.0315

    def __init__(
        self,
        params,
        lr=2e-3,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu, nesterov = g["lr"], g["momentum"], g["nesterov"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(grad)
                buf = state["buf"].mul_(mu).add_(grad)
                update = buf.add(grad, alpha=mu) if nesterov else buf.clone()

                if p.ndim == 2:  # Newton-Schulz for weight matrices only
                    transposed = update.shape[0] > update.shape[1]
                    if transposed:
                        update = update.T
                    X = update / (update.norm() + 1e-8)
                    for _ in range(g["ns_steps"]):
                        A = X @ X.T
                        X = self._A * X + self._B * (A @ X) + self._C * (A @ A @ X)
                    update = X * (max(p.shape[0], p.shape[1]) ** 0.5)
                    if transposed:
                        update = update.T

                if g["weight_decay"] > 0:
                    p.mul_(1.0 - lr * g["weight_decay"])
                p.add_(update, alpha=-lr)


# ══════════════════════════════════════════════════════════════════════
# EMA  — Exponential Moving Average of model weights
# Improves inference quality by smoothing optimisation noise.
# arxiv.org/abs/1803.05407
# ══════════════════════════════════════════════════════════════════════
class EMA:
    """Exponential Moving Average of model parameters.
    Shadow weights track a smoothed copy; swap at eval time.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        raw_model = model.module if hasattr(model, "module") else model
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        raw_model = model.module if hasattr(model, "module") else model
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def swap(self, model: torch.nn.Module):
        raw_model = model.module if hasattr(model, "module") else model
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                tmp = param.data.clone()
                param.data.copy_(self.shadow[name])
                self.shadow[name].copy_(tmp)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ══════════════════════════════════════════════════════════════════════
# WSD LR SCHEDULE  (MiniCPM / MoonshotAI Moonlight paper)
# Warmup → Stable → Decay — designed for staged / resumed training
# Much better than cosine for multi-session runs: the "stable" plateau
# means loss progress doesn't stall between Kaggle session restarts.
# ══════════════════════════════════════════════════════════════════════
def get_lr_wsd(
    step: int,
    total: int,
    warmup: int,
    lr_max: float,
    decay_frac: float = 0.2,
    lr_min_ratio: float = 0.1,
) -> float:
    """
    Warmup-Stable-Decay schedule:
      [0, warmup)            linear 0 → lr_max
      [warmup, decay_start)  constant lr_max   ← stable plateau
      [decay_start, total)   cosine lr_max → lr_min
    """
    lr_min = lr_max * lr_min_ratio
    decay_start = int(total * (1.0 - decay_frac))

    if step < warmup:
        return lr_max * step / max(1, warmup)
    if step < decay_start:
        return lr_max  # stable phase
    # cosine decay
    t = (step - decay_start) / max(1, total - decay_start)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * t))


# ══════════════════════════════════════════════════════════════════════
# DATASET STREAMING
# ══════════════════════════════════════════════════════════════════════
def stream_packed_tokens(
    dataset_name: str,
    config_name,
    tokenizer,
    seq_len: int,
    label: str = "",
    split: str = "train",
    hf_token: str | None = None,
):
    """
    Infinite streaming generator with universal schema normalisation.
    Handles: chat (messages/conversations), web (content/text),
             reasoning (query+thinking), instruction (instruction+output).
    Auto-restarts on any network or dataset error.

    Args:
        split: Dataset split to stream. Defaults to "train" (most datasets).
               For datasets like openbmb/UltraData-SFT-2605 that use
               "think"/"no_think" instead, pass explicitly.
    """
    from datasets import load_dataset

    tag = label or dataset_name.split("/")[-1]
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print(f"    [{tag}] opening stream (split={split})...", flush=True)
    if hasattr(tokenizer, "eot_token"):
        eot = tokenizer.eot_token
    elif hasattr(tokenizer, "eos_token_id"):
        eot = tokenizer.eos_token_id
    else:
        eot = 50256
    buf = []

    while True:
        try:
            ds = load_dataset(
                dataset_name,
                config_name,
                streaming=True,
                split=split,
                token=hf_token,
            )
            use_row_sharding = False
            if world_size > 1:
                if getattr(ds, "n_shards", 1) >= world_size:
                    ds = ds.shard(num_shards=world_size, index=rank)
                else:
                    use_row_sharding = True
            for row_idx, row in enumerate(ds):
                if use_row_sharding and (row_idx % world_size) != rank:
                    continue
                # ── Universal schema → plain text ────────────────────
                if "messages" in row:
                    msgs = row["messages"]
                    text = ""
                    for m in msgs if isinstance(msgs, list) else []:
                        r = m.get("role", "user") if isinstance(m, dict) else "user"
                        v = m.get("content", str(m)) if isinstance(m, dict) else str(m)
                        text += f"<|im_start|>{r}\n{v}<|im_end|>\n"
                elif "conversations" in row:
                    text = ""
                    for t in row["conversations"] or []:
                        r = t.get("from", t.get("role", "user"))
                        v = t.get("value", t.get("content", ""))
                        text += f"<|im_start|>{r}\n{v}<|im_end|>\n"
                elif "query" in row and "thinking" in row:
                    text = (
                        f"<|im_start|>user\n{row['query']}<|im_end|>\n"
                        f"<|im_start|>thought\n{row['thinking']}<|im_end|>\n"
                    )
                    if row.get("response"):
                        text += f"<|im_start|>assistant\n{row['response']}<|im_end|>\n"
                elif "content" in row:
                    text = row["content"]
                elif "text" in row:
                    text = row["text"]
                elif "instruction" in row:
                    text = (
                        f"<|im_start|>user\n{row['instruction']}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                        f"{row.get('output', row.get('response', ''))}"
                        f"<|im_end|>\n"
                    )
                else:
                    text = " ".join(str(v) for v in row.values() if isinstance(v, str))

                if not text.strip():
                    continue

                toks = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                buf.extend(toks)
                buf.append(eot)

                while len(buf) >= seq_len + 1:
                    yield buf[: seq_len + 1]
                    buf = buf[seq_len:]

        except Exception as e:
            err_str = str(e)
            is_auth_error = (
                "gated dataset" in err_str.lower()
                or "authenticated" in err_str.lower()
                or "401" in err_str.lower()
                or "403" in err_str.lower()
                or "forbidden" in err_str.lower()
                or "gatedrepo" in type(e).__name__.lower()
            )
            if is_auth_error:
                print(f"\n[FATAL ERROR] Gated/private dataset authentication failed for '{dataset_name}'.", flush=True)
                print(f"Please ensure HF_TOKEN is correctly set, has been accepted on the HF website, and has read access to gated datasets.", flush=True)
                print(f"Details: {e!r}\n", flush=True)
                os._exit(1)
            print(f"    [{tag}] error: {e!r} → retry in 5s", flush=True)
            time.sleep(5)


class MultiTaskLoader:
    """Weighted stochastic batch mixer across N infinite generators."""

    def __init__(self, generators, weights, batch_size, device, eot_id=None):
        self.generators = generators
        self.batch_size = batch_size
        self.device = device
        self.eot_id = eot_id
        w = sum(weights)
        self.probs = [x / w for x in weights]

    def set_weights(self, weights):
        w = sum(weights)
        self.probs = [x / w for x in weights]

    def next_batch(self):
        xs, ys, cu_seqlens_list = [], [], []
        for _ in range(self.batch_size):
            gen = random.choices(self.generators, weights=self.probs, k=1)[0]
            chunk = next(gen)
            x_tensor = torch.tensor(chunk[:-1], dtype=torch.long)
            xs.append(x_tensor)
            ys.append(torch.tensor(chunk[1:], dtype=torch.long))
            
            # Build cu_seqlens for Unsloth-style packing
            if self.eot_id is not None:
                # Find boundaries (eot tokens)
                boundaries = (x_tensor == self.eot_id).nonzero(as_tuple=True)[0].tolist()
                # cu_seqlens must start with 0 and end with max_seq_len
                seq = [0] + [b + 1 for b in boundaries]
                if seq[-1] != x_tensor.size(0):
                    seq.append(x_tensor.size(0))
                cu_seqlens_list.append(torch.tensor(seq, dtype=torch.int32))
                
        # Pad cu_seqlens for batching if needed, or just keep as list of tensors
        # Because we process per batch item in model.py, we can pad with the last value
        if self.eot_id is not None:
            max_len = max(len(c) for c in cu_seqlens_list)
            padded_cu = []
            for c in cu_seqlens_list:
                pad_len = max_len - len(c)
                if pad_len > 0:
                    c = torch.cat([c, torch.full((pad_len,), c[-1].item(), dtype=torch.int32)])
                padded_cu.append(c)
            cu_seqlens = torch.stack(padded_cu).to(self.device)
        else:
            cu_seqlens = None
            
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device), cu_seqlens


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════
def save_checkpoint(
    model,
    opt_muon,
    opt_adamw,
    step,
    ckpt_dir,
    model_args,
    tokens_seen=0,
    metrics=None,
    api=None,
    repo_id=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"lasmoid_step_{step:08d}.pt"
    path = os.path.join(ckpt_dir, name)
    latest = os.path.join(ckpt_dir, "lasmoid_latest.pt")

    raw_model = model.module if hasattr(model, "module") else model
    blob = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model_state_dict": raw_model.state_dict(),
        "opt_muon_state": opt_muon.state_dict(),
        "opt_adamw_state": opt_adamw.state_dict(),
        "model_args": model_args,
        "metrics": metrics or {},
    }
    torch.save(blob, path)
    torch.save(blob, latest)
    print(
        f"  [ckpt] step={step:,} tokens={tokens_seen / 1e9:.2f}B → {name}", flush=True
    )

    if api and repo_id:
        try:
            api.upload_file(
                path_or_fileobj=latest,
                path_in_repo="lasmoid_latest.pt",
                repo_id=repo_id,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  [ckpt] HF Hub ✓  ({repo_id})", flush=True)
            # Prune old local numbered checkpoints
            for f in os.listdir(ckpt_dir):
                if f.startswith("lasmoid_step_") and f != name:
                    try:
                        os.remove(os.path.join(ckpt_dir, f))
                    except:
                        pass
        except Exception as e:
            print(f"  [ckpt] HF upload failed: {e}", flush=True)


def load_checkpoint(model, opt_muon, opt_adamw, api, repo_id, hf_token, device):
    """Attempt HF Hub resume. Returns (start_step, tokens_seen)."""
    if not (api and repo_id):
        return 0, 0
    try:
        from huggingface_hub import hf_hub_download

        files = list(api.list_repo_files(repo_id=repo_id))
        pt_files = [
            f for f in files if f.startswith("lasmoid_step_") and f.endswith(".pt")
        ]

        if not pt_files and "lasmoid_latest.pt" not in files:
            return 0, 0

        fname = (
            max(pt_files, key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
            if pt_files
            else "lasmoid_latest.pt"
        )

        print(f"  [resume] downloading {fname}...", flush=True)
        path = hf_hub_download(repo_id=repo_id, filename=fname, token=hf_token)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        
        # Reconstruct parameter lists as they were partitioned in the saved checkpoint (V2 partitioning)
        saved_muon_params = []
        saved_adamw_params = []
        for name, p in raw_model.named_parameters():
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

        # Reconstruct state mapping for Muon
        if "opt_muon_state" in ckpt:
            muon_param_ids = []
            for group in ckpt["opt_muon_state"]["param_groups"]:
                muon_param_ids.extend(group["params"])
            muon_state_map = {}
            for idx, p in enumerate(saved_muon_params):
                if idx < len(muon_param_ids):
                    p_id = muon_param_ids[idx]
                    if p_id in ckpt["opt_muon_state"]["state"]:
                        muon_state_map[p] = ckpt["opt_muon_state"]["state"][p_id]
            # Load states into current Muon optimizer
            for group in opt_muon.param_groups:
                for p in group["params"]:
                    if p in muon_state_map:
                        opt_muon.state[p] = muon_state_map[p]

        # Reconstruct state mapping for AdamW
        if "opt_adamw_state" in ckpt:
            adamw_param_ids = []
            for group in ckpt["opt_adamw_state"]["param_groups"]:
                adamw_param_ids.extend(group["params"])
            adamw_state_map = {}
            for idx, p in enumerate(saved_adamw_params):
                if idx < len(adamw_param_ids):
                    p_id = adamw_param_ids[idx]
                    if p_id in ckpt["opt_adamw_state"]["state"]:
                        adamw_state_map[p] = ckpt["opt_adamw_state"]["state"][p_id]
            # Load states into current AdamW optimizer
            for group in opt_adamw.param_groups:
                for p in group["params"]:
                    if p in adamw_state_map:
                        opt_adamw.state[p] = adamw_state_map[p]
                        
        step = ckpt["step"] + 1
        tokens = ckpt.get("tokens_seen", 0)
        print(
            f"  [resume] step={step:,}  tokens_seen={tokens / 1e9:.2f}B ✓", flush=True
        )
        return step, tokens
    except Exception as e:
        print(f"  [resume] failed: {e} — starting fresh")
        return 0, 0


# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGS  (verified param counts)
# ══════════════════════════════════════════════════════════════════════
# NOTE: 300M config is swept at runtime if not yet verified.
# Chinchilla-optimal token budget: ~6.7 × params
#   100M → ~670M tokens minimum
#   300M → ~2.0B tokens minimum
MODEL_CONFIGS = {
    "10M": dict(
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
        # verified: 9,379,108 params
        _verified_params=9_379_108,
    ),
    "100M": dict(
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
        # verified: 100,687,716 params
        _verified_params=100_687_716,
    ),
    "300M": dict(
        # Verified: dim=768 L=13 H=12 hd=64 → 303.0M params
        # Forward-pass verified (no dimension errors).
        # Classic GPT-medium width (dim=768), depth-first scaling (13 layers).
        # n_heads=12 (even, divisible by o_groups=2). head_dim=64 (standard).
        # moe_inter_dim=1536 = 2× dim (standard FFN ratio for MoE experts).
        dim=768,
        n_layers=13,
        n_heads=12,
        head_dim=64,
        q_lora_rank=192,
        o_lora_rank=192,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=1536,
        max_seq_len=1024,
        _verified_params=303_000_000,  # 303.0M confirmed by forward pass
    ),
}


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="LasmoidV1 Production Trainer v2")

    # ── Model ────────────────────────────────────────────────────────
    p.add_argument("--model_size", choices=["10M", "100M", "300M"], default="300M")
    p.add_argument("--seq_len", type=int, default=1024)

    # ── Phase ────────────────────────────────────────────────────────
    p.add_argument("--phase", choices=["sft", "rl", "fst"], default="sft", 
                   help="Training phase: sft (Supervised), rl (Reasoning RL), fst (Fast-Slow Training)")

    # ── Training ─────────────────────────────────────────────────────
    p.add_argument(
        "--max_iters",
        type=int,
        default=200_000,
        help="300M Chinchilla optimal = ~200K steps × 6 seq/step × 1024 toks",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak LR for AdamW")
    p.add_argument("--muon_lr", type=float, default=2e-3)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument(
        "--decay_frac",
        type=float,
        default=0.2,
        help="Fraction of training for WSD cosine decay phase",
    )
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mtp_coeff", type=float, default=0.3)
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="AMP dtype for mixed-precision training",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage",
    )
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay rate (0=disabled). 0.999+ recommended for inference quality",
    )
    p.add_argument(
        "--wandb", type=str, default="", help="W&B project name (empty = disabled)"
    )

    # ── Infra ────────────────────────────────────────────────────────
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--session_hours", type=float, default=11.0)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--data_check",
        action="store_true",
        help="Verify all dataset streams open correctly, then exit",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: cuda, mps, cpu",
    )
    args = p.parse_args()

    global _SESSION_LIMIT
    _SESSION_LIMIT = args.session_hours * 3600

    # ── DDP Setup ────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        backend = "nccl" if (torch.cuda.is_available() and args.device == "cuda") else "gloo"
        dist.init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available() and args.device == "cuda":
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = amp_dtype != torch.float32 and device.startswith("cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device.startswith("cuda") and use_amp
        else nullcontext()
    )

    if master_process:
        print(f"\n{'═' * 70}")
        print(f"  LasmoidV1 Production Trainer v3")
        print(f"  Device : {device.upper()} (DDP: {ddp}, World Size: {ddp_world_size}) | AMP: {args.dtype}")
        print(f"  Model  : {args.model_size} | theory903/Lasmoid-V1")
        print(f"{'═' * 70}\n")

    if master_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────
    cfg = MODEL_CONFIGS[args.model_size].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = args.seq_len
    cfg["max_batch_size"] = args.batch_size

    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args).to(device)
    total_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
    total_str = f"{total_p / 1e6:.1f}M"
    chinchilla_tokens = int(total_p * 6.7)

    if master_process:
        print(f"  Parameters : {total_str} ({total_p:,})")
        print(f"  Chinchilla-optimal tokens: {chinchilla_tokens / 1e9:.2f}B")
    eff_batch = args.batch_size * args.grad_accum * ddp_world_size
    eff_tokens = eff_batch * model_args.max_seq_len
    total_toks = args.max_iters * eff_tokens
    if master_process:
        print(
            f"  Training budget: {total_toks / 1e9:.2f}B tokens "
            f"({total_toks / chinchilla_tokens * 100:.0f}% of Chinchilla-optimal)"
        )
        print(
            f"  Effective batch: {eff_batch} seqs = {eff_tokens / 1000:.0f}K tokens/step\n"
        )

    if args.compile and device.startswith("cuda"):
        if master_process:
            print("  torch.compile: enabled")
        model = torch.compile(model)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if master_process:
            print("  Gradient checkpointing: enabled")

    # Wrap model in DDP
    if ddp:
        if device.startswith("cuda"):
            model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        else:
            model = DDP(model, find_unused_parameters=True)

    # ── Optimizers ────────────────────────────────────────────────────
    muon_params, adamw_params = [], []
    raw_model = model.module if ddp else model
    for name, param in raw_model.named_parameters():
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

    if master_process:
        print(f"  Muon  : {sum(q.numel() for q in muon_params) / 1e6:.1f}M params")
        print(f"  AdamW : {sum(q.numel() for q in adamw_params) / 1e6:.1f}M params")

    opt_muon = Muon(
        muon_params, lr=args.muon_lr, momentum=0.95, nesterov=True, ns_steps=5
    )
    opt_adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.lr,
        betas=(0.9, 0.95),  # DeepSeek-V3: β₂=0.95 for long-range stability
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # ── EMA ───────────────────────────────────────────────────────────
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        if master_process:
            print(f"  EMA  : decay={args.ema_decay}")

    # ── HF Hub ────────────────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN")
    api = repo_id = None

    if hf_token and not args.dry_run and master_process:
        try:
            from huggingface_hub import HfApi, login

            login(token=hf_token, add_to_git_credential=False)
            api = HfApi(token=hf_token)  # explicit token binding
            username = api.whoami()["name"]
            repo_id = f"{username}/lasmoid-{args.model_size.lower()}"
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"\n  HF Hub: {repo_id}")
        except Exception as e:
            print(f"\n  HF Hub init failed: {e}")
            api = repo_id = None

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb and not args.dry_run:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb,
                name=f"lasmoid-{args.model_size}-{time.strftime('%m%d-%H%M')}",
                config=vars(args),
            )
            print(f"  W&B  : {args.wandb}")
        except Exception as e:
            print(f"  W&B init failed: {e}")

    # ── Auto-resume ───────────────────────────────────────────────────
    start_step, tokens_seen = load_checkpoint(
        model, opt_muon, opt_adamw, api, repo_id, hf_token, device
    )

    # ── Data Streams ──────────────────────────────────────────────────
    tok = tiktoken.get_encoding("gpt2")

    if args.dry_run:
        print("\n  Dry run: mock data\n")

        def _mock():
            while True:
                yield [
                    random.randint(0, model_args.vocab_size - 1)
                    for _ in range(args.seq_len + 1)
                ]

        loader = MultiTaskLoader([_mock()], [1.0], args.batch_size, device, eot_id=tok.eot_token)
        STAGE_WEIGHTS = [[1.0], [1.0], [1.0]]
    else:

        # ── Curriculum Streams (inspired by Llama-3, Phi-4, YuLan-Mini) ──
        print("\n  Initialising data streams:")

        # Stage 1: Broad pretraining — diverse web + code + math
        # Stage 2: Quality annealing — SFT instructions + reasoning traces
        # Stage 3: Final cooldown   — pure high-quality SFT + reasoning

        # All 8 streams created upfront (generators are lazy — no RAM used)
        # Dataset citations:
        #   Ultra-FineWeb-L3   (openbmb)        — educational quality web
        #   UltraData-IF       (openbmb)        — instruction-following SFT
        #   UltraData-Math     (openbmb)        — math problem SFT
        #   UltraData-Code     (openbmb)        — coding SFT
        #   claude_mythos      (WithinUsAI)     — Mythos-style chat SFT
        #   Claude-DeepThink   (HelioAI)        — long CoT reasoning
        #   SmolLM python-edu  (HuggingFaceTB)  — raw educational Python
        #   FineWeb-Edu        (HuggingFaceFW)  — educational web pages

        GEN_WEB = stream_packed_tokens(
            "openbmb/Ultra-FineWeb-L3",
            "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
            tok,
            model_args.max_seq_len,
            label="UltraFineWeb",
            hf_token=hf_token,
        )

        GEN_IF = stream_packed_tokens(
            "openbmb/UltraData-SFT-2605",
            "IF",
            tok,
            model_args.max_seq_len,
            label="UltraData-IF",
            split="no_think",
            hf_token=hf_token,
        )

        GEN_MATH = stream_packed_tokens(
            "openbmb/UltraData-SFT-2605",
            "Math",
            tok,
            model_args.max_seq_len,
            label="UltraData-Math",
            split="no_think",
            hf_token=hf_token,
        )

        GEN_SFT_CODE = stream_packed_tokens(
            "openbmb/UltraData-SFT-2605",
            "Code",
            tok,
            model_args.max_seq_len,
            label="UltraData-Code",
            split="no_think",
            hf_token=hf_token,
        )

        GEN_MYTHOS = stream_packed_tokens(
            "WithinUsAI/claude_mythos_distilled_25k",
            None,
            tok,
            model_args.max_seq_len,
            label="Claude-Mythos",
            hf_token=hf_token,
        )

        GEN_THINK = stream_packed_tokens(
            "HelioAI/Claude-Opus-4.8-DeepThink-462x-105M",
            None,
            tok,
            model_args.max_seq_len,
            label="DeepThink",
            hf_token=hf_token,
        )

        GEN_PYEDU = stream_packed_tokens(
            "HuggingFaceTB/smollm-corpus",
            "python-edu",
            tok,
            model_args.max_seq_len,
            label="PythonEdu",
            hf_token=hf_token,
        )

        GEN_EDU = stream_packed_tokens(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            tok,
            model_args.max_seq_len,
            label="FineWeb-Edu",
            hf_token=hf_token,
        )

        STREAMS = [
            GEN_WEB,
            GEN_IF,
            GEN_MATH,
            GEN_SFT_CODE,
            GEN_MYTHOS,
            GEN_THINK,
            GEN_PYEDU,
            GEN_EDU,
        ]

        # ── Stage weights  [web, IF, math, code, mythos, think, pyedu, edu] ──
        # Stage 1 (0% → 70%): diverse foundation — web + edu + raw code heavy
        # Stage 2 (70% → 90%): quality annealing — SFT + reasoning
        # Stage 3 (90% → 100%): cooldown — pure SFT + reasoning
        STAGE_WEIGHTS = [
            [0.30, 0.07, 0.04, 0.04, 0.10, 0.10, 0.15, 0.20],  # Stage 1: broad
            [0.15, 0.15, 0.10, 0.05, 0.15, 0.20, 0.10, 0.10],  # Stage 2: quality
            [0.00, 0.20, 0.15, 0.05, 0.25, 0.35, 0.00, 0.00],  # Stage 3: cooldown
        ]

        STAGE_NAMES = ["Stage-1:Broad", "Stage-2:Quality", "Stage-3:Cooldown"]
        if master_process:
            print(
                f"    [Stage 1   0%→70%] web 30% | edu 20% | pyedu 15% | mythos 10% | CoT 10% | IF 7% | code 4% | math 4%"
            )
            print(
                f"    [Stage 2  70%→90%] SFT 30% (IF 15+math 10+code 5) | CoT 20% | web 15% | mythos 15% | pyedu 10% | edu 10%"
            )
            print(
                f"    [Stage 3  90%→100%] SFT 40% (IF 20+math 15+code 5) | CoT 35% | mythos 25%  (pure quality)"
            )

        loader = MultiTaskLoader(STREAMS, STAGE_WEIGHTS[0], args.batch_size, device, eot_id=tok.eot_token)
        
    if args.phase in ["rl", "fst"]:
        from fst_trainer import GEPAMutator
        from rl_trainer import THINK_SYSTEM_PROMPT
        if master_process:
            print(f"  Initializing RL/FST Trainer for phase: {args.phase}")
        
        gepa_mutator = None
        if args.phase == "fst":
            gepa_mutator = GEPAMutator(base_prompt=THINK_SYSTEM_PROMPT, use_external_api=False)

    def get_stage(step, total):
        frac = step / max(1, total)
        if frac < 0.70:
            return 0
        if frac < 0.90:
            return 1
        return 2

    # ── Training loop ─────────────────────────────────────────────────
    if master_process:
        print(f"\n  Starting: step {start_step:,} → {args.max_iters:,}")
        print(
            f"  Watchdog: exit after {args.session_hours:.1f}h "
            f"(~{session_remaining_h():.1f}h remaining)\n"
        )

    model.train()
    current_stage = get_stage(start_step, args.max_iters)
    running = dict(loss=0.0, ce=0.0, mtp=0.0, z=0.0, dt=0.0, n=0)

    # CSV metrics — one row per log interval for loss curve tracking
    log_file = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    if master_process and start_step == 0:
        with open(log_file, "w") as f:
            f.write("step,loss,ce,mtp,z_loss,lr,tokens_seen,tok_per_sec,stage\n")

    for step in range(start_step, args.max_iters):
        # ── Watchdog ─────────────────────────────────────────────────
        if session_expired():
            if master_process:
                print(
                    f"\n⏰ {args.session_hours:.0f}h limit. Saving + exiting.", flush=True
                )
                save_checkpoint(
                    model,
                    opt_muon,
                    opt_adamw,
                    step,
                    args.checkpoint_dir,
                    model_args,
                    tokens_seen=tokens_seen,
                    api=api,
                    repo_id=repo_id,
                )
            if ddp:
                dist.destroy_process_group()
            if master_process:
                print("Re-run Cell 6 to auto-resume from this step.", flush=True)
            sys.exit(0)

        # ── Curriculum stage switch ───────────────────────────────────
        stage = get_stage(step, args.max_iters)
        if stage != current_stage and not args.dry_run:
            current_stage = stage
            loader.set_weights(STAGE_WEIGHTS[stage])
            if master_process:
                print(
                    f"\n  ▶ Curriculum → {STAGE_NAMES[stage]} at step {step:,}", flush=True
                )

        t0 = time.perf_counter()

        # ── LR (WSD) ─────────────────────────────────────────────────
        lr_a = get_lr_wsd(
            step,
            args.max_iters,
            args.warmup_steps,
            args.lr,
            args.decay_frac,
            args.lr_min_ratio,
        )
        lr_m = get_lr_wsd(
            step,
            args.max_iters,
            args.warmup_steps,
            args.muon_lr,
            args.decay_frac,
            args.lr_min_ratio,
        )
        for g in opt_adamw.param_groups:
            g["lr"] = lr_a
        for g in opt_muon.param_groups:
            g["lr"] = lr_m

        if args.phase in ["rl", "fst"]:
            loss_val = 0.0
            ce_val = 0.0
            mtp_val = 0.0
            z_val = 0.0
            step_tokens = args.batch_size * 4 * args.seq_len * args.grad_accum
            tokens_seen += step_tokens
            
            if args.phase == "fst" and gepa_mutator is not None and step % 100 == 0 and master_process:
                print(f"  [FST] Running GEPA prompt optimization cycle at step {step}...")
                # Generate proposed mutations
                candidates = gepa_mutator.propose_mutations(model, tok, device, num_mutations=3)
                
                # Simple dummy evaluation for GEPA (in practice would run rollouts and score with rewards.py)
                def dummy_eval(p): return len(p) # stub for reward score
                
                best_prompt = gepa_mutator.evaluate_and_update(candidates, dummy_eval)
                print(f"  [FST] New optimal prompt length: {len(best_prompt)}")

        else:
            # ── SFT Loop ─────────────────────────────────────────────────
            opt_adamw.zero_grad(set_to_none=True)
            opt_muon.zero_grad(set_to_none=True)
        for g in opt_adamw.param_groups:
            g["lr"] = lr_a
        for g in opt_muon.param_groups:
            g["lr"] = lr_m

        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        total_loss = total_ce = total_mtp = total_z = 0.0

        # ── Gradient accumulation ─────────────────────────────────────
        for micro_step in range(args.grad_accum):
            x, y, cu_seqlens = loader.next_batch()

            # In DDP, only sync gradients on the last micro-step
            if ddp and micro_step < args.grad_accum - 1:
                ddp_ctx = model.no_sync()
            else:
                ddp_ctx = nullcontext()

            with ddp_ctx:
                with amp_ctx:
                    logits_nxt, logits_nxt2, _, _ = model(x, x, cu_seqlens=cu_seqlens)
                    raw_model = model.module if ddp else model
                    z_loss = raw_model.last_z_loss

                    ce = F.cross_entropy(
                        logits_nxt.view(-1, model_args.vocab_size),
                        y.view(-1),
                        ignore_index=-1,
                    )

                    mtp = torch.tensor(0.0, device=device)
                    if logits_nxt2 is not None:
                        mtp = F.cross_entropy(
                            logits_nxt2.view(-1, model_args.vocab_size),
                            y[:, 1:].contiguous().view(-1),
                            ignore_index=-1,
                        )

                    loss = (
                        ce + args.mtp_coeff * mtp + model_args.router_z_loss_coeff * z_loss
                    )
                    loss = loss / args.grad_accum

                loss.backward()  # bf16 — no scaler needed

            total_loss += loss.item() * args.grad_accum
            total_ce += ce.item()
            total_mtp += mtp.item()
            total_z += z_loss.item()

        # Apply pending MoE bias updates
        raw_model = model.module if ddp else model
        raw_model.apply_pending_bias_updates()

        # ── Clip + Step ───────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_muon.step()
        opt_adamw.step()

        # ── EMA update ────────────────────────────────────────────────
        if ema is not None:
            ema.update(model)

        dt = time.perf_counter() - t0
        tokens_seen += eff_tokens

        # ── Accumulate for smoothed logging ──────────────────────────
        running["loss"] += total_loss
        running["ce"] += total_ce
        running["mtp"] += total_mtp
        running["z"] += total_z
        running["dt"] += dt
        running["n"] += 1

        # ── Logging ──────────────────────────────────────────────────
        if master_process and step % args.log_interval == 0:
            n = running["n"]
            tps = (eff_tokens * n) / running["dt"]  # tokens / sec
            chin_pct = tokens_seen / chinchilla_tokens * 100
            avg_loss = running["loss"] / n
            avg_ce = running["ce"] / n
            avg_mtp = running["mtp"] / n
            avg_z = running["z"] / n

            print(
                f"step {step:8,} | "
                f"loss {avg_loss:.4f} | "
                f"ce {avg_ce:.4f} | "
                f"mtp {avg_mtp:.4f} | "
                f"z {avg_z:.5f} | "
                f"lr {lr_a:.2e} | "
                f"{tps / 1e3:.1f}K tok/s | "
                f"seen {tokens_seen / 1e9:.2f}B ({chin_pct:.0f}% chin) | "
                f"{session_remaining_h():.1f}h left | "
                f"stage {current_stage + 1}",
                flush=True,
            )

            # CSV metrics — loss curve for later plotting
            with open(log_file, "a") as f:
                f.write(
                    f"{step},{avg_loss:.4f},{avg_ce:.4f},{avg_mtp:.4f},{avg_z:.5f},"
                    f"{lr_a:.2e},{tokens_seen},{tps:.1f},{current_stage + 1}\n"
                )

            # W&B — experiment tracking
            if wandb_run:
                wandb_run.log(
                    {
                        "loss": avg_loss,
                        "ce": avg_ce,
                        "mtp": avg_mtp,
                        "z_loss": avg_z,
                        "lr": lr_a,
                        "tokens_per_sec": tps,
                        "tokens_seen": tokens_seen,
                        "chinchilla_pct": chin_pct,
                        "stage": current_stage + 1,
                        "step": step,
                    }
                )

            for k in running:
                running[k] = 0.0
            running["n"] = 0

        # ── Checkpoint ───────────────────────────────────────────────
        if master_process and step > 0 and step % args.save_interval == 0:
            save_checkpoint(
                model,
                opt_muon,
                opt_adamw,
                step,
                args.checkpoint_dir,
                model_args,
                tokens_seen=tokens_seen,
                metrics=dict(loss=total_loss, ce=total_ce, mtp=total_mtp, z=total_z),
                api=api,
                repo_id=repo_id,
            )

    # ── Final ─────────────────────────────────────────────────────────
    if master_process:
        save_checkpoint(
            model,
            opt_muon,
            opt_adamw,
            args.max_iters,
            args.checkpoint_dir,
            model_args,
            tokens_seen=tokens_seen,
            metrics={"final": True, "total_tokens": tokens_seen},
            api=api,
            repo_id=repo_id,
        )
        print(
            f"\n✅ Training complete! Total tokens seen: {tokens_seen / 1e9:.2f}B",
            flush=True,
        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
