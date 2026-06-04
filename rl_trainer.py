"""
rl_trainer.py — LasmoidV1 GRPO Reasoning RL Trainer
=====================================================
Phase 2 training: GRPO (Group Relative Policy Optimization)
Implements the DeepSeek-R1-Zero style reasoning RL on top of the SFT checkpoint.

Algorithm (GRPO):
  1. Sample a prompt p from the math/IF dataset
  2. Generate G completions {o_1,...,o_G} from current policy π_θ
  3. Compute reward r_i for each o_i  (format + length + accuracy)
  4. Compute group-relative advantage Â_i = (r_i - μ) / σ
  5. Compute GRPO loss:
       L = -Σ_i Â_i · log π_θ(o_i | p) + β · KL(π_θ || π_ref)
  6. Update θ via AdamW

Compatible with DDP (torchrun --nproc_per_node=2).
Session watchdog exits gracefully and HF Hub auto-syncs.

Usage:
  torchrun --nproc_per_node=2 rl_trainer.py \\
    --model_size 10M --max_iters 500 --group_size 4 \\
    --rl_lr 5e-6 --kl_coef 0.01 --checkpoint_dir checkpoints_rl

References:
  DeepSeek-R1: https://arxiv.org/abs/2501.12948
  GRPO paper:  https://arxiv.org/abs/2402.03300
  RLOO:        https://arxiv.org/abs/2402.14740
"""

import os, sys, time, math, random, argparse, copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
from contextlib import nullcontext
from typing import List, Optional, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference.model import LasmoidV1, ModelArgs
from rewards import batch_rewards, compute_grpo_advantages, compute_rloo_advantages

# Reuse MODEL_CONFIGS and helpers from SFT trainer
from train_kaggle import (
    MODEL_CONFIGS, get_lr_wsd, EMA,
    session_remaining_h, session_expired,
)


# ══════════════════════════════════════════════════════════════════════
# THINKING PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════

THINK_SYSTEM_PROMPT = (
    "You are a helpful assistant. Explore multiple reasoning paths inside <Parallel><Path>...</Path>...</Parallel> tags, "
    "then provide a summary inside <Summary>...</Summary> tags before giving your final answer.\n\n"
)

def format_prompt(question: str) -> str:
    """Wrap a raw question in the thinking prompt format."""
    return THINK_SYSTEM_PROMPT + f"Question: {question}\n\nAnswer:"


# ══════════════════════════════════════════════════════════════════════
# MATH / IF PROMPT LOADER
# ══════════════════════════════════════════════════════════════════════

def stream_rl_prompts(
    tok: tiktoken.Encoding,
    max_seq_len: int,
    hf_token: Optional[str] = None,
):
    """
    Infinite generator of (prompt_tokens, gt_answer) tuples from
    math/IF SFT datasets. Each prompt is already tokenised.

    Dataset priority:
      1. UltraData-Math (openbmb/UltraData-SFT-2605) — has ground truth
      2. UltraData-IF   (openbmb/UltraData-SFT-2605) — format reward only
    """
    from datasets import load_dataset

    def _load(name, config, split, question_field, answer_field=None):
        ds = load_dataset(name, config, split=split, streaming=True, token=hf_token)
        for item in ds:
            q = item.get(question_field, "")
            a = item.get(answer_field) if answer_field else None
            if not q:
                continue
            prompt_str = format_prompt(str(q))
            tokens = tok.encode(prompt_str, allowed_special={"<|endoftext|>"})
            # Truncate if too long (leave room for response)
            tokens = tokens[: max(1, max_seq_len // 2)]
            yield tokens, (str(a) if a else None)

    # Interleave math (70%) and IF (30%)
    import itertools
    math_gen = _load("openbmb/UltraData-SFT-2605", "Math", "no_think",
                     question_field="prompt", answer_field="response")
    if_gen   = _load("openbmb/UltraData-SFT-2605", "IF",   "no_think",
                     question_field="prompt", answer_field=None)

    math_iter = itertools.cycle(math_gen)
    if_iter   = itertools.cycle(if_gen)

    weights = [0.70, 0.30]
    sources = [math_iter, if_iter]
    while True:
        src = random.choices(sources, weights=weights, k=1)[0]
        yield next(src)


# ══════════════════════════════════════════════════════════════════════
# GRPO LOSS
# ══════════════════════════════════════════════════════════════════════

def grpo_loss(
    policy_logps: torch.Tensor,   # (G,) log-probs of each completion
    advantages:   torch.Tensor,   # (G,) normalised group advantages
    ref_logps:    torch.Tensor,   # (G,) reference model log-probs
    kl_coef:      float = 0.01,
    clip_eps:     float = 0.2,    # PPO-style clip (optional — set 0 to disable)
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO objective with optional PPO-style clipping and KL penalty.

    L = -E[clip(r, 1-ε, 1+ε) · Â] + kl_coef · KL(π || π_ref)

    Where r = exp(log π - log π_ref) is the importance ratio.

    Returns (loss_scalar, metrics_dict).
    """
    # Importance ratio
    log_ratio  = policy_logps - ref_logps.detach()
    ratio      = torch.exp(log_ratio)

    # Policy gradient term (with optional PPO clip)
    pg_unclipped = ratio * advantages
    if clip_eps > 0:
        ratio_clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        pg_clipped    = ratio_clipped * advantages
        pg_loss       = -torch.min(pg_unclipped, pg_clipped).mean()
    else:
        pg_loss = -pg_unclipped.mean()

    # KL penalty: KL(π || π_ref) ≈ ratio - 1 - log_ratio  (unbiased estimator)
    kl_penalty = (ratio - 1 - log_ratio).mean()
    kl_penalty = kl_penalty.clamp(min=0)  # KL ≥ 0

    loss = pg_loss + kl_coef * kl_penalty

    metrics = {
        "pg_loss":    pg_loss.item(),
        "kl_penalty": kl_penalty.item(),
        "mean_ratio": ratio.mean().item(),
    }
    return loss, metrics


# ══════════════════════════════════════════════════════════════════════
# SEQUENCE LOG-PROB COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def sequence_logp(
    model: LasmoidV1,
    input_ids: torch.Tensor,   # (1, T)
    response_start: int,        # index where response begins in input_ids
) -> torch.Tensor:
    """
    Compute the sum of log-probabilities of tokens from response_start onward.
    This is log π(response | prompt).

    Uses teacher-forcing (full sequence forward pass).
    """
    with torch.no_grad() if not model.training else nullcontext():
        logits, _, _, _ = model(input_ids, input_ids)  # (1, T, vocab)

    # Shift: predict token t+1 from hidden state at t
    logits_shifted  = logits[:, response_start - 1 : -1, :]  # (1, L, vocab)
    targets         = input_ids[:, response_start:]            # (1, L)
    L               = targets.shape[1]

    log_probs = F.log_softmax(logits_shifted.float(), dim=-1)  # (1, L, vocab)
    token_lps = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)  # (1, L)

    # Sum log-probs → sequence log-prob
    return token_lps.sum(dim=-1).squeeze(0)  # scalar


# ══════════════════════════════════════════════════════════════════════
# ROLLOUT ENGINE
# ══════════════════════════════════════════════════════════════════════

def rollout(
    model: LasmoidV1,
    prompt_tokens: List[int],
    tok: tiktoken.Encoding,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> Tuple[str, List[int]]:
    """
    Generate one completion from the model given prompt_tokens.
    Returns (decoded_string, full_token_ids_including_prompt).
    """
    model.eval()
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    max_len = model.args.max_seq_len

    # Pad prompt to max_len for prefill
    cond_len = idx.shape[1]
    if cond_len < max_len:
        pad = torch.full(
            (1, max_len - cond_len),
            tok.eot_token, dtype=torch.long, device=device
        )
        idx_padded = torch.cat([pad, idx], dim=1)
    else:
        idx_padded = idx[:, -max_len:]

    with torch.no_grad():
        logits, _, concept_db, memory_state = model(idx_padded, idx_padded, start_pos=0)

    generated = list(prompt_tokens)
    last_tok   = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
    current_pos = max_len

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _, _, _ = model(
                x_enc=None, x_dec=last_tok,
                concept_db=concept_db, memory_state=memory_state,
                start_pos=current_pos,
            )
        logits_last = logits[:, -1, :] / max(temperature, 1e-8)
        probs = torch.softmax(logits_last, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tid = next_tok.item()
        generated.append(tid)
        if tid == tok.eot_token:
            break
        last_tok    = next_tok
        current_pos += 1
        if current_pos - max_len >= max_new_tokens:
            break

    model.train()
    # Decode only the response portion
    response_ids = generated[len(prompt_tokens):]
    response_str = tok.decode(response_ids)
    return response_str, generated


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_sft_checkpoint(
    model: LasmoidV1,
    opt: torch.optim.Optimizer,
    api,
    repo_id: str,
    hf_token: str,
    device: str,
    rl_checkpoint_dir: str,
) -> int:
    """
    Load starting point for RL training.
    Priority:
      1. Latest RL checkpoint in rl_checkpoint_dir
      2. Latest SFT checkpoint from HF Hub (lasmoid-10m)
    Returns start_step.
    """
    import glob

    # 1. Check local RL checkpoints
    rl_ckpts = sorted(glob.glob(os.path.join(rl_checkpoint_dir, "rl_step_*.pt")))
    if rl_ckpts:
        ckpt = torch.load(rl_ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if opt is not None:
            opt.load_state_dict(ckpt["opt_state_dict"])
        print(f"  [rl resume] loaded {rl_ckpts[-1]}, step={ckpt['step']}")
        return ckpt["step"]

    # 2. Load SFT checkpoint from Hub as starting point
    if api and repo_id:
        try:
            from huggingface_hub import hf_hub_download
            files = list(api.list_repo_files(repo_id=repo_id))
            sft_ckpts = sorted([f for f in files if f.endswith(".pt")])
            if sft_ckpts:
                latest = sft_ckpts[-1]
                print(f"  [rl init] downloading SFT checkpoint {latest}...")
                path = hf_hub_download(repo_id, latest, token=hf_token,
                                       local_dir="/tmp/lasmoid_sft")
                ckpt = torch.load(path, map_location=device, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
                print(f"  [rl init] SFT checkpoint loaded ✓  (starting RL from step 0)")
        except Exception as e:
            print(f"  [rl init] Hub load failed: {e} — starting from random weights")
    return 0


def save_rl_checkpoint(model, opt, step, rl_checkpoint_dir, api, repo_id, master_process):
    if not master_process:
        return
    os.makedirs(rl_checkpoint_dir, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    path = os.path.join(rl_checkpoint_dir, f"rl_step_{step:08d}.pt")
    torch.save({
        "step": step,
        "model_state_dict": raw.state_dict(),
        "opt_state_dict":   opt.state_dict(),
    }, path)
    print(f"  [rl save] {path}")
    if api and repo_id:
        try:
            api.upload_file(path_or_fileobj=path,
                           path_in_repo=os.path.basename(path),
                           repo_id=repo_id)
            print(f"  [rl hub]  uploaded to {repo_id}")
        except Exception as e:
            print(f"  [rl hub]  upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# MAIN GRPO TRAINER
# ══════════════════════════════════════════════════════════════════════

def train_rl(args):
    # ── DDP setup ─────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        ddp_rank       = dist.get_rank()
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device         = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master_process = True

    # ── Model ─────────────────────────────────────────────────────────
    cfg = MODEL_CONFIGS[args.model_size].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"]    = args.max_seq_len
    cfg["max_batch_size"] = 1  # rollout one at a time
    model_args_obj = ModelArgs(**cfg)

    policy_model = LasmoidV1(model_args_obj).to(device)

    # ── HF Hub ────────────────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN")
    api = repo_id = None
    if hf_token and master_process:
        try:
            from huggingface_hub import HfApi, login
            login(token=hf_token, add_to_git_credential=False)
            api = HfApi(token=hf_token)
            username = api.whoami()["name"]
            repo_id  = f"{username}/lasmoid-{args.model_size.lower()}-rl"
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            if master_process:
                print(f"\n  HF Hub (RL): {repo_id}")
        except Exception as e:
            print(f"  HF Hub init failed: {e}")
            api = repo_id = None

    # ── Tokenizer ─────────────────────────────────────────────────────
    tok = tiktoken.get_encoding("gpt2")

    # ── Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.rl_lr, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8,
    )

    # ── Load checkpoint ───────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_step = load_sft_checkpoint(
        policy_model, opt, api,
        repo_id=f"{api.whoami()['name']}/lasmoid-{args.model_size.lower()}" if api else None,
        hf_token=hf_token,
        device=device,
        rl_checkpoint_dir=args.checkpoint_dir,
    )

    # ── Frozen reference model (copy of policy at RL start) ───────────
    # Memory cost: same as policy (~40MB for 10M), trivial on T4
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    if master_process:
        print(f"  Reference model frozen ✓  ({sum(p.numel() for p in ref_model.parameters())/1e6:.1f}M params)")

    # ── DDP wrap ──────────────────────────────────────────────────────
    if ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # ── Data stream ───────────────────────────────────────────────────
    prompt_stream = stream_rl_prompts(tok, args.max_seq_len, hf_token)

    if master_process:
        print(f"\n{'═'*70}")
        print(f"  LasmoidV1 GRPO Reasoning RL Trainer")
        print(f"  Model: {args.model_size} | Group size G={args.group_size}")
        print(f"  Method: {'RLOO' if args.use_rloo else 'GRPO'}")
        print(f"  Rewards: format(1.0) + length(0.5) + accuracy(2.0)")
        print(f"  KL coef: {args.kl_coef}  |  clip_eps: {args.clip_eps}")
        print(f"  LR: {args.rl_lr}  |  Max iters: {args.max_iters}")
        print(f"{'═'*70}\n")

    # ── Training loop ─────────────────────────────────────────────────
    policy_model.train()
    log_file = os.path.join(args.checkpoint_dir, "rl_metrics.csv")
    if master_process and start_step == 0:
        with open(log_file, "w") as f:
            f.write("step,loss,pg_loss,kl,mean_reward,format_rate,accuracy_rate,lr\n")

    running = dict(loss=0.0, pg=0.0, kl=0.0, reward=0.0, fmt=0.0, acc=0.0, n=0)

    for step in range(start_step, args.max_iters):
        if session_expired() and master_process:
            print(f"\n⏰ Session limit. Saving RL checkpoint at step {step}.")
            save_rl_checkpoint(policy_model, opt, step, args.checkpoint_dir, api, repo_id, master_process)
            break

        # ── Sample a prompt ───────────────────────────────────────────
        prompt_tokens, gt_answer = next(prompt_stream)

        # ── Generate G completions (rollouts) ─────────────────────────
        responses_str  = []
        responses_ids  = []  # full token lists including prompt
        raw = policy_model.module if ddp else policy_model

        for _ in range(args.group_size):
            resp_str, full_ids = rollout(
                raw, prompt_tokens, tok,
                max_new_tokens=args.max_new_tokens,
                temperature=args.rollout_temp,
                device=device,
            )
            responses_str.append(resp_str)
            responses_ids.append(full_ids)

        # ── Compute rewards ───────────────────────────────────────────
        gt_list  = [gt_answer] * args.group_size
        rewards, breakdowns = batch_rewards(
            responses_str, gt_list,
            w_format=1.0, w_length=0.5, w_accuracy=2.0,
        )

        # ── Compute advantages ────────────────────────────────────────
        if args.use_rloo:
            advantages = compute_rloo_advantages(rewards)
        else:
            advantages = compute_grpo_advantages(rewards)

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # ── Compute policy and reference log-probs ────────────────────
        policy_logps = []
        ref_logps    = []
        response_start = len(prompt_tokens)

        for full_ids in responses_ids:
            full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            # Policy log-prob (with grad)
            policy_model.train()
            plp = sequence_logp(raw, full_t, response_start)
            policy_logps.append(plp)
            # Reference log-prob (no grad)
            with torch.no_grad():
                rlp = sequence_logp(ref_model, full_t, response_start)
            ref_logps.append(rlp)

        policy_logps_t = torch.stack(policy_logps)      # (G,)
        ref_logps_t    = torch.stack(ref_logps)          # (G,)

        # ── GRPO loss + backward ──────────────────────────────────────
        lr = get_lr_wsd(step, args.max_iters, args.warmup_steps, args.rl_lr,
                        decay_frac=0.3, lr_min_ratio=0.1)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss, metrics = grpo_loss(
            policy_logps_t, adv_t, ref_logps_t,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        opt.step()

        # ── Accumulate metrics ────────────────────────────────────────
        mean_r   = sum(rewards) / len(rewards)
        fmt_rate = sum(1 for b in breakdowns if b["format"] >= 1.0) / len(breakdowns)
        acc_rate = sum(1 for b in breakdowns if b["accuracy"] > 0)  / len(breakdowns)
        running["loss"]   += loss.item()
        running["pg"]     += metrics["pg_loss"]
        running["kl"]     += metrics["kl_penalty"]
        running["reward"] += mean_r
        running["fmt"]    += fmt_rate
        running["acc"]    += acc_rate
        running["n"]      += 1

        # ── Log ───────────────────────────────────────────────────────
        if master_process and step % args.log_interval == 0 and running["n"] > 0:
            n = running["n"]
            print(
                f"rl_step {step:6,} | "
                f"loss {running['loss']/n:.4f} | "
                f"pg {running['pg']/n:.4f} | "
                f"kl {running['kl']/n:.4f} | "
                f"reward {running['reward']/n:.3f} | "
                f"fmt {running['fmt']/n*100:.0f}% | "
                f"acc {running['acc']/n*100:.0f}% | "
                f"lr {lr:.2e} | "
                f"{session_remaining_h():.1f}h left",
                flush=True,
            )
            with open(log_file, "a") as f:
                f.write(
                    f"{step},{running['loss']/n:.4f},{running['pg']/n:.4f},"
                    f"{running['kl']/n:.4f},{running['reward']/n:.4f},"
                    f"{running['fmt']/n:.4f},{running['acc']/n:.4f},{lr:.2e}\n"
                )
            running = dict(loss=0.0, pg=0.0, kl=0.0, reward=0.0, fmt=0.0, acc=0.0, n=0)

        # ── Save checkpoint ───────────────────────────────────────────
        if master_process and step > 0 and step % args.save_interval == 0:
            save_rl_checkpoint(policy_model, opt, step, args.checkpoint_dir, api, repo_id, master_process)

        # ── Optionally refresh reference model ────────────────────────
        if args.ref_update_interval > 0 and step % args.ref_update_interval == 0 and step > 0:
            raw_params = (policy_model.module if ddp else policy_model).state_dict()
            ref_model.load_state_dict(raw_params)
            ref_model.eval()
            if master_process:
                print(f"  [ref] Updated reference model at step {step}")

    # ── Final save ────────────────────────────────────────────────────
    save_rl_checkpoint(policy_model, opt, args.max_iters, args.checkpoint_dir, api, repo_id, master_process)
    if ddp:
        dist.destroy_process_group()
    if master_process:
        print("\n✅ GRPO training complete.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LasmoidV1 GRPO Reasoning RL Trainer")
    p.add_argument("--model_size",    type=str,   default="10M")
    p.add_argument("--max_seq_len",   type=int,   default=256)
    p.add_argument("--max_iters",     type=int,   default=500,
                   help="Number of GRPO update steps")
    p.add_argument("--group_size",    type=int,   default=4,
                   help="G: rollouts per prompt (GRPO group size)")
    p.add_argument("--max_new_tokens",type=int,   default=128,
                   help="Max tokens generated per rollout")
    p.add_argument("--rollout_temp",  type=float, default=0.9,
                   help="Temperature for rollout sampling (> 0 for diversity)")
    p.add_argument("--rl_lr",         type=float, default=5e-6,
                   help="Learning rate for RL (should be 10-100× smaller than SFT LR)")
    p.add_argument("--warmup_steps",  type=int,   default=20)
    p.add_argument("--kl_coef",       type=float, default=0.01,
                   help="KL penalty coefficient (prevents reward hacking)")
    p.add_argument("--clip_eps",      type=float, default=0.2,
                   help="PPO-style importance ratio clip (0 = disable)")
    p.add_argument("--use_rloo",      action="store_true",
                   help="Use RLOO instead of GRPO advantage estimation")
    p.add_argument("--ref_update_interval", type=int, default=0,
                   help="Steps between reference model updates (0 = never)")
    p.add_argument("--checkpoint_dir",type=str,   default="checkpoints_rl")
    p.add_argument("--log_interval",  type=int,   default=10)
    p.add_argument("--save_interval", type=int,   default=100)
    p.add_argument("--session_hours", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_rl(args)
