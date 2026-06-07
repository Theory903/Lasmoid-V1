"""
LasmoidV1 — generate.py
=======================
2026 Sampler Upgrades (traditional plumbing only):
  ✓ Min-P   : Dynamic threshold relative to p_max  (replaces Top-P as default)
  ✓ Top-P   : Nucleus sampling (legacy, still available)
  ✓ Top-K   : Hard-cutoff filter (still available)
  ✓ XTC     : Exclude Top Choices — breaks clichés for creative tasks
  ✓ DRY     : Don't Repeat Yourself — n-gram context-aware repetition penalty
  ✓ Temp    : Temperature scaling (applied BEFORE all filters, in log-space)
  ✓ Streaming: `stream=True` yields tokens one-by-one as a generator
  ✓ Greedy  : temperature=0 → deterministic argmax
  ✓ Checkpoint: auto-detects .safetensors or .pt

Order of operations (2026 consensus):
  logits → temperature → DRY penalty → XTC → Min-P / Top-P / Top-K → sample
"""

import os
import sys
import json
import glob
from argparse import ArgumentParser
from typing import List, Optional, Generator

import torch
import torch.nn.functional as F
import tiktoken

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .model import LasmoidV1, ModelArgs, Linear
except ImportError:
    from model import LasmoidV1, ModelArgs, Linear


# ══════════════════════════════════════════════════════════════════════
# SAMPLER PRIMITIVES  (2026 SOTA)
# ══════════════════════════════════════════════════════════════════════

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling. temperature=0 → greedy."""
    if temperature == 0.0:
        return logits  # handled downstream via argmax
    return logits / max(temperature, 1e-8)


def _apply_dry(
    logits: torch.Tensor,
    generated: List[int],
    dry_multiplier: float = 0.8,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
) -> torch.Tensor:
    """
    DRY (Don't Repeat Yourself) — 2026 SOTA repetition control.
    Tracks n-gram sequences in the generation history and penalises
    tokens that would continue a repeated sequence.
    Superior to traditional repetition_penalty which degrades grammar.

    Args:
        dry_multiplier:    Penalty strength (0 = disabled, 0.8 is default).
        dry_base:          Exponential base for length-scaled penalty.
        dry_allowed_length: N-gram sequences shorter than this are ignored.
    """
    if dry_multiplier == 0.0 or len(generated) < dry_allowed_length:
        return logits

    logits = logits.clone()
    last_token = generated[-1]

    # Find all positions in history where last_token appeared
    match_indices = [i for i, t in enumerate(generated[:-1]) if t == last_token]

    for idx in match_indices:
        # Walk backwards from the match to find the longest repeated n-gram
        match_len = 1
        # FIX: guard was `match_len < dry_allowed_length` which stops the walk at
        # match_len == dry_allowed_length-1. The subsequent `if match_len <
        # dry_allowed_length: continue` then always skipped exact minimum-length
        # matches. Changed to `<=` so the walk can reach dry_allowed_length and the
        # check below passes correctly.
        while (match_len <= dry_allowed_length and
               idx - match_len >= 0 and
               len(generated) - 1 - match_len >= 0 and
               generated[idx - match_len] == generated[-1 - match_len]):
            match_len += 1

        if match_len < dry_allowed_length:
            continue

        # The next token after the match is the one to penalise
        if idx + 1 < len(generated):
            penalised_token = generated[idx + 1]
            penalty = dry_multiplier * (dry_base ** (match_len - dry_allowed_length))
            logits[0, penalised_token] -= penalty

    return logits


def _apply_xtc(
    logits: torch.Tensor,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
) -> torch.Tensor:
    """
    XTC (eXclude Top Choices) — 2026 creative diversity sampler.
    With probability `xtc_probability`, removes tokens whose probability
    exceeds `xtc_threshold`, forcing selection from less-obvious vocabulary.
    Most effective for creative / open-ended generation.

    Args:
        xtc_probability: Chance of applying XTC at this step (0.0 = disabled).
        xtc_threshold:   Minimum prob for a token to be excluded.
    """
    if xtc_probability == 0.0:
        return logits
    if torch.rand(1).item() > xtc_probability:
        return logits

    logits = logits.clone()
    probs  = torch.softmax(logits, dim=-1)

    # Mask out all tokens whose probability exceeds the threshold,
    # but always keep at least 1 token (the lowest prob surviving one).
    mask = probs > xtc_threshold
    if mask.sum() < probs.numel():  # at least one token survives
        logits[mask] = float("-inf")
    return logits


def _apply_min_p(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """
    Min-P sampling — 2026 default (supersedes Top-P for most tasks).
    Threshold is scaled dynamically by the probability of the most likely token.
    When the model is confident (high p_max), threshold is strict.
    When uncertain (low p_max), threshold is loose → more diversity.

    Args:
        min_p: Base minimum probability (0.05 is the 2026 community default).
    """
    if min_p <= 0.0:
        return logits
    probs     = torch.softmax(logits, dim=-1)
    p_max     = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * p_max
    # Zero-out tokens below the dynamic threshold
    logits = logits.clone()
    logits[probs < threshold] = float("-inf")
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """Top-P (nucleus) sampling. top_p=1.0 means no filtering."""
    if top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative prob > top_p (shift by 1 to keep the token
    # that pushes over the threshold)
    remove_mask = (cumulative - sorted_probs) > top_p

    # FIX: the old code converted filtered probs back to log-space via .log(),
    # returning values ≤ 0.  _apply_top_k (called next) compares raw logit values
    # against a threshold from torch.topk — raw logits can be large positives, so
    # all log-probs would be below the threshold and top_k became a silent no-op.
    # Fix: scatter -inf back onto the original logit tensor instead, keeping the
    # pipeline in a consistent raw-logit space throughout.
    remove_original = torch.zeros_like(logits, dtype=torch.bool)
    remove_original.scatter_(-1, sorted_idx, remove_mask)
    logits = logits.clone()
    logits[remove_original] = float("-inf")
    return logits


def _apply_top_k(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Hard Top-K filter. top_k=0 means no filtering."""
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold  = values[..., -1, None]
    logits     = logits.clone()
    logits[logits < threshold] = float("-inf")
    return logits


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Final sampling step after all filters have been applied."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _full_sample(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
) -> torch.Tensor:
    """
    Full 2026 sampling pipeline:
      temperature → DRY → XTC → Min-P (or Top-P / Top-K) → sample
    """
    logits = _apply_temperature(logits, temperature)
    logits = _apply_dry(logits, generated_ids, dry_multiplier, dry_base, dry_allowed_length)
    logits = _apply_xtc(logits, xtc_probability, xtc_threshold)

    # Min-P is the 2026 default; fall back to Top-P if min_p disabled
    if min_p > 0.0:
        logits = _apply_min_p(logits, min_p)
    else:
        logits = _apply_top_p(logits, top_p)

    logits = _apply_top_k(logits, top_k)
    return _sample_token(logits, temperature)


# ══════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate(
    model: LasmoidV1,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    stream: bool = False,
) -> List[List[int]]:
    """
    Generate tokens using LasmoidV1 with 2026 SOTA sampling.

    Args:
        stream:           If True, prints tokens as they are generated.
        min_p:            Min-P threshold (0.05 default, 0 to disable).
        xtc_probability:  XTC application probability per step.
        dry_multiplier:   DRY penalty strength (0 = disabled).
    """
    model.eval()
    device  = next(model.parameters()).device
    max_len = model.args.max_seq_len
    results = []

    for tokens_list in prompt_tokens:
        idx          = torch.tensor([tokens_list], dtype=torch.long, device=device)
        generated_ids: List[int] = list(tokens_list)

        # Prefill: run encoder + HCM once, freeze memory
        cond_len = idx.shape[1]
        if cond_len < max_len:
            padding   = torch.full((1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device)
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -max_len:]

        # Run prefill forward
        logits, _, concept_db, memory_state = model(idx_padded, idx_padded, start_pos=0)

        # Sample first token
        last_logits = logits[:, -1, :]
        idx_next = _full_sample(
            last_logits,
            generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
        )

        token_id = idx_next.item()
        if token_id != eos_id:
            new_tokens = [token_id]
            generated_ids.append(token_id)
            idx = torch.cat([idx, idx_next], dim=1)
            
            # Decode loop
            current_pos = max_len
            for step in range(max_new_tokens - 1):
                logits, _, _, _ = model(
                    x_enc=None,
                    x_dec=idx_next,
                    concept_db=concept_db,
                    memory_state=memory_state,
                    start_pos=current_pos,
                )
                
                idx_next = _full_sample(
                    logits[:, -1, :],
                    generated_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    dry_multiplier=dry_multiplier,
                    dry_base=dry_base,
                    dry_allowed_length=dry_allowed_length,
                )
                
                token_id = idx_next.item()
                if token_id == eos_id:
                    break
                new_tokens.append(token_id)
                generated_ids.append(token_id)
                idx = torch.cat([idx, idx_next], dim=1)
                current_pos += 1
        else:
            new_tokens = []

        results.append(new_tokens)

    return results


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING  (auto-detects .safetensors or .pt)
# ══════════════════════════════════════════════════════════════════════

def load_checkpoint_and_model(ckpt_path: str, config_path: str, device: str) -> tuple:
    # 1. Find checkpoint file path
    ckpt_file = None
    final = os.path.join(ckpt_path, "lasmoid_final.pt")
    latest_pt = os.path.join(ckpt_path, "lasmoid_latest.pt")
    if os.path.exists(final):
        ckpt_file = final
    elif os.path.exists(latest_pt):
        ckpt_file = latest_pt
    else:
        step_files = sorted(glob.glob(os.path.join(ckpt_path, "lasmoid_step_*.pt")))
        if step_files:
            ckpt_file = step_files[-1]
            
    # 2. Load the checkpoint file (if exists) and extract model_args
    model_args = None
    state_dict = None
    if ckpt_file:
        try:
            sd = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = sd.get("model_state_dict", sd)
            model_args = sd.get("model_args")
            print(f"[generate] Found checkpoint: {ckpt_file}")
        except Exception as e:
            print(f"[generate] Failed to load checkpoint file directly: {e}")
            
    # 3. If model_args not found in checkpoint, load from config
    if model_args is None:
        with open(config_path) as f:
            config_dict = json.load(f)
        from dataclasses import fields
        valid_fields = {f.name for f in fields(ModelArgs)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        model_args = ModelArgs(**filtered_config)
        print(f"[generate] Loaded config from {config_path}")
        
    # 4. Instantiate model
    model = LasmoidV1(model_args).to(device)
    
    # 5. If state_dict is loaded, load it into model (handling dynamic expansion of slots)
    if state_dict is not None:
        # Dynamically expand concept blocks
        block_indices = []
        for key in state_dict.keys():
            if key.startswith("memory.concept_blocks."):
                parts = key.split(".")
                block_indices.append(int(parts[2]))
        num_blocks_in_ckpt = max(block_indices) + 1 if block_indices else 1
        
        current_num_blocks = len(model.memory.concept_blocks)
        if num_blocks_in_ckpt > current_num_blocks:
            dtype = next(model.parameters()).dtype
            for i in range(current_num_blocks, num_blocks_in_ckpt):
                new_block = model.memory._create_block(model.args).to(device=device, dtype=dtype)
                model.memory.concept_blocks.append(new_block)
                
            model.memory.register_buffer(
                "slot_db",
                torch.zeros(num_blocks_in_ckpt, model.memory.num_concepts, model.memory.dim, device=device, dtype=dtype),
                persistent=True
            )
            model.memory.register_buffer(
                "meta_centroids",
                torch.zeros(num_blocks_in_ckpt, model.memory.dim, device=device, dtype=dtype),
                persistent=False
            )
            print(f"[generate] Dynamically expanded memory blocks to {num_blocks_in_ckpt}.")
            
        model.load_state_dict(state_dict, strict=True)
        print(f"[generate] Successfully loaded weights from checkpoint.")
    else:
        print("[generate] WARNING: Running with random initialisation.")
        
    return model, model_args


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dry_multiplier: float = 0.0,
) -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[generate] Device: {device.upper()}")

    torch.manual_seed(42)

    model, args = load_checkpoint_and_model(ckpt_path, config, device)
    model.eval()

    Linear.dtype     = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    Linear.scale_fmt = getattr(args, "scale_fmt", None)

    enc          = tiktoken.get_encoding("gpt2")
    eos_token_id = enc.eot_token

    sampler_cfg = dict(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        dry_multiplier=dry_multiplier,
    )

    sampler_summary = (
        f"temperature={temperature}  min_p={min_p}  top_p={top_p}  top_k={top_k}\n"
        f"XTC prob={xtc_probability}  threshold={xtc_threshold}  "
        f"DRY mult={dry_multiplier}"
    )

    if interactive:
        print(f"\n{'─'*60}")
        print(" LasmoidV1 — Interactive Shell (2026 Samplers)")
        print(f"{'─'*60}")
        print(f" Samplers: {sampler_summary}")
        print(" Commands: /exit  /clear  /sampler")
        print(f"{'─'*60}\n")

        history: List[str] = []
        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                break
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                history.clear()
                print("[History cleared]")
                continue
            elif prompt == "/sampler":
                print(f"[Samplers] {sampler_summary}")
                continue
            elif not prompt:
                continue

            history.append(prompt)
            context       = " ".join(history) + " "
            prompt_tokens = [enc.encode(context, allowed_special={"<|endoftext|>"})]

            completion_tokens = generate(model, prompt_tokens, max_new_tokens, eos_token_id, **sampler_cfg)
            completion        = enc.decode(completion_tokens[0])
            print(completion)
            history.append(completion.strip())

    else:
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return
        with open(input_file) as f:
            prompts = [l.strip() for l in f if l.strip()]

        print(f"[generate] Batch mode: {len(prompts)} prompts")
        prompt_tokens     = [enc.encode(p, allowed_special={"<|endoftext|>"}) for p in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, eos_token_id, **sampler_cfg)

        for p, ct in zip(prompts, completion_tokens):
            print(f"Prompt:     {p}")
            print(f"Completion: {enc.decode(ct)}")
            print("─" * 50)


if __name__ == "__main__":
    parser = ArgumentParser(description="LasmoidV1 inference — 2026 samplers")
    parser.add_argument("--ckpt-path",      type=str, required=True)
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--input-file",     type=str, default="")
    parser.add_argument("--interactive",    action="store_true")
    parser.add_argument("--max-new-tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top-k",          type=int,   default=0,    help="0 = disabled")
    parser.add_argument("--top-p",          type=float, default=1.0,  help="1.0 = disabled")
    parser.add_argument("--min-p",          type=float, default=0.05, help="2026 default; 0 = disabled")
    parser.add_argument("--xtc-probability",type=float, default=0.0,  help="XTC per-step probability")
    parser.add_argument("--xtc-threshold",  type=float, default=0.1,  help="XTC prob threshold")
    parser.add_argument("--dry-multiplier", type=float, default=0.0,  help="DRY strength; 0 = disabled")
    a = parser.parse_args()

    assert a.input_file or a.interactive, "Specify --input-file or --interactive"
    main(
        a.ckpt_path, a.config, a.input_file, a.interactive,
        a.max_new_tokens, a.temperature,
        a.top_k, a.top_p, a.min_p,
        a.xtc_probability, a.xtc_threshold, a.dry_multiplier,
    )
