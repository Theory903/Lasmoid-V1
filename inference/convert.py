"""
convert.py - LasmoidV1 Checkpoint Converter
============================================
Converts LasmoidV1 checkpoints between formats:
  - PyTorch (.pt) <-> SafeTensors (.safetensors)
  - Model-parallel sharding / unsharding
  - HuggingFace-style state dict -> LasmoidV1 native key format

Compatible with DeepSeek-V3 convert.py design.

Usage:
    # Convert .pt checkpoint to safetensors:
    python convert.py --mode pt2sf --input checkpoint.pt --output ./safetensors_dir

    # Shard a checkpoint for model parallelism:
    python convert.py --mode shard --input ./safetensors_dir --output ./sharded_dir --mp 2

    # Merge sharded checkpoints back:
    python convert.py --mode merge --input ./sharded_dir --output ./merged_dir --mp 2

    # Convert HuggingFace-style state dict:
    python convert.py --mode hf2native --input ./hf_dir --output ./native_dir --n-experts 16

Example (100M training checkpoint -> inference):
    python convert.py --mode pt2sf --input ../checkpoints/lasmoid_100M_step1000.pt \\
                      --output ../checkpoints/lasmoid_100M_sf/
"""

import os
import json
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file, load_file

# ──────────────────────────────────────────────────────────────────
# Key remapping table: HuggingFace-style -> LasmoidV1 native names
# Mirrors DeepSeek-V3 mapping convention.
# ──────────────────────────────────────────────────────────────────
HF_TO_NATIVE = {
    "embed_tokens":               ("emb",           0),       # vocab embedding
    "input_layernorm":            ("attn_norm",      None),
    "post_attention_layernorm":   ("ffn_norm",       None),
    "q_proj":                     ("wq",             0),
    "q_a_proj":                   ("wq_a",           None),
    "q_a_layernorm":              ("q_norm",         None),
    "q_b_proj":                   ("wq_b",           0),
    "kv_a_proj_with_mqa":         ("wkv_a",          None),
    "kv_a_layernorm":             ("kv_norm",        None),
    "kv_b_proj":                  ("wkv_b",          0),
    "o_proj":                     ("wo",             1),
    "gate":                       ("gate",           None),   # MoE router
    "gate_proj":                  ("w_gate",         0),      # ConceptExpert
    "down_proj":                  ("w_down",         1),
    "up_proj":                    ("w_up",           0),
    "norm":                       ("decoder_norm",   None),
    "lm_head":                    ("head",           0),
    "scale":                      ("scale",          None),   # FP8 scale inv
}


def remap_hf_keys(state_dict: dict, n_experts: int, mp_rank: int, mp: int) -> dict:
    """
    Remap a HuggingFace-style state dict to LasmoidV1 native key format.
    Handles model-parallel sharding of expert and attention tensors.
    """
    n_local_experts = n_experts // mp
    new_state = {}

    for name, param in state_dict.items():
        # Strip leading "model." prefix if present
        if name.startswith("model."):
            name = name[len("model."):]

        # Rename sub-module paths
        name = name.replace("self_attn", "attn")
        name = name.replace("mlp", "ffn")
        name = name.replace("weight_scale_inv", "scale")

        key = name.split(".")[-2]
        if key not in HF_TO_NATIVE:
            # Pass-through (e.g. HCM buffers, mHC parameters)
            new_state[name] = param
            continue

        new_key, dim = HF_TO_NATIVE[key]
        name = name.replace(key, new_key)

        # Expert sharding: only keep experts for this MP rank
        if "routed" in name and "shared" not in name:
            expert_idx = int(name.split(".")[-3])
            if expert_idx < mp_rank * n_local_experts or expert_idx >= (mp_rank + 1) * n_local_experts:
                continue  # Not this rank's expert

        # Tensor-parallel sharding of attention / head projections
        elif dim is not None and mp > 1:
            assert param.size(dim) % mp == 0, (
                f"Dimension {dim} of tensor '{name}' (size {param.size(dim)}) "
                f"is not divisible by mp={mp}."
            )
            shard = param.size(dim) // mp
            param = param.narrow(dim, mp_rank * shard, shard).contiguous()

        new_state[name] = param

    return new_state


# ──────────────────────────────────────────────────────────────────
# Mode: pt2sf  —  PyTorch .pt checkpoint -> SafeTensors
# ──────────────────────────────────────────────────────────────────
def pt_to_safetensors(input_path: str, output_dir: str):
    print(f"[convert] Loading PyTorch checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)

    # Support both raw state_dict and training wrapper {"model": state_dict, ...}
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        meta = {k: str(v) for k, v in ckpt.items() if k != "model"}
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {}
    else:
        state_dict = ckpt
        meta = {}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, out_path, metadata=meta)
    print(f"[convert] Saved safetensors -> {out_path}")

    # Write index JSON for compatibility
    index = {"metadata": meta, "weight_map": {k: "model.safetensors" for k in state_dict}}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("[convert] Wrote model.safetensors.index.json")


# ──────────────────────────────────────────────────────────────────
# Mode: shard  —  Single safetensors -> model-parallel shards
# ──────────────────────────────────────────────────────────────────
def shard_checkpoint(input_dir: str, output_dir: str, n_experts: int, mp: int):
    print(f"[convert] Sharding checkpoint (mp={mp}, n_experts={n_experts})")
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    loaded_files: dict = {}

    def get_tensor(name):
        fname = weight_map[name]
        if fname not in loaded_files:
            loaded_files[fname] = load_file(os.path.join(input_dir, fname), device="cpu")
        return loaded_files[fname][name]

    os.makedirs(output_dir, exist_ok=True)
    for rank in range(mp):
        print(f"  Rank {rank}/{mp} ...")
        full_state = {name: get_tensor(name) for name in weight_map}
        sharded = remap_hf_keys(full_state, n_experts, rank, mp)
        out_file = os.path.join(output_dir, f"model{rank}-mp{mp}.safetensors")
        save_file(sharded, out_file)
        print(f"  Saved -> {out_file}")

    # Copy tokenizer files if present
    for fpath in glob(os.path.join(input_dir, "*token*")):
        shutil.copyfile(fpath, os.path.join(output_dir, os.path.basename(fpath)))


# ──────────────────────────────────────────────────────────────────
# Mode: merge  —  Model-parallel shards -> single safetensors
# ──────────────────────────────────────────────────────────────────
def merge_shards(input_dir: str, output_dir: str, mp: int):
    print(f"[convert] Merging {mp} shards from {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged: dict = {}

    for rank in range(mp):
        shard_path = os.path.join(input_dir, f"model{rank}-mp{mp}.safetensors")
        print(f"  Loading shard {rank}: {shard_path}")
        shard = load_file(shard_path, device="cpu")
        for k, v in shard.items():
            if k not in merged:
                merged[k] = v
            else:
                # Concatenate along first dim (shared tensors just overwrite)
                try:
                    merged[k] = torch.cat([merged[k], v], dim=0)
                except Exception:
                    pass  # Non-parallelisable tensors: keep first

    out_path = os.path.join(output_dir, "model.safetensors")
    save_file(merged, out_path)
    print(f"[convert] Merged checkpoint saved -> {out_path}")

    index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in merged}}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


# ──────────────────────────────────────────────────────────────────
# Mode: hf2native  —  HuggingFace state dict -> LasmoidV1 native
# ──────────────────────────────────────────────────────────────────
def hf_to_native(input_dir: str, output_dir: str, n_experts: int):
    print(f"[convert] Remapping HuggingFace weights -> LasmoidV1 native format")
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"  Processing {fname}")
        state = load_file(fpath, device="cpu")
        native = remap_hf_keys(state, n_experts, mp_rank=0, mp=1)
        save_file(native, os.path.join(output_dir, fname))

    # Copy index and tokenizer artifacts
    for pattern in ["*.index.json", "*token*"]:
        for src in glob(os.path.join(input_dir, pattern)):
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copyfile(src, dst)
    print(f"[convert] Done. Output in {output_dir}")


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="LasmoidV1 checkpoint conversion utility")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["pt2sf", "shard", "merge", "hf2native"],
        help="Conversion mode"
    )
    parser.add_argument("--input",  type=str, required=True,  help="Input path (file or directory)")
    parser.add_argument("--output", type=str, required=True,  help="Output directory")
    parser.add_argument("--n-experts",  type=int, default=16, help="Total number of routed experts (for sharding)")
    parser.add_argument("--mp",         type=int, default=1,  help="Model parallelism factor")

    args = parser.parse_args()

    if args.mode == "pt2sf":
        pt_to_safetensors(args.input, args.output)
    elif args.mode == "shard":
        assert args.n_experts % args.mp == 0, "n-experts must be divisible by mp"
        shard_checkpoint(args.input, args.output, args.n_experts, args.mp)
    elif args.mode == "merge":
        merge_shards(args.input, args.output, args.mp)
    elif args.mode == "hf2native":
        hf_to_native(args.input, args.output, args.n_experts)
