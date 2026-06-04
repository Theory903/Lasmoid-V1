"""
fp8_cast_bf16.py - LasmoidV1 FP8 → BF16 Weight Dequantization
===============================================================
Converts a LasmoidV1 checkpoint saved in FP8 (float8_e4m3fn) precision
back to BF16 for inference on hardware that does not support FP8 natively
(e.g. Apple Silicon MPS, older NVIDIA GPUs, CPU).

Works with:
  - SafeTensors checkpoints produced by train.py / convert.py
  - Both "model.safetensors" (single-file) and sharded formats

Usage:
    # Dequantize a single safetensors file:
    python fp8_cast_bf16.py --input-fp8-path ./checkpoints/fp8/ \\
                            --output-bf16-path ./checkpoints/bf16/

    # With custom block size (default 128, matching kernel.py):
    python fp8_cast_bf16.py --input-fp8-path ./fp8 --output-bf16-path ./bf16 \\
                            --block-size 128

Notes:
    - FP8 weights are identified by element_size() == 1 (torch.float8_e4m3fn).
    - Each FP8 weight tensor requires a paired "_scale_inv" tensor to dequantize.
    - The scale tensor format is (out_features // block_size, in_features // block_size).
    - BF16 output tensors are computed as: W_bf16 = W_fp8.to(bf16) * scale_inv
    - Memory is managed with a 2-file LRU cache for large sharded checkpoints.
"""

import os
import json
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import load_file, save_file

try:
    from .kernel import weight_dequant
except ImportError:
    from kernel import weight_dequant


def _make_index(output_dir: str, weight_map: dict):
    """Write a model.safetensors.index.json for the converted checkpoint."""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def main(fp8_path: str, bf16_path: str, block_size: int = 128):
    """
    Convert all FP8-quantized tensors in a LasmoidV1 safetensors checkpoint
    directory to BF16 and write results to bf16_path.

    Args:
        fp8_path:   Directory containing FP8 .safetensors files + index JSON.
        bf16_path:  Destination directory for BF16 .safetensors output.
        block_size: Quantization tile size (must match the value used in kernel.py).
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    # ── Locate & parse the weight index ────────────────────────────────────
    index_path = os.path.join(fp8_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            model_index = json.load(f)
        weight_map: dict = model_index.get("weight_map", {})
    else:
        # Single-file mode: no index, infer from glob
        sf_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {fp8_path}")
        weight_map = {}
        for sf in sf_files:
            fname = os.path.basename(sf)
            tmp = load_file(sf, device="cpu")
            for k in tmp:
                weight_map[k] = fname

    # ── 2-file LRU shard cache ──────────────────────────────────────────────
    loaded_files: dict = {}

    def get_tensor(tensor_name: str) -> torch.Tensor:
        fname = weight_map[tensor_name]
        if fname not in loaded_files:
            fpath = os.path.join(fp8_path, fname)
            loaded_files[fname] = load_file(fpath, device="cpu")
        return loaded_files[fname][tensor_name]

    # ── Process each shard ──────────────────────────────────────────────────
    shard_files = sorted(set(weight_map.values()))
    fp8_weight_names: list[str] = []
    new_weight_map: dict = {}

    print(f"[fp8_cast] Found {len(shard_files)} shard(s) in {fp8_path}")

    for shard_fname in shard_files:
        print(f"  Processing shard: {shard_fname}")
        shard_path = os.path.join(fp8_path, shard_fname)
        current = load_file(shard_path, device="cpu")
        loaded_files[shard_fname] = current

        new_state: dict = {}
        for weight_name, weight in current.items():

            # Skip scale_inv tensors (they'll be handled by their FP8 partner)
            if weight_name.endswith("_scale_inv"):
                continue

            if weight.element_size() == 1:
                # ── FP8 weight: dequantize ──────────────────────────────
                scale_inv_name = f"{weight_name}_scale_inv"
                if scale_inv_name in weight_map:
                    scale_inv = get_tensor(scale_inv_name)
                    bf16_weight = weight_dequant(weight, scale_inv)
                    new_state[weight_name] = bf16_weight
                    fp8_weight_names.append(weight_name)
                    print(f"    Dequantized: {weight_name} "
                          f"{list(weight.shape)} FP8 -> BF16")
                else:
                    # No paired scale: cast directly (best-effort)
                    print(f"    Warning: no scale_inv for {weight_name}, "
                          f"casting directly to BF16.")
                    new_state[weight_name] = weight.to(torch.bfloat16)

            elif weight.dtype == torch.float8_e4m3fn:
                # Catch any other FP8 dtype variant without scale
                new_state[weight_name] = weight.to(torch.bfloat16)
            else:
                # Non-quantized tensor: pass through
                new_state[weight_name] = weight.to(torch.bfloat16) if weight.is_floating_point() else weight

        # Write converted shard
        out_shard = os.path.join(bf16_path, shard_fname)
        save_file(new_state, out_shard)
        for k in new_state:
            new_weight_map[k] = shard_fname
        print(f"  Saved -> {out_shard}")

        # LRU eviction: keep at most 2 shards in memory
        if len(loaded_files) > 2:
            oldest = next(iter(loaded_files))
            del loaded_files[oldest]

    # ── Write updated index (scale_inv entries removed) ─────────────────────
    _make_index(bf16_path, new_weight_map)
    print(f"\n[fp8_cast] Converted {len(fp8_weight_names)} FP8 tensors to BF16.")
    print(f"[fp8_cast] Output written to: {bf16_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert LasmoidV1 FP8 checkpoint weights to BF16"
    )
    parser.add_argument(
        "--input-fp8-path",
        type=str,
        required=True,
        help="Directory containing FP8 .safetensors checkpoint files"
    )
    parser.add_argument(
        "--output-bf16-path",
        type=str,
        required=True,
        help="Output directory for BF16 converted checkpoint"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Quantization block size used during FP8 training (default: 128)"
    )
    args = parser.parse_args()
    main(args.input_fp8_path, args.output_bf16_path, args.block_size)
