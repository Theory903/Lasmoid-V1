import os
import sys
import torch
import json
import tiktoken

# Add the parent directory of scratch to sys.path so we can import from inference
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.model import LasmoidV1, ModelArgs
from inference.generate import generate

def prepare_model_for_checkpoint(model, state_dict):
    # Find the number of concept blocks from state_dict
    block_indices = []
    for key in state_dict.keys():
        if key.startswith("memory.concept_blocks."):
            parts = key.split(".")
            block_indices.append(int(parts[2]))
    num_blocks_in_ckpt = max(block_indices) + 1 if block_indices else 1
    
    # Expand model's concept blocks
    current_num_blocks = len(model.memory.concept_blocks)
    if num_blocks_in_ckpt > current_num_blocks:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for i in range(current_num_blocks, num_blocks_in_ckpt):
            new_block = model.memory._create_block(model.args).to(device=device, dtype=dtype)
            model.memory.concept_blocks.append(new_block)
            
        # Re-register buffers with correct shape
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
        print(f"Dynamically expanded concept memory from {current_num_blocks} to {num_blocks_in_ckpt} blocks.")

def evaluate():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    checkpoints_dir = os.path.join(parent_dir, "checkpoints")
    checkpoint_files = [
        "lasmoid_step_00000050.pt",
        "lasmoid_step_00000200.pt",
        "lasmoid_step_00000400.pt",
        "lasmoid_step_00000600.pt",
        "lasmoid_step_00000800.pt",
        "lasmoid_step_00001000.pt",
        "lasmoid_latest.pt"
    ]
    
    prompts = [
        "ROMEO:",
        "JULIET:",
        "To be, or not to be:"
    ]
    
    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = enc.eot_token
    
    for ckpt_name in checkpoint_files:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name} (not found)")
            continue
            
        print(f"\n==========================================")
        print(f"Loading checkpoint: {ckpt_name}")
        print(f"==========================================")
        
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = sd.get("model_state_dict", sd)
            
            # Load ModelArgs from checkpoint if available, otherwise fallback to local config.json
            model_args = sd.get("model_args")
            if model_args is None:
                config_path = os.path.join(parent_dir, "inference", "configs", "config.json")
                with open(config_path) as f:
                    config_dict = json.load(f)
                from dataclasses import fields
                valid_fields = {f.name for f in fields(ModelArgs)}
                filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
                model_args = ModelArgs(**filtered_config)
            
            # Instantiate model with correct args from the checkpoint
            model = LasmoidV1(model_args).to(device)
            
            # Re-register freqs_cis with a larger size to avoid out-of-bounds index error in decode loop
            from inference.model import precompute_freqs_cis
            larger_freqs_cis = precompute_freqs_cis(
                model_args.rope_head_dim,
                model_args.max_seq_len + 1024,
                model_args.original_seq_len,
                model_args.rope_theta,
                model_args.rope_factor,
                model_args.beta_fast,
                model_args.beta_slow
            ).to(device)
            model.register_buffer("freqs_cis", larger_freqs_cis, persistent=False)
            
            # Prepare the model to match checkpoint's dynamically grown memory blocks
            prepare_model_for_checkpoint(model, state_dict)
            
            # Load weights
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            # Evaluate each prompt
            for prompt in prompts:
                prompt_tokens = [enc.encode(prompt, allowed_special={"<|endoftext|>"})]
                completion_tokens = generate(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=40,
                    eos_id=eos_token_id,
                    temperature=0.8,
                    top_p=0.9,
                    min_p=0.0
                )
                completion = enc.decode(completion_tokens[0])
                print(f"Prompt: {prompt}")
                print(f"Output: {completion}")
                print(f"------------------------------------------")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error evaluating {ckpt_name}: {e}")

if __name__ == "__main__":
    evaluate()
