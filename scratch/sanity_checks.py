import os
import sys
import json
import torch
import torch.nn.functional as F
import tiktoken

# Add the parent directory of scratch to sys.path so we can import from inference
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.model import LasmoidV1, ModelArgs
from inference.generate import generate

def test_0_tokenizer_and_vocab():
    print("\n--- Test 0: Tokenizer & Vocab Size Verification ---")
    enc = tiktoken.get_encoding("gpt2")
    text = "hello world"
    ids = enc.encode(text)
    decoded = enc.decode(ids)
    print(f"Original Text: {text!r}")
    print(f"Decoded Text:  {decoded!r}")
    
    if text == decoded:
        print("Tokenizer encode/decode check: PASS ✓")
    else:
        print("Tokenizer encode/decode check: FAIL ✗")
        
    # Vocab size check
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(parent_dir, "checkpoints", "lasmoid_latest.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = sd.get("model_state_dict", sd)
        emb_weight = state_dict.get("emb.weight")
        if emb_weight is not None:
            ckpt_vocab_size = emb_weight.shape[0]
            tokenizer_vocab_size = enc.n_vocab
            print(f"Checkpoint Embedding Vocab Size: {ckpt_vocab_size}")
            print(f"Tokenizer Vocab Size:           {tokenizer_vocab_size}")
            if ckpt_vocab_size == tokenizer_vocab_size:
                print("Vocabulary size match check: PASS ✓")
            else:
                print("Vocabulary size match check: FAIL ✗")
        else:
            print("Could not find emb.weight in checkpoint state dict.")
    else:
        print(f"Checkpoint not found at {ckpt_path}")

def test_1_fast_diagnostics():
    print("\n--- Test 1: Fast Diagnostic Prompts ---")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(parent_dir, "checkpoints")
    config_path = os.path.join(parent_dir, "inference", "configs", "config.json")
    
    from inference.generate import load_checkpoint_and_model
    try:
        model, model_args = load_checkpoint_and_model(ckpt_dir, config_path, device)
        model.eval()
        
        enc = tiktoken.get_encoding("gpt2")
        eos_token_id = enc.eot_token
        
        prompts = ["aaaaaa", "123456", "hello", "the", "and", "cat", "dog"]
        
        for prompt in prompts:
            prompt_tokens = [enc.encode(prompt, allowed_special={"<|endoftext|>"})]
            completion_tokens = generate(
                model=model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=25,
                eos_id=eos_token_id,
                temperature=0.8,
                top_p=0.9,
                min_p=0.0
            )
            completion = enc.decode(completion_tokens[0])
            print(f"Prompt: {prompt!r}")
            print(f"Output: {completion!r}")
            print("-" * 40)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to run Fast Diagnostics: {e}")

@torch.inference_mode()
def test_2_next_token_probs():
    print("\n--- Test 2: Next Token Probability Test ---")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(parent_dir, "checkpoints")
    config_path = os.path.join(parent_dir, "inference", "configs", "config.json")
    
    from inference.generate import load_checkpoint_and_model
    try:
        model, model_args = load_checkpoint_and_model(ckpt_dir, config_path, device)
        model.eval()
        
        enc = tiktoken.get_encoding("gpt2")
        prompt = "To be or not to be, that is the"
        idx = torch.tensor([enc.encode(prompt, allowed_special={"<|endoftext|>"})], dtype=torch.long, device=device)
        
        # Pad to max_len to match prefill behavior
        max_len = model.args.max_seq_len
        cond_len = idx.shape[1]
        eos_id = enc.eot_token
        if cond_len < max_len:
            padding = torch.full((1, max_len - cond_len), eos_id, dtype=idx.dtype, device=device)
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -max_len:]
            
        logits, _, _, _ = model(idx_padded, idx_padded, start_pos=0)
        last_logits = logits[0, -1, :] # shape: [vocab_size]
        probs = F.softmax(last_logits, dim=-1)
        
        topk_probs, topk_idxs = torch.topk(probs, 10)
        
        print(f"Prompt: {prompt!r}")
        print("Top 10 predicted next tokens:")
        for i in range(10):
            token_id = topk_idxs[i].item()
            token_prob = topk_probs[i].item()
            token_text = enc.decode([token_id])
            print(f"  {i+1}: {token_text!r} (ID: {token_id}, Prob: {token_prob:.4%})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed next token test: {e}")

if __name__ == "__main__":
    test_0_tokenizer_and_vocab()
    test_1_fast_diagnostics()
    test_2_next_token_probs()
