import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Add the parent directory of scratch to sys.path so we can import from inference
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.model import LasmoidV1, ModelArgs
from train import Muon

def overfit_test():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 1. Create a tiny toy dataset
    qa_pairs = [
        ("2+2", "4"),
        ("cat", "animal"),
        ("hello", "hi"),
        ("india", "india"),
        ("Who is building Lasmoid?", "Abhishek")
    ]
    
    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc.eot_token
    
    dataset_texts = []
    for q, a in qa_pairs:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        dataset_texts.append(text)
        
    print("Toy dataset texts:")
    for t in dataset_texts:
        print(repr(t))
        
    # Model Args
    model_args = ModelArgs(
        dim=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=128,
        max_batch_size=8,
        num_residual_streams=3 # Hyper, Memory, Concept (from config.json)
    )
    
    # Encode and pad inputs
    max_len = model_args.max_seq_len
    xs, ys = [], []
    for text in dataset_texts:
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        tokens.append(eot_id)
        
        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        x_tokens = tokens[:-1]
        y_tokens = tokens[1:]
        
        # Pad with EOT
        pad_len = (max_len - 1) - len(x_tokens)
        if pad_len > 0:
            x_tokens = x_tokens + [eot_id] * pad_len
            y_tokens = y_tokens + [eot_id] * pad_len
            
        xs.append(torch.tensor(x_tokens, dtype=torch.long))
        ys.append(torch.tensor(y_tokens, dtype=torch.long))
        
    x_batch = torch.stack(xs).to(device) # shape: [5, 127]
    y_batch = torch.stack(ys).to(device) # shape: [5, 127]
    
    # 2. Initialize Model
    model = LasmoidV1(model_args).to(device)
    print("Model initialized.")
    
    # Partition Parameters for Muon / AdamW
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if (
            len(p.shape) == 2
            and "emb" not in name
            and "head" not in name
            and "adj" not in name
            and "gate" not in name
            and "hc" not in name
        ):
            muon_params.append(p)
        else:
            adamw_params.append(p)
            
    opt_muon = Muon(muon_params, lr=1e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=6e-4)
    
    # 3. Train Loop
    model.train()
    epochs = 150
    print(f"Starting overfitting test for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        
        # Forward pass: Encoder gets context (x_batch), Decoder gets sequence (x_batch)
        logits_next, logits_next_next, _, _ = model(x_batch, x_batch)
        
        # CE Loss
        loss_ce = F.cross_entropy(logits_next.view(-1, model_args.vocab_size), y_batch.view(-1))
        
        # MTP Loss
        loss_mtp = torch.tensor(0.0, device=device)
        if logits_next_next is not None:
            loss_mtp = F.cross_entropy(
                logits_next_next.view(-1, model_args.vocab_size),
                y_batch[:, 1:].contiguous().view(-1)
            )
            
        loss = loss_ce + 0.3 * loss_mtp
        loss.backward()
        
        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        opt_muon.step()
        opt_adamw.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs:3d} | Loss: {loss.item():.6f} | CE: {loss_ce.item():.6f} | MTP: {loss_mtp.item():.6f}")
            
    # 4. Evaluate Memorization
    model.eval()
    print("\n--- Evaluating Memorization ---")
    
    with torch.no_grad():
        for q, expected in qa_pairs:
            prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            prompt_tokens = enc.encode(prompt)
            idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            cond_len = idx.shape[1]
            
            # Run encoder once (no padding needed, keep prompt at pos 0 to match training)
            logits, _, concept_db, memory_state = model(idx, idx, start_pos=0)
            
            # Autoregressively generate next tokens
            idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens = []
            
            current_pos = cond_len
            for step in range(15):
                token_id = idx_next.item()
                # Print debug information for generated tokens
                print(f"  [Step {step}] Predicted ID: {token_id} -> {enc.decode([token_id])!r}")
                
                # Check for stop tokens
                if token_id == eot_id:
                    break
                    
                generated_tokens.append(token_id)
                
                logits, _, _, _ = model(
                    x_enc=None,
                    x_dec=idx_next,
                    concept_db=concept_db,
                    memory_state=memory_state,
                    start_pos=current_pos
                )
                idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_pos += 1
                
            completion = enc.decode(generated_tokens).strip()
            # Clean up trailing special tags if any
            if "<|im_end|>" in completion:
                completion = completion.split("<|im_end|>")[0].strip()
            print(f"Prompt:   {q!r}")
            print(f"Expected: {expected!r}")
            print(f"Model:    {completion!r}")
            if completion == expected:
                print("Result:   PASS ✓")
            else:
                print("Result:   FAIL ✗")
            print("-" * 40)

if __name__ == "__main__":
    overfit_test()
