import torch
import torch.nn.functional as F
import random
import tiktoken
import sys
import os

# Add root folder to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.model import LasmoidV1, ModelArgs
from train import Muon

enc = tiktoken.get_encoding("gpt2")
AGENTS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
PASSCODES = ["1122", "9988", "4455", "7733", "6611"]
FILLERS = [
    "The weather is cloudy today.",
    "The system is running nominal checks.",
    "Waiting for further instructions.",
    "Commencing background diagnostic."
]

def generate_synthetic_batch(batch_size=16):
    """
    Generates a batch of associative recall prompts.
    Split into x_enc (prompt only) and x_dec (full sequence up to passcode) to prevent future-token leakage in encoder.
    """
    x_enc_batch = []
    x_dec_batch = []
    y_batch = []
    
    for _ in range(batch_size):
        agent = random.choice(AGENTS)
        code = random.choice(PASSCODES)
        
        pre_filler = random.choice(FILLERS)
        post_filler = random.choice(FILLERS)
        
        # 1. Prompt Only (no answer code at the end)
        prompt_only = f"{pre_filler} Agent {agent} has passcode {code}. {post_filler} Query: What is the passcode for {agent}? Answer:"
        # 2. Full Sequence (contains answer code and EOT)
        full_seq = f"{pre_filler} Agent {agent} has passcode {code}. {post_filler} Query: What is the passcode for {agent}? Answer: {code} <|endoftext|>"
        
        tokens_enc = enc.encode(prompt_only, allowed_special={"<|endoftext|>"})
        tokens_dec = enc.encode(full_seq, allowed_special={"<|endoftext|>"})
        
        x_dec_tokens = tokens_dec[:-1]
        y_dec_tokens = tokens_dec[1:]
        
        seq_len = 64
        
        # Pad encoder sequence
        if len(tokens_enc) < seq_len:
            tokens_enc = [enc.eot_token] * (seq_len - len(tokens_enc)) + tokens_enc
        else:
            tokens_enc = tokens_enc[:seq_len]
            
        # Pad decoder sequences
        if len(x_dec_tokens) < seq_len:
            pad_len = seq_len - len(x_dec_tokens)
            x_dec_tokens = [enc.eot_token] * pad_len + x_dec_tokens
            y_dec_tokens = [enc.eot_token] * pad_len + y_dec_tokens
        else:
            x_dec_tokens = x_dec_tokens[:seq_len]
            y_dec_tokens = y_dec_tokens[:seq_len]
            
        x_enc_batch.append(torch.tensor(tokens_enc, dtype=torch.long))
        x_dec_batch.append(torch.tensor(x_dec_tokens, dtype=torch.long))
        y_batch.append(torch.tensor(y_dec_tokens, dtype=torch.long))
        
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return (
        torch.stack(x_enc_batch).to(device),
        torch.stack(x_dec_batch).to(device),
        torch.stack(y_batch).to(device)
    )

def run_test():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Concept Retrieval Test on {device.upper()}...")

    args = ModelArgs()
    model = LasmoidV1(args).to(device)

    # Partition optimizers
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if len(p.shape) == 2 and "emb" not in name and "head" not in name and "adj" not in name: muon_params.append(p)
        else: adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=1e-3)

    test_iters = 1500
    print(f"Training on synthetic associative data for {test_iters} steps...")

    for step in range(test_iters + 1):
        x_enc, x_dec, yb = generate_synthetic_batch(batch_size=16)
        
        # We do not compute next-next token loss during synthetic recall to focus purely on the retrieval target
        logits, _, _, _ = model(x_enc, x_dec)
        
        ce_loss = F.cross_entropy(logits.view(-1, args.vocab_size), yb.view(-1))
        
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        ce_loss.backward()
        opt_muon.step()
        opt_adamw.step()
        
        if step % 150 == 0:
            print(f"Step {step:4d} | CE Loss: {ce_loss.item():.4f}")

    print("\n=== RUNNING O(1) RETRIEVAL EVALUATION ===")
    model.eval()

    test_agent = "Bravo"
    test_code = "9988"
    test_prompt = f"The system is running nominal checks. Agent Bravo has passcode {test_code}. Waiting for further instructions. Query: What is the passcode for {test_agent}? Answer:"

    print(f"Context: '{test_prompt}'")
    print(f"Expected Target: ' {test_code}'")

    idx = torch.tensor(enc.encode(test_prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device).unsqueeze(0)

    # Generate 2 tokens using model.generate (with max_len=64 matching training sequence shape)
    with torch.no_grad():
        generated_ids = model.generate(idx, max_new_tokens=2, temperature=0.1, top_k=1, pad_token=enc.eot_token, max_len=64)

    output_text = enc.decode(generated_ids[0].tolist())
    retrieved_answer = output_text.split("Answer:")[-1]

    print(f"Retrieved Answer:{retrieved_answer}")

    if test_code in retrieved_answer:
        print("\n[SUCCESS] The Write Master successfully retrieved the Needle from the Hierarchical Concept Memory!")
    else:
        print("\n[FAILED] The Hierarchical Concept Slots failed to hold the specific passcode.")

if __name__ == "__main__":
    run_test()
