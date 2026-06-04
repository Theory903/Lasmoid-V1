import torch
from huggingface_hub import hf_hub_download
import os, sys
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS

def main():
    cfg = MODEL_CONFIGS["10M"].copy()
    cfg.pop("_verified_params", None)
    cfg["max_seq_len"] = 256
    cfg["max_batch_size"] = 2
    
    model_args = ModelArgs(**cfg)
    model = LasmoidV1(model_args)
    
    # Download checkpoint
    token = os.getenv("HF_TOKEN")
    repo_id = "Theory903/lasmoid-10m"
    filename = "lasmoid_step_00000400.pt"
    
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    model.train()
    
    # Let's register hooks to capture activations and their gradients.
    activations = {}
    gradients = {}
    
    def get_hook(name):
        def hook(module, input, output):
            # output could be a tensor or tuple of tensors
            if isinstance(output, tuple):
                activations[name] = [o.detach().clone() if isinstance(o, torch.Tensor) else None for o in output]
            else:
                activations[name] = output.detach().clone()
        return hook

    def get_grad_hook(name):
        def hook(grad):
            gradients[name] = grad.detach().clone()
            return grad
        return hook

    # Hooks for activations
    model.emb.register_forward_hook(get_hook("emb"))
    model.encoder_norm.register_forward_hook(get_hook("encoder_norm"))
    model.encoder_attn.register_forward_hook(get_hook("encoder_attn"))
    model.memory.register_forward_hook(get_hook("memory"))
    
    # Let's run a single forward step
    xb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    yb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    
    # We want to register hooks on specific tensors during the forward pass.
    # To do that, we can temporarily patch ElasticSparseConceptMemory.process_chunk
    original_process_chunk = model.memory.process_chunk
    
    tensors_to_watch = {}
    def patched_process_chunk(encoder_hidden, block_idx=0):
        # We capture intermediate tensors here
        B, N_enc, D = encoder_hidden.shape
        B = encoder_hidden.size(0)
        Q = model.memory.latent_queries.expand(B, -1, -1)
        
        n_heads = 8
        head_dim = model.memory.dim // n_heads
        
        Q_4d = Q.view(B, model.memory.num_concepts, n_heads, head_dim).transpose(1, 2).to(encoder_hidden.dtype)
        KV_4d = encoder_hidden.view(B, N_enc, n_heads, head_dim).transpose(1, 2)
        
        # Register gradients
        Q_4d.requires_grad_(True)
        KV_4d.requires_grad_(True)
        
        tensors_to_watch["Q_4d"] = Q_4d
        tensors_to_watch["KV_4d"] = KV_4d
        
        Q_4d.register_hook(get_grad_hook("Q_4d"))
        KV_4d.register_hook(get_grad_hook("KV_4d"))
        
        pooled_4d = F.scaled_dot_product_attention(Q_4d, KV_4d, KV_4d)
        tensors_to_watch["pooled_4d"] = pooled_4d
        pooled_4d.register_hook(get_grad_hook("pooled_4d"))
        
        pooled = pooled_4d.transpose(1, 2).contiguous().view(B, model.memory.num_concepts, model.memory.dim)
        tensors_to_watch["pooled"] = pooled
        pooled.register_hook(get_grad_hook("pooled"))
        
        quantized, loss = model.memory.concept_blocks[block_idx](pooled)
        tensors_to_watch["quantized"] = quantized
        quantized.register_hook(get_grad_hook("quantized"))
        tensors_to_watch["hcm_loss"] = loss
        
        # we will bypass the spawn checking logic to keep it simple, just return quantized and loss
        # but wait, let's keep EMA update
        with torch.no_grad():
            mean_quant = quantized.detach().mean(dim=0, keepdim=True)
            model.memory.slot_ema.data.copy_(model.memory.hcm_ema_alpha * model.memory.slot_ema.data + (1.0 - model.memory.hcm_ema_alpha) * mean_quant)
            model.memory.slot_db.data[block_idx].copy_(model.memory.slot_ema.data[0])
            model.memory.meta_centroids.data[block_idx].copy_(torch.mean(model.memory.slot_db[block_idx], dim=0))
            
        return quantized, loss

    model.memory.process_chunk = patched_process_chunk

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_nxt, logits_nxt2, _, _ = model(xb, yb)
        z_loss = model.last_z_loss
        ce = F.cross_entropy(logits_nxt.view(-1, model_args.vocab_size), yb.view(-1))
        # Keep it identical to test_train_stability.py loss
        loss = ce + 0.3 * logits_nxt2.mean() * 0.0 + model_args.router_z_loss_coeff * z_loss

    loss.backward()

    # Now let's print activations stats
    print("\n--- Activation Statistics ---")
    for name in ["emb", "encoder_norm", "encoder_attn"]:
        act = activations.get(name)
        if act is not None:
            if isinstance(act, list):
                print(f"  {name:15s} | List of len {len(act)}")
            else:
                print(f"  {name:15s} | shape: {str(list(act.shape)):15s} | mean: {act.mean().item():.4f} | std: {act.std().item():.4f} | max: {act.abs().max().item():.4f}")
    
    for name, t in tensors_to_watch.items():
        if isinstance(t, torch.Tensor) and t.ndim > 0:
            print(f"  {name:15s} | shape: {str(list(t.shape)):15s} | mean: {t.mean().item():.4f} | std: {t.std().item():.4f} | max: {t.abs().max().item():.4f}")
        elif isinstance(t, torch.Tensor):
            print(f"  {name:15s} | value: {t.item():.4f}")

    # Now print gradient stats
    print("\n--- Intermediate Tensor Gradient Statistics ---")
    for name, grad in gradients.items():
        print(f"  {name:15s} | shape: {str(list(grad.shape)):15s} | mean: {grad.mean().item():.4f} | std: {grad.std().item():.4f} | max: {grad.abs().max().item():.4f} | norm: {grad.norm().item():.4f}")

    print("\n--- Parameter Gradient Norms (Encoder & Embedding) ---")
    norms = []
    for name, p in model.named_parameters():
        if "encoder" in name or "emb" in name or "memory" in name:
            if p.grad is not None:
                norms.append((name, p.grad.norm().item(), p.shape))
    norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm, shape in norms:
        print(f"  {name:50s} | shape: {str(list(shape)):15s} | grad norm: {norm:.4f}")

if __name__ == "__main__":
    main()
