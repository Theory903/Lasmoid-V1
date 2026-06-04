import torch
from huggingface_hub import hf_hub_download
import os, sys
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model import LasmoidV1, ModelArgs
from train_kaggle import MODEL_CONFIGS
from inference.kernel import act_quant

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
    
    # Run the encoder attn manually
    xb = torch.randint(0, model_args.vocab_size - 1, (2, 256), dtype=torch.long)
    
    with torch.no_grad():
        H_enc = model.emb(xb).bfloat16()
        print(f"H_enc std: {H_enc.std().item():.4f}")
        
        x = model.encoder_norm(H_enc)
        print(f"encoder_norm(H_enc) std: {x.std().item():.4f}")
        
        # Step-by-step MLA forward
        mla = model.encoder_attn
        freqs_cis = model.freqs_cis[:256]
        
        B, N, _ = x.shape
        
        # 1. Q projection
        q_a = mla.wq_a(x)
        print(f"q_a std: {q_a.std().item():.4f}")
        
        q_norm = mla.q_norm(q_a)
        print(f"q_norm std: {q_norm.std().item():.4f}")
        
        q_b = mla.wq_b(q_norm)
        print(f"q_b std: {q_b.std().item():.4f}")
        
        q = q_b.unflatten(-1, (mla.n_heads, mla.head_dim))
        print(f"q unflattened std: {q.std().item():.4f}")
        
        # Per-head RMS normalisation
        q_rsqrt = torch.rsqrt(q.square().mean(-1, keepdim=True) + mla.eps)
        q = q * q_rsqrt
        print(f"q after rms norm std: {q.std().item():.4f}")
        
        q_nope, q_rope = q[..., :-mla.rope_head_dim], q[..., -mla.rope_head_dim:]
        # apply_rotary_emb
        from inference.model import apply_rotary_emb
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        print(f"q_rope after rotary std: {q_rope.std().item():.4f}")
        q = torch.cat([q_nope, q_rope], dim=-1)
        print(f"q final std: {q.std().item():.4f}")
        
        # 2. KV projection
        kv = mla.wkv(x)
        print(f"kv raw std: {kv.std().item():.4f}")
        kv = mla.kv_norm(kv)
        print(f"kv_norm std: {kv.std().item():.4f}")
        
        kv_nope, kv_rope = kv[..., :-mla.rope_head_dim], kv[..., -mla.rope_head_dim:]
        kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)
        
        # QAT: simulate FP8 on nope dims
        scale_fmt = model_args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if model_args.scale_dtype == "fp8" else torch.float32
        nope_q, scale = act_quant(kv[..., :-mla.rope_head_dim].contiguous(), 64, scale_fmt, scale_dtype)
        nope_q = nope_q.to(kv.dtype) * scale.to(kv.dtype)
        kv = torch.cat([nope_q, kv[..., -mla.rope_head_dim:]], dim=-1)
        print(f"kv final std: {kv.std().item():.4f}")
        
        # 3. Attention SDPA
        q_t = q.transpose(1, 2)
        kv_h = kv.unsqueeze(1).expand(-1, mla.n_heads, -1, -1)
        print(f"q_t std: {q_t.std().item():.4f}, kv_h std: {kv_h.std().item():.4f}")
        
        is_causal = (N > 1)
        attn_out = F.scaled_dot_product_attention(
            q_t, kv_h, kv_h,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=mla.softmax_scale,
        )
        print(f"attn_out std: {attn_out.std().item():.4f}")
        
        # De-rotate rope dims
        attn_out_perm = attn_out.transpose(1, 2)
        attn_nope, attn_rope = attn_out_perm[..., :-mla.rope_head_dim], attn_out_perm[..., -mla.rope_head_dim:]
        attn_rope = apply_rotary_emb(attn_rope, freqs_cis, inverse=True)
        attn_out_perm = torch.cat([attn_nope, attn_rope], dim=-1)
        print(f"attn_out_perm final std: {attn_out_perm.std().item():.4f}")
        
        # 4. Grouped O projection
        o = attn_out_perm.reshape(B, N, mla.n_groups, -1)
        print(f"o reshape std: {o.std().item():.4f}")
        
        wo_a_w = mla.wo_a.weight.view(mla.n_groups, mla.o_lora_rank, -1)
        print(f"wo_a_w std: {wo_a_w.std().item():.4f}")
        
        o_einsum = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a_w.float())
        print(f"o_einsum std: {o_einsum.std().item():.4f}")
        
        o_flat = o_einsum.flatten(2).to(x.dtype)
        print(f"o_flat std: {o_flat.std().item():.4f}")
        
        out = mla.wo_b(o_flat)
        print(f"out final std: {out.std().item():.4f}")

if __name__ == "__main__":
    main()
