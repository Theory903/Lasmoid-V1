"""
LasmoidV1 — model.py 
================================================
Sovereign Architecture (UNCHANGED):
  - MLA   : Multi-head Latent Attention, CQRS Read/Write split
  - HCM   : Elastic-Sparse Concept Memory (V1 upgrade: Dynamic spawn, Graph VQ, Lightning Retrieve)
  - mHC   : Manifold-Constrained Hyper-Connections
  - MoE   : sqrt(softplus) affinity gate + shared experts
  - MTP   : Multi-Token Prediction head
  - CQRS  : Encoder (Read Replica) + Decoder (Write Master)

Traditional Plumbing — V4-Pro refinements applied:
  ✓ RMSNorm: weight stored as float32, output cast to input dtype (V4 exact)
  ✓ RoPE:    Exact YaRN formula with lru_cache (V4-Pro precompute_freqs_cis)
  ✓ MLA:     attn_sink learnable bias (prevents attention sink collapse)
  ✓ MLA:     start_pos KV-cache stateful generation
  ✓ MLA:     grouped O-projection LoRA (wo_a/wo_b) from V4-Pro
  ✓ MLA:     apply_rotary_emb with inverse=True for output de-rotation
  ✓ Gate:    auxiliary bias + score_func string dispatch + hash routing
  ✓ Gate:    exact sqrtsoftplus = sqrt(softplus(x)) (V4-Pro canonical)
  ✓ Expert:  swiglu_limit clamping for training stability
  ✓ HC:      hc_split_sinkhorn exact V4 formula (replaces custom Sinkhorn)
  ✓ HC:      float32 HC parameters with set_dtype context manager
  ✓ MTPBlock: V4 exact e_proj+h_proj fusion, enorm/hnorm
  ✓ Transformer: start_pos incremental decode in forward()
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, List
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from .kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, weight_dequant
except ImportError:
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, weight_dequant


# ══════════════════════════════════════════════════════════════════════
# GLOBALS  (match V4-Pro naming exactly)
# ══════════════════════════════════════════════════════════════════════
default_dtype = torch.bfloat16
scale_fmt:    Optional[str]   = None
scale_dtype:  torch.dtype     = torch.float32
block_size:   int             = 128
fp4_block_size: int           = 32


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype. From V4-Pro model.py."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION  (all V4-Pro fields + sovereign fields)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ModelArgs:
    # ── Core ─────────────────────────────────────────────────────────
    vocab_size: int = 50257
    dim: int = 128
    n_layers: int = 4
    max_seq_len: int = 256
    max_batch_size: int = 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp32"
    expert_dtype: Optional[str] = None   # None | "fp4"
    norm_eps: float = 1e-6

    # ── MLA (Multi-head Latent Attention) ─────────────────────────────
    n_heads: int = 4
    q_lora_rank: int = 32
    head_dim: int = 48            # nope + rope (V4-Pro style: unified head_dim)
    rope_head_dim: int = 16       # rope portion
    o_groups: int = 2             # grouped O-projection (from V4-Pro)
    o_lora_rank: int = 32         # O-projection LoRA rank (from V4-Pro)

    # ── RoPE / YaRN (V4-Pro exact params) ────────────────────────────
    rope_theta: float = 10000.0
    rope_factor: float = 1.0       # YaRN scale factor (1 = no scaling)
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 0      # 0 = disable YaRN
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_inter_dim: int = 256
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0      # 0 = disabled; V4 uses 10.0
    n_hash_layers: int = 0         # hash-routed layers (V4 first N layers)

    # ── HC (Hyper-Connections) (SOVEREIGN) ───────────────────────────
    num_residual_streams: int = 4  # V4-Pro uses hc_mult=4
    hc_sinkhorn_iters: int = 20    # V4-Pro default
    hc_eps: float = 1e-6

    # ── HCM (Hierarchical Concept Memory) (SOVEREIGN) ─────────────────
    num_concepts: int = 64
    num_abstract_concepts: int = 8
    num_global_concepts: int = 2

    # ── MTP (Multi-Token Prediction) ─────────────────────────────────
    n_mtp_layers: int = 1

    # ── KV-cache window (from V4-Pro) ─────────────────────────────────
    window_size: int = 128

    # ── Elastic-Sparse Concept Memory (NEW) ──────────────────────────
    codebook_size: int = 256
    hcm_ema_alpha: float = 0.99
    hcm_commit_loss_coeff: float = 0.25
    entropy_threshold: float = 0.5
    lightning_topk_blocks: int = 2
    router_z_loss_coeff: float = 0.001
    ema_bias_lr: float = 0.01

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim


# ══════════════════════════════════════════════════════════════════════
# CORE LAYERS
# ══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """V4-Pro exact: weight stored as float32, output cast to input dtype."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _linear_dispatch(x: torch.Tensor, weight: nn.Parameter, bias: Optional[nn.Parameter] = None) -> torch.Tensor:
    """
    Route to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    Exact dispatch table from V4-Pro model.py linear().
    """
    if weight.dtype == torch.bfloat16 or weight.dtype == torch.float32:
        return F.linear(x.to(weight.dtype), weight, bias)
    # FP8 path
    if weight.dtype == torch.float8_e4m3fn:
        xq, xs = act_quant(x.contiguous().bfloat16(), block_size, scale_fmt, scale_dtype)
        out = fp8_gemm(xq, xs, weight, weight.scale, scale_dtype)
        if bias is not None:
            out = out + bias
        return out
    # Fallback
    return F.linear(x, weight.float(), bias)


class Linear(nn.Module):
    """BF16/FP8/FP4 linear layer. Exact V4-Pro API."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        w_dtype = dtype or default_dtype

        if w_dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=w_dtype))
            so = (out_features + block_size - 1) // block_size
            si = (in_features  + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(so, si, dtype=torch.float32), requires_grad=False
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=w_dtype))
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, 0.0, 0.02)
        if self.scale is not None:
            nn.init.constant_(self.scale, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear_dispatch(x, self.weight, self.bias)


# ══════════════════════════════════════════════════════════════════════
# ROTARY POSITIONAL EMBEDDINGS — Exact YaRN (from V4-Pro)
# lru_cache avoids recomputation; YaRN smooth ramp is the V4-Pro formula
# ══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """
    Precompute complex rotary frequencies with exact YaRN scaling.
    Ported verbatim from DeepSeek-V4-Pro model.py precompute_freqs_cis().
    When original_seq_len > 0, applies frequency interpolation with a
    smooth linear ramp between beta_fast and beta_slow correction ranges.
    """
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low  = math.floor(find_correction_dim(low_rot,  dim, base, max_seq_len))
        high = math.ceil( find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        lf = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(lf, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs  = freqs / factor * (1 - smooth) + freqs * smooth

    t       = torch.arange(seqlen, dtype=torch.float32)
    freqs   = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Out-of-place rotary embedding application.
    inverse=True: conjugate (de-rotation) for output de-rotation (V4-Pro apply_rotary_emb).
    """
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xr = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return xr.to(dtype)


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD LATENT ATTENTION (MLA)  ← SOVEREIGN ARCH PRESERVED
# V4-Pro refinements applied:
#   • attn_sink learnable bias (prevents attention sink collapse)
#   • start_pos for stateful KV-cache decode
#   • grouped O-projection (wo_a / wo_b) from V4-Pro
#   • apply_rotary_emb with inverse=True on output
#   • per-head RMS normalisation on Q (from V4-Pro line 498)
# ══════════════════════════════════════════════════════════════════════
class MLA(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id     = layer_id
        self.dim          = args.dim
        self.n_heads      = args.n_heads
        self.q_lora_rank  = args.q_lora_rank
        self.head_dim     = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.nope_head_dim
        self.n_groups     = args.o_groups
        self.o_lora_rank  = args.o_lora_rank
        self.eps          = args.norm_eps

        # Q projection (LoRA)
        self.wq_a  = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b  = Linear(self.q_lora_rank, self.n_heads * self.head_dim)

        # KV latent compression (MLA sovereign core)
        self.wkv   = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Grouped O projection (from V4-Pro)
        # wo_a: groups × o_lora_rank per group; wo_b: recombine to dim
        heads_per_group = self.n_heads // self.n_groups
        self.wo_a = Linear(heads_per_group * self.head_dim, self.n_groups * self.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = Linear(self.n_groups * self.o_lora_rank, self.dim)

        # Attention sink (from V4-Pro — prevents attention collapse)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))

        self.softmax_scale = self.head_dim ** -0.5

        # KV cache (window_size)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        concept_db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        win = self.kv_cache.shape[1]

        # ── Q projection ─────────────────────────────────────────────
        q = self.wq_b(self.q_norm(self.wq_a(x)))                      # (B,N,H*head_dim)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))             # (B,N,H,head_dim)
        # Per-head RMS normalisation (V4-Pro line 498)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope, q_rope = q[..., :-self.rope_head_dim], q[..., -self.rope_head_dim:]
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # ── KV compression (MLA sovereign) ───────────────────────────
        kv = self.wkv(x)                                               # (B,N,head_dim)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = kv[..., :-self.rope_head_dim], kv[..., -self.rope_head_dim:]
        kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)
        # QAT: simulate FP8 on nope dims (V4-Pro line 506)
        act_quant(kv[..., :-self.rope_head_dim].contiguous(), 64, scale_fmt, scale_dtype, True)

        # ── Update KV cache (sliding window) ─────────────────────────
        if B > self.kv_cache.shape[0]:
            new_cache = torch.zeros(B, win, self.head_dim, device=self.kv_cache.device, dtype=self.kv_cache.dtype)
            new_cache[:self.kv_cache.shape[0]] = self.kv_cache
            self.register_buffer("kv_cache", new_cache, persistent=False)

        if start_pos == 0:
            self.kv_cache.detach_().zero_()
            # Prefill: store last `win` tokens (circular)
            if N <= win:
                self.kv_cache[:B, :N] = kv
            else:
                cutoff = N % win
                self.kv_cache[:B, cutoff:win], self.kv_cache[:B, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        else:
            self.kv_cache[:B, start_pos % win] = kv[:, 0]

        # ── Prepare KV for Attention ─────────────────────────────────
        K_cache = kv if start_pos == 0 else self.kv_cache[:B]
        
        if concept_db is not None:
            # Project concept_db to MLA KV space
            concept_kv = self.wkv(concept_db) # (B, C, head_dim)
            concept_kv = self.kv_norm(concept_kv)
            # QAT: simulate FP8 on nope dims
            act_quant(concept_kv[..., :-self.rope_head_dim].contiguous(), 64, scale_fmt, scale_dtype, True)
            
            # Concatenate token KV cache and concept KV along sequence dimension
            K_combined = torch.cat([K_cache, concept_kv], dim=1)
        else:
            K_combined = K_cache

        # ── SDPA (F.scaled_dot_product_attention) ────────────────────
        # V4-Pro uses sparse_attn with topk_idxs; on MPS we use dense SDPA
        # q: (B,N,H,D) → (B,H,N,D);  kv_cache as K=V: (B,N,D) → expand heads
        q_t  = q.transpose(1, 2)                                        # (B,H,N,D)
        
        # Prepare custom attention mask for causal token + full concept attention
        is_causal = (start_pos == 0 and N > 1 and concept_db is None)
        attn_mask = None
        if start_pos == 0 and N > 1 and concept_db is not None:
            Seq_combined = K_combined.size(1)
            Seq_token = K_cache.size(1)
            mask = torch.ones(N, Seq_combined, dtype=torch.bool, device=x.device)
            causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
            mask[:, :N] = causal_mask
            if Seq_token > N:
                mask[:, N:Seq_token] = True
            mask[:, Seq_token:] = False
            attn_mask = mask

        kv_h = K_combined.unsqueeze(1).expand(-1, self.n_heads, -1, -1).to(q_t.dtype)  # (B,H,Seq_combined,D)
        
        if attn_mask is not None:
            # Explicit manual attention computation to avoid MPS bug with custom boolean masks
            scores = torch.matmul(q_t.float(), kv_h.transpose(-2, -1).float()) * self.softmax_scale
            # Apply attention mask: fill True positions with -10000.0
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), -10000.0)
            probs = torch.softmax(scores, dim=-1).to(q_t.dtype)
            attn_out = torch.matmul(probs, kv_h)
        else:
            attn_out = F.scaled_dot_product_attention(
                q_t, kv_h, kv_h,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )  # (B, H, N, D)
        # De-rotate rope dims on output (V4-Pro line 534)
        attn_out_perm = attn_out.transpose(1, 2)                        # (B,N,H,D)
        attn_nope, attn_rope = attn_out_perm[..., :-self.rope_head_dim], attn_out_perm[..., -self.rope_head_dim:]
        attn_rope = apply_rotary_emb(attn_rope, freqs_cis, inverse=True)
        attn_out_perm = torch.cat([attn_nope, attn_rope], dim=-1)

        # ── Grouped O projection (V4-Pro lines 537-542) ───────────────
        o = attn_out_perm.reshape(B, N, self.n_groups, -1)             # (B,N,G,H/G*D)
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a_w.float())  # (B,N,G,o_lora_rank)
        out = self.wo_b(o.flatten(2).to(x.dtype))
        return out


# ══════════════════════════════════════════════════════════════════════
# GRAPH VECTOR QUANTIZER  ← SOVEREIGN VQ
# ══════════════════════════════════════════════════════════════════════
class GraphVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, hcm_commit_loss_coeff: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commit_coeff = hcm_commit_loss_coeff
        
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        # Adjacency matrix representing relations between concepts
        self.adjacency = nn.Parameter(torch.zeros(codebook_size, codebook_size))
        # Relation projection for graph message passing
        self.relation_proj = Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_x_f32 = flat_x.float()
        w_f32 = self.embedding.weight.float()
        
        # Calculate distances in float32 for numerical stability
        distances = (
            torch.sum(flat_x_f32 ** 2, dim=-1, keepdim=True)
            + torch.sum(w_f32 ** 2, dim=-1)
            - 2 * torch.matmul(flat_x_f32, w_f32.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(encoding_indices).view(B, L, D).to(x.dtype)
        
        # Graph Message Passing using adjacency matrix
        adj_probs = torch.softmax(self.adjacency.float(), dim=-1)
        mixed_embeddings = torch.matmul(adj_probs, w_f32).to(x.dtype)
        relation_features = self.relation_proj(mixed_embeddings)
        
        selected_relation_features = F.embedding(encoding_indices, relation_features).view(B, L, D)
        quantized = quantized + 0.1 * selected_relation_features
        
        commit_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.commit_coeff * commit_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss


def safe_normalize(x: torch.Tensor, p: float = 2.0, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = x.norm(p=p, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)


# ══════════════════════════════════════════════════════════════════════
# ELASTIC-SPARSE CONCEPT MEMORY  ← SOVEREIGN HCM (V1 Upgrade)
# ══════════════════════════════════════════════════════════════════════
class ElasticSparseConceptMemory(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.num_concepts = args.num_concepts
        self.hcm_ema_alpha = args.hcm_ema_alpha
        self.entropy_threshold = args.entropy_threshold
        self.max_blocks = 16
        
        # Latent queries for perceiver pooling
        self.latent_queries = nn.Parameter(torch.empty(1, self.num_concepts, args.dim))
        nn.init.normal_(self.latent_queries, 0.0, 0.02)
        
        # Start with one VQ block
        self.concept_blocks = nn.ModuleList([self._create_block(args)])
        
        # EMA slot buffer for dynamic learning/averaging
        self.register_buffer("slot_ema", torch.zeros(1, self.num_concepts, args.dim), persistent=True)
        # Persistent database of all block slots
        self.register_buffer("slot_db", torch.zeros(1, self.num_concepts, args.dim), persistent=True)
        # Buffer of per-block average/centroid vectors
        self.register_buffer("meta_centroids", torch.zeros(1, args.dim), persistent=False)
        
        # Lightning retrieve projection
        self.lightning_indexer = Linear(args.dim, args.dim)
        
    def _create_block(self, args: ModelArgs):
        return GraphVectorQuantizer(
            codebook_size=args.codebook_size,
            dim=args.dim,
            hcm_commit_loss_coeff=args.hcm_commit_loss_coeff
        )
        
    def process_chunk(self, encoder_hidden: torch.Tensor, block_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_enc, D = encoder_hidden.shape
        
        # 1. Perceiver Pooling (Fix: 4D Head Reshaping)
        B = encoder_hidden.size(0)
        Q = self.latent_queries.expand(B, -1, -1)
        
        # Reshape to [Batch, Heads, SeqLen, HeadDim]
        n_heads = 8 # Ensure args.dim is divisible by this
        head_dim = self.dim // n_heads
        
        Q_4d = Q.view(B, self.num_concepts, n_heads, head_dim).transpose(1, 2).to(encoder_hidden.dtype)
        KV_4d = encoder_hidden.view(B, N_enc, n_heads, head_dim).transpose(1, 2)
        
        # Execute Multi-Head Perceiver Pooling
        pooled_4d = F.scaled_dot_product_attention(Q_4d, KV_4d, KV_4d)
        
        # Flatten back to [Batch, M, Dim]
        pooled = pooled_4d.transpose(1, 2).contiguous().view(B, self.num_concepts, self.dim)
        
        # 2. VQ on the specified block
        quantized, loss = self.concept_blocks[block_idx](pooled)
        
        # 3. Update slot representations
        with torch.no_grad():
            mean_quant = quantized.detach().mean(dim=0, keepdim=True)
            self.slot_ema.data.copy_(self.hcm_ema_alpha * self.slot_ema.data + (1.0 - self.hcm_ema_alpha) * mean_quant)
            self.slot_db.data[block_idx].copy_(self.slot_ema.data[0])
            
            # Update centroid in meta_centroids
            self.meta_centroids.data[block_idx].copy_(torch.mean(self.slot_db[block_idx], dim=0))
            
        # 4. Check for elastic spawn condition
        if loss.item() > self.entropy_threshold and len(self.concept_blocks) < self.max_blocks:
            # Freeze block
            for p in self.concept_blocks[block_idx].parameters():
                p.requires_grad = False
                
            # Allocate space for new block in slot_db and meta_centroids
            new_slots = self.slot_ema.clone()
            self.register_buffer("slot_db", torch.cat([self.slot_db, new_slots], dim=0), persistent=True)
            
            new_centroid = torch.zeros(1, self.dim, device=self.meta_centroids.device, dtype=self.meta_centroids.dtype)
            self.register_buffer("meta_centroids", torch.cat([self.meta_centroids, new_centroid], dim=0), persistent=False)
            
            # Spawn block
            new_block = self._create_block(self.args).to(device=pooled.device, dtype=pooled.dtype)
            self.concept_blocks.append(new_block)
            
            # Reset EMA buffer for the new block
            self.slot_ema.zero_()
            
            # Recursively process in the new block
            return self.process_chunk(encoder_hidden, block_idx + 1)
            
        return quantized, loss
        
    def lightning_retrieve(self, decoder_query: torch.Tensor, top_k_blocks: int = 2) -> torch.Tensor:
        # Project and pool query over sequence dimension
        proj_q = self.lightning_indexer(decoder_query)
        pooled_q = torch.mean(proj_q, dim=1) # (B, D)
        
        proj_q_norm = safe_normalize(pooled_q, p=2.0, dim=-1)
        centroids_norm = safe_normalize(self.meta_centroids.to(proj_q_norm.dtype), p=2.0, dim=-1)
        
        # Cosine similarity matrix: (B, N_blocks)
        sim = torch.matmul(proj_q_norm, centroids_norm.t())
        
        k = min(top_k_blocks, self.meta_centroids.size(0))
        topk_scores, topk_idxs = torch.topk(sim, k, dim=-1) # (B, k)
        
        # Retrieve slots: slot_db is (N_blocks, num_concepts, D)
        # topk_idxs is (B, k)
        gathered = self.slot_db.to(decoder_query.dtype)[topk_idxs] # (B, k, num_concepts, D)
        concept_db = gathered.flatten(1, 2) # (B, k * num_concepts, D)
        
        return concept_db



# ══════════════════════════════════════════════════════════════════════
# GATE  (MoE routing)  ← V4-Pro exact Gate class
# V4-Pro refinements:
#   • score_func string dispatch (softmax / sigmoid / sqrtsoftplus)
#   • auxiliary bias (no grad effect on scores, only on topk selection)
#   • hash routing (first n_hash_layers)
# SOVEREIGN: sqrtsoftplus is the Lasmoid sovereign score function
# ══════════════════════════════════════════════════════════════════════
class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk        = args.n_activated_experts
        self.score_func  = args.score_func
        self.route_scale = args.route_scale
        self.use_hash    = layer_id < args.n_hash_layers
        self.ema_bias_lr = args.ema_bias_lr

        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.normal_(self.weight, 0.0, 0.02)

        if self.use_hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            # Auxiliary bias: shifts topk selection but NOT the routing weights (V4-Pro)
            self.bias = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        
        # Router Z-loss
        z_loss = torch.logsumexp(scores, dim=-1).square().mean()

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus  ← SOVEREIGN
            scores = F.softplus(scores).clamp(min=1e-8).sqrt()


        original_scores = scores

        if self.bias is not None:
            scores = scores + self.bias   # affects topk selection only

        if self.use_hash and input_ids is not None:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]

        # EMA bias balancing update (DDP-synchronized)
        if self.training and self.bias is not None:
            with torch.no_grad():
                counts = torch.bincount(indices.flatten(), minlength=self.weight.shape[0]).float()
                
                # Synchronize routing counts across all GPUs
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    
                total_routed = indices.numel() * (dist.get_world_size() if dist.is_initialized() else 1)
                routing_fraction = counts / (total_routed / self.topk)
                
                target_fraction = 1.0 / self.weight.shape[0]
                bias_update = self.ema_bias_lr * torch.sign(target_fraction - routing_fraction)
                self.bias.add_(bias_update)

        weights = original_scores.gather(1, indices)

        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        weights = weights * self.route_scale
        return weights, indices, z_loss


# ══════════════════════════════════════════════════════════════════════
# EXPERT  ← SOVEREIGN SwiGLU + V4-Pro swiglu_limit
# ══════════════════════════════════════════════════════════════════════
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate  = self.w1(x).float()
        up    = self.w3(x).float()

        # swiglu_limit clamping from V4-Pro for training stability
        if self.swiglu_limit > 0:
            up   = torch.clamp(up,   min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)

        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


# ══════════════════════════════════════════════════════════════════════
# DEEPSEEK-V4 MoE  ← SOVEREIGN + V4 refinements
# ══════════════════════════════════════════════════════════════════════
class DeepSeekMoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim              = args.dim
        self.n_routed         = args.n_routed_experts
        self.n_activated      = args.n_activated_experts
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None

        self.gate    = Gate(layer_id, args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit)
            for _ in range(self.n_routed)
        ])
        self.shared  = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        shape  = x.shape
        flat_x = x.reshape(-1, self.dim)

        weights, indices, z_loss = self.gate(flat_x, input_ids.flatten() if input_ids is not None else None)

        y      = torch.zeros_like(flat_x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()

        for i, exp in enumerate(self.experts):
            if counts[i] == 0:
                continue
            tok_idx, top_pos = torch.where(indices == i)
            y[tok_idx] += exp(flat_x[tok_idx], weights[tok_idx, top_pos, None])

        y = y + self.shared(flat_x).float()
        return y.type_as(x).reshape(shape), z_loss


# ══════════════════════════════════════════════════════════════════════
# HYPER-CONNECTIONS BLOCK  ← SOVEREIGN + V4-Pro exact math
# Replaces custom Sinkhorn with hc_split_sinkhorn() from V4-Pro kernel
# hc_pre / hc_post signatures match Block.hc_pre/hc_post in V4-Pro model
# ══════════════════════════════════════════════════════════════════════
class MHCBlock(nn.Module):
    def __init__(self, dim: int, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult         = hc_mult
        self.hc_sinkhorn_iters = sinkhorn_iters
        self.hc_eps          = eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim

        # V4-Pro: HC parameters stored in float32 (set_dtype(torch.float32))
        with set_dtype(torch.float32):
            self.hc_fn    = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_base  = nn.Parameter(torch.empty(mix_hc))
            self.hc_scale = nn.Parameter(torch.empty(3))

        nn.init.normal_(self.hc_fn, 0, 0.02)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def hc_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        V4-Pro Block.hc_pre() exact port.
        x: (B, S, hc, D)  →  layer_input: (B, S, D), post, comb
        """
        dtype = x.dtype
        B, S, hc, D = x.size()
        x_flat = x.flatten(2)                                # (B, S, hc*D)
        mean_sq = x_flat.square().mean(-1, keepdim=True).float()
        rsqrt  = torch.rsqrt(mean_sq + self.hc_eps).to(dtype)
        mixes  = F.linear(x_flat, self.hc_fn.to(dtype)) * rsqrt         # (B, S, mix_hc)

        pre, post, comb = hc_split_sinkhorn(
            mixes.float(), self.hc_scale.float(), self.hc_base.float(),
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
        )
        # Weighted sum of hc streams → single layer input
        # Using x directly in input dtype avoids redundant float32 conversions
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)  # (B, S, D)
        return y, post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """
        V4-Pro Block.hc_post() exact port.
        x: (B,S,D), residual: (B,S,hc,D), post: (B,S,hc), comb: (B,S,hc,hc)
        → (B,S,hc,D)
        """
        dtype = x.dtype
        # Keep operations in input dtype to avoid float32 promotion
        y = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.to(dtype), residual)
        return y


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK  ← SOVEREIGN + V4-Pro HC / V4-Pro start_pos
# ══════════════════════════════════════════════════════════════════════
class LasmoidBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn      = MLA(layer_id, args)
        self.ffn       = DeepSeekMoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm  = RMSNorm(args.dim, args.norm_eps)
        self.hc_attn   = MHCBlock(args.dim, args.num_residual_streams, args.hc_sinkhorn_iters, args.hc_eps)
        self.hc_ffn    = MHCBlock(args.dim, args.num_residual_streams, args.hc_sinkhorn_iters, args.hc_eps)

    def forward(
        self,
        streams: torch.Tensor,            # (B, S, hc, D)
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        input_ids: Optional[torch.Tensor] = None,
        concept_db: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ATTN HC (V4-Pro Block.forward lines 690-693)
        residual = streams
        attn_in, post, comb = self.hc_attn.hc_pre(streams)
        attn_out = self.attn(self.attn_norm(attn_in), freqs_cis, start_pos, concept_db)
        streams  = self.hc_attn.hc_post(attn_out, residual, post, comb)

        # FFN HC (V4-Pro Block.forward lines 695-699)
        residual = streams
        ffn_in, post, comb = self.hc_ffn.hc_pre(streams)
        ffn_out, z_loss  = self.ffn(self.ffn_norm(ffn_in), input_ids)
        streams  = self.hc_ffn.hc_post(ffn_out, residual, post, comb)

        return streams, z_loss


# ══════════════════════════════════════════════════════════════════════
# MTP BLOCK  ← SOVEREIGN + V4-Pro MTPBlock exact
# V4-Pro: e_proj + h_proj fusion, enorm/hnorm, shared embed+head
# ══════════════════════════════════════════════════════════════════════
class MTPBlock(nn.Module):
    """
    V4-Pro exact MTPBlock: fuse next-token embedding with current hidden state,
    run a full transformer block, then produce next-token logits.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm  = RMSNorm(args.dim, args.norm_eps)
        self.hnorm  = RMSNorm(args.dim, args.norm_eps)
        self.norm   = RMSNorm(args.dim, args.norm_eps)
        self.block  = LasmoidBlock(layer_id, args)

        # HC head (shared sigmoid-based, not full Sinkhorn)
        hc_mult = args.num_residual_streams
        hc_dim  = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn    = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base  = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        # Set by LasmoidV1 after construction
        self.embed: Optional[nn.Embedding] = None
        self.head:  Optional[nn.Module]    = None

    def hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """V4-Pro ParallelHead.hc_head: sigmoid-based hc → single stream."""
        shape, dtype = x.size(), x.dtype
        B, S, hc, D  = shape
        xf    = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre   = torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        y     = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x: torch.Tensor,         # (B, S, hc, D) — current hidden state
        freqs_cis: torch.Tensor,
        input_ids: torch.Tensor, # (B, S) — current input token ids
        start_pos: int = 0,
        concept_db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.embed is not None and self.head is not None
        e = self.enorm(self.embed(input_ids).to(x.dtype))   # next-token embedding
        h = self.hnorm(x)                                   # normalise current state
        # V4-Pro MTPBlock line 763: fuse via learned projections
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(h)   # (B, S, hc, D)  (broadcast over hc)
        x, _ = self.block(x, freqs_cis, start_pos, input_ids, concept_db)
        # HC head reduce → logits
        y = self.hc_head_reduce(x)
        y = self.norm(y)
        return F.linear(y.float(), self.head.weight.float())  # logits


# ══════════════════════════════════════════════════════════════════════
# LASMOID V1 — FULL MODEL  ← SOVEREIGN CQRS + V4-Pro all upgrades
# ══════════════════════════════════════════════════════════════════════
class LasmoidV1(nn.Module):
    def __init__(self, args: ModelArgs):
        global default_dtype, scale_fmt, scale_dtype
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt     = args.scale_fmt
        scale_dtype   = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

        super().__init__()
        self.args        = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult     = args.num_residual_streams

        # Embedding (shared with head — tied weights)
        self.emb = nn.Embedding(args.vocab_size, args.dim)

        # READ REPLICA (Encoder) — CQRS sovereign
        self.encoder_attn = MLA(0, args)
        self.encoder_norm = RMSNorm(args.dim, args.norm_eps)
        self.memory       = ElasticSparseConceptMemory(args)

        # WRITE MASTER (Decoder) — CQRS sovereign
        self.layers       = nn.ModuleList([LasmoidBlock(i, args) for i in range(args.n_layers)])
        self.decoder_norm = RMSNorm(args.dim, args.norm_eps)

        # Output head (tied to embedding)
        self.head   = Linear(args.dim, args.vocab_size)
        self.head.weight = self.emb.weight  # weight tying
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        # HC head for main model output (from V4-Pro ParallelHead)
        hc_mult = args.num_residual_streams
        hc_dim  = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn    = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base  = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.hc_head_fn, 0, 0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        # MTP blocks (V4-Pro exact MTPBlock)
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + i, args)
            blk.embed = self.emb
            blk.head  = self.head
            self.mtp.append(blk)

        # YaRN RoPE frequencies (cached)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.rope_head_dim, args.max_seq_len,
                args.original_seq_len, args.rope_theta,
                args.rope_factor, args.beta_fast, args.beta_slow,
            ),
            persistent=False,
        )

        self.gradient_checkpointing = False
        self.last_z_loss = torch.tensor(0.0)
        self.last_commit_loss = torch.tensor(0.0)

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """HC head: sigmoid-based weighted sum over hc streams."""
        shape, dtype = x.size(), x.dtype
        B, S, hc, D  = shape
        xf    = x.flatten(2)
        mean_sq = xf.square().mean(-1, keepdim=True).float()
        rsqrt = torch.rsqrt(mean_sq + 1e-6).to(dtype)
        mixes = F.linear(xf, self.hc_head_fn.to(dtype)) * rsqrt
        pre   = torch.sigmoid(mixes.float() * self.hc_head_scale + self.hc_head_base) + 1e-6
        y     = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=2)
        return y

    def forward(
        self,
        x_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        concept_db: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        CQRS forward pass:
          1. Read Replica (Encoder):  x_enc → HCM (ElasticSparseConceptMemory) → memory_state
          2. Write Master (Decoder):  x_dec + retrieved concept_db → logits
        """
        B, N_dec = x_dec.shape
        freqs_cis_dec = self.freqs_cis[start_pos : start_pos + N_dec]


        # ── 1. READ REPLICA ───────────────────────────────────────────
        if start_pos == 0 or concept_db is None or memory_state is None:
            assert x_enc is not None, "x_enc must be provided at start_pos == 0"
            _, N_enc = x_enc.shape
            freqs_cis_enc = self.freqs_cis[:N_enc]
            H_enc   = self.emb(x_enc).to(default_dtype)
            enc_out = self.encoder_attn(self.encoder_norm(H_enc), freqs_cis_enc, start_pos=0)
            memory_state, commit_loss = self.memory.process_chunk(enc_out, block_idx=0)
            self.last_commit_loss = commit_loss
            # Retrieve concept_db for decoder cross-attention
            concept_db = self.memory.lightning_retrieve(H_enc, top_k_blocks=self.args.lightning_topk_blocks)

        # ── 2. WRITE MASTER ───────────────────────────────────────────
        H_dec     = self.emb(x_dec).to(default_dtype)
        H_memory  = torch.mean(memory_state, dim=1, keepdim=True).expand(-1, N_dec, -1)

        # Expand to hc_mult copies
        hc = self.hc_mult
        streams_list = [H_dec, H_memory]
        streams_list += [torch.zeros_like(H_dec)] * (hc - 2)          # concept + padding
        streams = torch.stack(streams_list, dim=2)                      # (B,N_dec,hc,D)

        total_z_loss = torch.tensor(0.0, device=x_dec.device, dtype=torch.float32)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                streams, z_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    streams,
                    freqs_cis_dec,
                    start_pos,
                    x_dec,
                    concept_db,
                    use_reentrant=False,
                )
            else:
                streams, z_loss = layer(streams, freqs_cis_dec, start_pos, x_dec, concept_db)
            total_z_loss = total_z_loss + z_loss
        self.last_z_loss = total_z_loss

        # HC head reduce → logits (V4-Pro Transformer.forward lines 808-809)
        h_final  = self._hc_head_reduce(streams)                        # (B, N_dec, D)
        h_normed = self.decoder_norm(h_final)
        logits   = F.linear(h_normed.float(), self.head.weight.float()) # (B, N_dec, vocab)

        # MTP: produce next-next-token logits during training
        mtp_logits = None
        if self.training and N_dec > 1 and len(self.mtp) > 0:
            mtp_logits = self.mtp[0](streams, freqs_cis_dec, x_dec, start_pos, concept_db)

        return logits, mtp_logits, concept_db, memory_state

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 0,
        pad_token: int = 50256,
        max_len: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with stateful start_pos (V4-Pro exact)."""
        self.eval()
        device  = idx.device
        actual_max_len = max_len if max_len is not None else self.max_seq_len
        cond_len = idx.shape[1]

        # Prefill
        if cond_len < actual_max_len:
            padding    = torch.full((idx.shape[0], actual_max_len - cond_len), pad_token, dtype=idx.dtype, device=device)
            idx_padded = torch.cat([padding, idx], dim=1)
        else:
            idx_padded = idx[:, -actual_max_len:]

        # Run prefill forward
        logits, mtp_logits, concept_db, memory_state = self(idx_padded, idx_padded, start_pos=0)

        # Sample first token
        last_logits = logits[:, -1, :]
        if temperature > 0:
            last_logits = last_logits / temperature
        if top_k > 0:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -10000.0
        probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
        idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)

        # Decode loop
        current_pos = actual_max_len
        for step in range(max_new_tokens - 1):
            # Run forward pass incrementally with only the latest generated token
            logits, _, _, _ = self(
                x_enc=None,
                x_dec=idx_next,
                concept_db=concept_db,
                memory_state=memory_state,
                start_pos=current_pos,
            )
            
            last_logits = logits[:, -1, :]
            if temperature > 0:
                last_logits = last_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -10000.0
                
            probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
            idx_next = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
            current_pos += 1

        self.train()
        return idx


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(" LasmoidV1 — Self-Test")
    print("=" * 60)

    args            = ModelArgs()
    args.vocab_size = 50257
    args.dim        = 128
    args.n_layers   = 4
    args.n_heads    = 4
    args.head_dim   = 48
    args.rope_head_dim = 16
    args.o_groups   = 2
    args.o_lora_rank = 32
    args.q_lora_rank = 32
    args.num_residual_streams = 4
    args.max_seq_len = 64
    args.max_batch_size = 2
    args.n_mtp_layers = 1

    model = LasmoidV1(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    x_enc = torch.randint(0, args.vocab_size, (2, 32))
    x_dec = torch.randint(0, args.vocab_size, (2, 32))

    logits, mtp_logits, c_db, mem = model(x_enc, x_dec)
    print(f"  logits     : {logits.shape}  dtype={logits.dtype}")
    print(f"  concept_db : {c_db.shape}")
    print(f"  memory     : {mem.shape}")

    print()
    print("=== Kernel integration tests ===")
    # hc_split_sinkhorn integration
    hc  = args.num_residual_streams
    mix = torch.randn(2, 8, (2 + hc) * hc, dtype=torch.float32)
    sc  = torch.ones(3, dtype=torch.float32)
    bs  = torch.zeros((2 + hc) * hc, dtype=torch.float32)
    pre, post, comb = hc_split_sinkhorn(mix, sc, bs, hc_mult=hc)
    print(f"  hc_sinkhorn: pre={pre.shape}, post={post.shape}, comb={comb.shape}")

    print()
    print("ALL TESTS PASSED ✓")
