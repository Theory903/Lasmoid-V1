# Lasmoid-V1: Neuro-Symbolic MoE Model with Manifold-Constrained Hyper-Connections

<div align="center">
  <h2>Lasmoid-V1</h2>
  <p>A mathematically optimized Stage-9 Master Architecture implementing advanced Multi-Head Latent Attention (MLA), DeepSeek-V4-style Shared/Routed Mixture-of-Experts (MoE), 3-stream Manifold-Constrained Hyper-Connections (mHC), Multi-Scale Hierarchical Concept Memory, and Multi-Token Prediction (MTP).</p>
</div>

<hr>
     
## Table of Contents
1. [Introduction](#1-introduction)
2. [Architectural Highlights](#2-architectural-highlights)
3. [Repository Structure](#3-repository-structure)
4. [Installation & Requirements](#4-installation--requirements)
5. [Quick Start (Inference CLI)](#5-quick-start-inference-cli)
6. [Training & Associative Recall Benchmarking](#6-training--associative-recall-benchmarking)
7. [License](#7-license)

---

## 1. Introduction

**Lasmoid-V1** is a state-of-the-art neuro-symbolic language model architecture. By separating token processing into a **Read Replica** (which compiles multi-scale concepts into a hierarchical memory database) and a **Write Master** (which decodes outputs using parallel residual streams), Lasmoid-V1 achieves extremely stable learning and flat $O(1)$ memory footprints during inference.

---

## 2. Architectural Highlights

- **Multi-Head Latent Attention (MLA)**: Compresses the Key-Value (KV) cache dimension, reducing memory footprint during long-sequence generation.
- **DeepSeek-V4 style MoE with Sqrt-Softplus Affinity Gating**: Separates computational capacity into shared (always active) and routed experts, using an auxiliary-loss-free routing strategy based on sqrt-softplus gate activations.
- **Manifold-Constrained Hyper-Connections (mHC)**: Distributes training gradients across three parallel residual streams (Hyper, Memory, and Concept) mapped dynamically via Sinkhorn log projections.
- **Hierarchical Concept Memory**: Automatically abstracts sequence information into three conceptual levels (Local, Abstract, and Global) bound using a gated update mechanism.
- **Multi-Token Prediction (MTP)**: Trains the model to predict both next token ($t+1$) and next-next token ($t+2$) in parallel, accelerating training representation learning.

---

## 3. Repository Structure

This repository is organized to mirror the official **DeepSeek-V3** repository structure for maximum developer familiarity:

```
lasmoid_v1/
├── LICENSE
├── README.md
├── requirements.txt          # Root dependencies
├── train.py                  # Tiny Shakespeare training loop with Muon + AdamW
├── retrieval_test.py         # Associative recall concept test (O(1) memory check)
└── inference/
    ├── configs/
    │   └── config.json       # Hyperparameters for LasmoidV1
    ├── convert.py            # Weight checkpoint formatter
    ├── fp8_cast_bf16.py      # simulated FP8-to-BF16 casting utility
    ├── generate.py           # Generation CLI supporting interactive and batch modes
    ├── kernel.py             # Triton kernels with local PyTorch fallbacks for Mac/MPS
    ├── model.py              # The core PyTorch LasmoidV1 implementation
    └── requirements.txt      # Inference specific dependencies
```

---

## 4. Installation & Requirements

Ensure you have Python 3.10+ and a PyTorch environment (works on macOS MPS, Linux CUDA, or CPU).

Install dependencies from the root directory:
```bash
pip install -r requirements.txt
```

---

## 5. Quick Start (Inference CLI)

We provide a lightweight inference CLI (`generate.py`) that matches the API of the `DeepSeek-V3` inference scripts:

### Interactive Chat Mode
Run an interactive session where prompt contexts are processed once, compiling concept states in $O(1)$ memory:
```bash
python3 inference/generate.py --ckpt-path checkpoints/ --config inference/configs/config.json --interactive
```

### Batch Mode
Process a set of prompts in batch:
```bash
python3 inference/generate.py --ckpt-path checkpoints/ --config inference/configs/config.json --input-file prompts.txt
```

---

## 6. Training & Associative Recall Benchmarking

### Model Pre-Training
Pre-train the model on the Tiny Shakespeare dataset. The script automatically shards parameters to update 2D weight matrices via the **Muon** optimizer and 1D/biases via **AdamW**:
```bash
python3 train.py --max_iters 1000 --batch_size 8 --learning_rate 6e-4
```

### Synthetic Needle-In-A-Haystack (NIAH) Test
Verify the zero-shot associative recall capabilities of the Hierarchical Memory database:
```bash
python3 retrieval_test.py
```

---

## 7. License

The repository code is licensed under the MIT License.
