# 🧠 LLM — Modern Autoregressive Pipeline

A modular, production-grade Language Model training pipeline written from scratch in **Rust** and **PyTorch**.

This repository provides an end-to-end framework to build, train, align, and serve modern causal Transformers on consumer workstations, portable devices, or enterprise GPU clusters.

---

## 📌 Executive Overview

Most open-source educational LLM implementations are frozen in 2019: they replicate the original GPT-2 architecture using standard LayerNorm, static learned positional embeddings, slow Python-bound tokenizers, and unmasked next-token objectives.

This project delivers a **fully modernized architecture** matching the architectural foundations of state-of-the-art models (such as Llama 3, Mistral, and Gemma):
- **Core Engine:** Pre-training, instruction fine-tuning, and streaming inference implemented in pure PyTorch with dynamic device dispatch (`CUDA`, `MPS`, `CPU`).
- **Data Engineering:** High-performance Byte Pair Encoding (BPE) tokenizer and zero-copy binary dataset packing engineered in **Rust** using multi-threaded work-stealing parallelism (`rayon`).
- **Hardware-Aware Design:** Linear memory scaling via decoupled physical micro-batching and gradient accumulation, enabling deep pre-training runs within strict memory envelopes without running out of memory.

---

## 🔬 In-Depth Technical Architecture

This codebase is built to scale: the same architecture and data pipeline can power small experimental models on a laptop or scale up to multi-billion parameter foundations by adjusting configuration hyperparameters.

```text
                  Raw Text Corpus (~40 GB)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 1. BPE Tokenizer  │ ──► Multi-threaded Rust (`rayon`)
                   │    Engine (Rust)  │     32k vocabulary extraction
                   └─────────┬─────────┘
                             │
                             ▼
                mon_vocabulaire_bpe.json
                             │
                             ▼
                   ┌───────────────────┐
                   │ 2. Pre-Tokenize & │ ──► Parallel memory mapping
                   │    Binary Packing │     Contiguous uint16 stream (`train.bin`)
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ 3. Modern LLM     │ ──► RMSNorm + RoPE + SwiGLU + Weight Tying
                   │    Pre-Training   │     Cosine LR Schedule + Gradient Clipping
                   └─────────┬─────────┘     Micro-batching + Gradient Accumulation
                             │
                             ▼
                     model.safetensors (Base Foundation)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 4. Masked SFT     │ ──► Alpaca instruction dataset
                   │    Fine-Tuning    │     Strict prompt masking (`target = -100`)
                   └─────────┬─────────┘
                             │
                             ▼
                 model_instruct.safetensors (Aligned Assistant)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 5. Real-Time CLI  │ ──► Autoregressive generation
                   │    Streaming      │     Repetition penalty & dynamic sampling
                   └───────────────────┘

```

### 1. High-Throughput Rust Tokenization

Tokenizing tens of gigabytes of text in Python typically hits the Global Interpreter Lock (GIL) and incurs heavy serialization penalties.

* The tokenization subsystem (`src/gpt01_tokenizer/bpe` and `src/gpt02_data_prep`) is written entirely in native Rust.
* Utilizes `rayon` for data-parallel pair counting across all available CPU performance cores.
* Generates a fixed 32,768-token vocabulary saved in compact JSON format.
* Packs the training split directly into a contiguous binary sequence of unsigned 16-bit integers (`uint16`), enabling instant zero-copy OS memory mapping (`np.memmap`) during training without RAM bloat.

### 2. Modern Transformer Upgrades

The neural architecture (`src/gpt03_pretraining/gpt.py`) integrates four foundational improvements:

* **RMSNorm (Root Mean Square Normalization):** Replaces standard LayerNorm across all pre-attention and pre-FFN positions. RMSNorm scales activations by their root mean square rather than subtracting the mean, reducing computational overhead by ~10–15% per backward pass while maintaining numerical stability.
* **RoPE (Rotary Position Embeddings):** Discards static absolute positional tables in favor of complex-valued rotation matrices applied directly to Query ($Q$) and Key ($K$) projections. This injects relative positional awareness naturally decaying over distance, allowing better sequence extrapolation past the default training window.
* **SwiGLU Activation:** Replaces standard GELU feed-forward networks with a gated linear unit:

$$\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{gate}})) \otimes (x W_{\text{up}}) W_{\text{down}}$$



This configuration significantly improves gradient flow and representational capacity per parameter compared to traditional two-layer MLPs.
* **Weight Tying:** The input embedding matrix and the final linear projection layer share identical weights ($W_{\text{embed}} = W_{\text{head}}^T$). This eliminates ~25–30% of total model parameters (saving hundreds of megabytes of VRAM) with zero degradation in perplexity.

### 3. Training Dynamics & Memory Engineering

* **Micro-Batch Gradient Accumulation:** Rather than requiring massive physical batches to stabilize training, the engine decouples forward execution (`BATCH_SIZE = 1`) from weight updates (`GRAD_ACCUM_STEPS = 64–72`), achieving a virtual batch size of 65k–75k tokens while keeping physical VRAM usage **under 3 GB**.
* **Cosine Learning Rate Decay:** Incorporates a dynamic warmup-and-decay schedule transitioning from $1 \times 10^{-4}$ down to $1 \times 10^{-5}$, preventing loss stagnation on complex cross-entropy plateaus.
* **Stateful Interrupt Recovery:** Intercepts `SIGINT` (`Ctrl+C`) gracefully, dumping execution counters, step indices, and optimizer states to `train_meta.json` alongside `.safetensors` checkpoints for seamless resumption.

### 4. Masked Supervised Fine-Tuning (SFT)

Instruction alignment (`src/gpt04_finetuning/finetune.py`) formats raw prompts into structured conversational pairs:

```text
### Instruction:
{user_query}

### Response:
{assistant_completion}<|endoftext|>

```

To prevent catastrophic forgetting, all instruction tokens are mapped to target index `-100`. PyTorch's `CrossEntropyLoss` ignores these indices, computing gradients **strictly on the assistant's output and completion tokens**.

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── raw/                       # Raw text corpus & alpaca_data.json
│   ├── vocab/                     # Serialized BPE merges & 32k vocabulary
│   ├── processed/                 # Memory-mapped binary token dataset (train.bin)
│   └── trainedmodel/              # Checkpoints & execution metadata
│       ├── model.safetensors          # Base pre-trained weights
│       ├── model_instruct.safetensors # SFT aligned weights
│       └── train_meta.json            # Step counter & resumption state
│
├── src/
│   ├── gpt01_tokenizer/
│   │   └── bpe/                   # Multi-threaded Rust BPE training crate
│   ├── gpt02_data_prep/
│   │   └── src/                   # Rust high-speed dataset binarizer
│   ├── gpt03_pretraining/
│   │   ├── config.py              # Hyperparameters, scaling ratios & devices
│   │   ├── gpt.py                 # Modern Transformer (RMSNorm, RoPE, SwiGLU)
│   │   └── train.py               # Pre-training loop with Cosine scheduler
│   ├── gpt04_finetuning/
│   │   └── finetune.py            # Alpaca SFT with target mask (-100)
│   └── gpt05_inference/
│       └── generate.py            # Terminal CLI with streaming & repetition penalty
│
├── .gitignore
└── README.md

```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

* **Python 3.10+**
* **Rust & Cargo** (for native tokenization binaries)
* **PyTorch 2.2+** with target backend support (NVIDIA CUDA, Apple Silicon MPS, or CPU)

Verify your environment:

```bash
python3 --version
cargo --version
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | MPS: {torch.backends.mps.is_available()}')"

```

### 2. Environment Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy torch torchvision tqdm safetensors

```

---

## 🎛️ Hardware Sizing & Scaling Guide

Before launching training, open `src/gpt03_pretraining/config.py` to match your physical hardware:

| Target Platform | Available Memory | Recommended `BATCH_SIZE` | Recommended `GRAD_ACCUM_STEPS` | Total VRAM Footprint |
| --- | --- | --- | --- | --- |
| **Consumer Laptop / Mac / CPU** | 8 – 16 GB | `1` | `64 – 72` | **~2.5 – 3.0 GB** |
| **Mid-Range GPU (RTX 4070 / 5070)** | 12 – 16 GB | `4` | `16` | **~6.0 – 8.0 GB** |
| **Data Center GPU (A100 / H100)** | 40 – 80 GB | `16 – 32` | `2 – 4` | **~24 – 40 GB** |

The codebase automatically routes computations:

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

```

---

## 🚀 Step-by-Step Execution

### Step 1: Extract BPE Vocabulary (Rust)

Compile and execute the native tokenizer on your raw text corpus:

```bash
cd src/gpt01_tokenizer/bpe
cargo run --release
cd ../../..

```

*Artifact generated: `data/vocab/mon_vocabulaire_bpe.json*`

### Step 2: Binarize the Corpus (Rust)

Encode your raw text files into a contiguous binary `uint16` buffer:

```bash
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

*Artifact generated: `data/processed/train.bin*`

### Step 3: Run Baseline Pre-Training

Train the base autoregressive model on the binary token stream:

```bash
python -m src.gpt03_pretraining.train

```

* Real-time logging displays step count, learning rate, training loss, and validation loss.
* Safely pause training at any time with `Ctrl+C`. Re-running the command automatically detects `train_meta.json` and resumes from the exact iteration.
*Artifact generated: `data/trainedmodel/model.safetensors*`

### Step 4: Supervised Instruction Fine-Tuning (SFT)

Align the base model to follow commands using the Alpaca instruction format:

```bash
python -m src.gpt04_finetuning.finetune

```

* Automatically tracks validation loss across evaluation checkpoints.
* Early stopping halts execution when validation loss converges, preserving optimal weights.
*Artifact generated: `data/trainedmodel/model_instruct.safetensors*`

### Step 5: Interactive Terminal Inference

Launch the interactive command-line interface with real-time character streaming:

```bash
python -m src.gpt05_inference.generate

```

```text
=======================================================
Alpaca instruction assistant (type 'exit' to quit)
=======================================================

Your instruction: List 3 advantages of renewable energy.

Response: Finding electricity. 
A new solar system is a system that can be used to generate electricity 
resources and the electricity needed by reducing the costs of its efficient speed...
-------------------------------------------------------

```

---

## 📊 Reference Benchmarks & Training Results

Baseline metrics recorded on a standard ~125M parameter configuration (`block_size = 1024`, `n_embd = 768`, `n_layer = 12`, `n_head = 12`, `vocab_size = 32768`):

| Stage | Dataset / Scope | Iterations / Steps | Convergence Metric | Peak Memory Footprint |
| --- | --- | --- | --- | --- |
| **BPE Tokenization** | ~40 GB Raw Corpus | Multi-threaded pass | 32,768 merges | CPU Multi-core bound |
| **Pre-Training** | ~600M Tokens | 8,598 steps | **Validation Loss: 3.9556** | **< 3.0 GB** |
| **Instruction SFT** | Alpaca (52k examples) | 1,200 steps (Early stop) | **Validation Loss: 3.3217** | **< 2.5 GB** |

---

## 💡 Future Scaling & Modern Extensions

Because this codebase follows modular modern Transformer conventions, it serves as an open foundation for further cutting-edge research:

* **Grouped-Query Attention (GQA):** Reduce KV cache memory pressure during inference by sharing key/value heads across query groups.
* **Mixture of Experts (MoE):** Replace the single SwiGLU feed-forward layer with sparsely gated top-k expert routing to scale parameter count without increasing FLOPs per token.
* **Direct Preference Optimization (DPO):** Stack preference alignment directly on top of the SFT checkpoint using pair-wise reward margins.

```
