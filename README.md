# 🧠 LLM — Modular End-to-End Pipeline

A modular, hardware-agnostic Language Model pipeline (~125M params) written from scratch in **Rust** and **PyTorch**.

Engineered to scale and dynamically adapt across any target hardware—from high-end **NVIDIA GPUs (CUDA)** to **Apple Silicon (MPS)** and **standard CPUs**—without changing the core codebase.

The architecture integrates modern industry standards (**RoPE, SwiGLU, RMSNorm, Weight Tying**) and offloads data preprocessing to a **multi-threaded Rust engine**.

---

## ⚡ Architectural Core

* **Multi-Threaded BPE Engine (Rust):** Built with `rayon` for zero-GIL, core-saturating tokenization and binary packing (`uint16`).
* **Modern Transformer Blocks:**
* **RMSNorm** replacing LayerNorm (faster backprop, no mean-centering).
* **SwiGLU** activation replacing standard GELU MLP.
* **RoPE (Rotary Position Embeddings)** for relative token attention.
* **Weight Tying** between input embeddings and the output projection head.


* **Hardware-Aware Execution:**
* Dynamic device auto-detection (`cuda` $\to$ `mps` $\to$ `cpu`).
* Memory-safe scaling using micro-batching (`BATCH_SIZE`) combined with gradient accumulation (`GRAD_ACCUM_STEPS`) to fit any VRAM/RAM constraint.
* **Cosine Learning Rate Decay** with interrupt-safe, resumable checkpointing (`train_meta.json`).


* **Supervised Fine-Tuning (SFT):** Alpaca instruction format using masked cross-entropy (`-100`) strictly evaluating assistant outputs.

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── vocab/          # BPE 32k merges (mon_vocabulaire_bpe.json)
│   ├── processed/      # Memory-mapped uint16 binary buffer (train.bin)
│   └── trainedmodel/   # Weights (model.safetensors, model_instruct.safetensors)
│
├── src/
│   ├── gpt01_tokenizer/ # Multi-threaded Rust BPE tokenizer
│   ├── gpt02_data_prep/ # Binary corpus binarization in Rust
│   ├── gpt03_pretraining/ # PyTorch model (RoPE, SwiGLU, RMSNorm) & training loop
│   ├── gpt04_finetuning/  # Alpaca SFT with -100 target masking
│   └── gpt05_inference/   # Interactive CLI with streaming & repetition penalty
│
└── README.md

```

---

## ⚙️ Hardware Sizing (`src/gpt03_pretraining/config.py`)

Adjust two variables to match your system's memory envelope:

| Target Hardware | VRAM / RAM | `BATCH_SIZE` | `GRAD_ACCUM_STEPS` | Total Batch Footprint |
| --- | --- | --- | --- | --- |
| **Edge / CPU / Laptop** | 8 – 16 GB | `1` | `64 – 72` | ~2 – 3 GB |
| **Consumer GPU (RTX 4070/5070)** | 12 – 16 GB | `4 – 8` | `8 – 16` | ~6 – 10 GB |
| **Enterprise GPU (A100 / H100)** | 40 – 80 GB | `16 – 32` | `2 – 4` | ~24 – 40 GB |

---

## 🚀 Execution Pipeline

### 1. Build Vocabulary & Binary Buffer (Rust)

```bash
# Generate 32k BPE merge rules
cd src/gpt01_tokenizer/bpe && cargo run --release && cd ../../..

# Pack text corpus into memory-mapped uint16 binary
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

### 2. Pre-Training

```bash
# Train base autoregressive model
python -m src.gpt03_pretraining.train

```

*Auto-resumes from last saved step via `train_meta.json` if interrupted (`Ctrl+C`).*

### 3. Supervised Fine-Tuning (Alpaca)

```bash
# Align instruction behavior with masked target loss
python -m src.gpt04_finetuning.finetune

```

### 4. Interactive Terminal Inference

```bash
# Launch interactive streaming CLI
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

```

---

## 📊 Baseline Reference Metrics

*Evaluated on ~125M parameter model (`block_size = 1024`, `n_embd = 768`, `n_layer = 12`, `n_head = 12`):*

* **Pre-Training Loss:** Converged to **`3.9556`** (~8,600 iterations on ~40 GB raw text).
* **Instruction Fine-Tuning Loss:** Converged to **`3.3217`** (early stopping on Alpaca dataset).
* **Memory Footprint:** Fully contained within **< 3 GB RAM/VRAM** using micro-accumulation.