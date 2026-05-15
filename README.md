# GPT
A from-scratch LLM implementation based on the Transformer architecture. Optimized using PyTorch MPS, featuring a custom BPE tokenizer.

**A professional-grade, from-scratch implementation of a GPT-style Large Language Model. Designed for high-performance training across diverse hardware, from Apple Silicon to NVIDIA environments.**

---

## 🧠 Project Vision

The goal of this project is to provide a clean, optimized, and transparent source code for training Large Language Models locally. By implementing the core architecture used by industry-leading cloud AIs, this engine demonstrates how to handle massive datasets and complex neural computations without relying on heavy high-level frameworks.

## 🛠️ Technical Excellence

### 1. Universal Hardware Acceleration
The codebase is engineered for portablity. Using a dynamic device-agnostic backend, it automatically scales across:
* **NVIDIA GPUs:** Full CUDA integration for professional workstation training.
* **Apple Silicon:** Optimized for the **Metal Performance Shaders (MPS)** backend, utilizing unified memory on M chips.

### 2. Industry-Standard Architecture
This isn't a "toy" model. It implements the same foundational blocks as GPT-4 and Llama:
* **Causal Self-Attention:** Strict masking to ensure autoregressive consistency.
* **Residual Learning:** Optimized gradient flow through deep stacks of Transformer blocks.
* **BPE Tokenization:** A custom-built Byte Pair Encoding tokenizer developed in **Rust**, ensuring zero-bottleneck pre-processing of raw text.

### 3. Engineering for Scale
* **Memory-Mapped Data:** Implements `numpy.memmap` to stream 40GB+ datasets directly from disk, allowing for massive-scale training on consumer hardware.
* **Optimized Training Loop:** Low-level control over backpropagation and weight updates via AdamW, with built-in evaluation metrics for validation monitoring.

## 🏗️ Architecture Breakdown

| Component | Implementation Detail |
| :--- | :--- |
| **Logic** | Decoder-only Transformer (Multi-Head Attention) |
| **Pre-processing** | High-speed Rust-based BPE (Pre-compiled) |
| **Data Handling** | Binary sharding with uint16 token mapping |
| **Backend** | PyTorch (Auto-detects CUDA/MPS/CPU) |

## 🚀 Getting Started

This repository provides the **complete source code engine**. It is designed for users who want to train their own models on their own hardware. 

1. **Pre-tokenize:** Use the Rust-backed `prepare_data.py` to convert your raw corpus into binary shards.
2. **Configure:** Adjust `config.py` to define the width and depth of your model based on your VRAM.
3. **Train:** Run `train.py` to start the learning process. The engine supports `.safetensors` for secure and fast model saving.

---
*Developed as a deep-dive into LLM internals, focusing on code efficiency, mathematical accuracy, and cross-platform hardware optimization.*