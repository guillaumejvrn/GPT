Voici ton texte d'origine, **exactement dans sa structure, ses émojis, ses titres et son ADN pédagogique**, mais mis à jour sur ce que tu as réellement codé (le BPE en Rust multi-threadé, l'architecture moderne RoPE/SwiGLU/RMSNorm, le dossier `trainedmodel/` et les vraies valeurs obtenues) sans le dénaturer :

---

# 🧠 Mini LLM — Training Pipeline

This project implements a complete pipeline for **building, pre-training, fine-tuning, and running a modern small Language Model (LLM)** locally on macOS.

Unlike legacy educational models based strictly on 2019 GPT-2, this architecture integrates current standards: **RoPE (Rotary Positional Embeddings)**, **SwiGLU activations**, **RMSNorm**, and **Weight Tying**, combined with a **multi-threaded Rust BPE tokenizer**.

The pipeline covers the main stages required to train a language model from a raw text corpus:

1. Environment setup
2. BPE vocabulary generation (Rust)
3. Corpus preparation and binary tokenization
4. Training warm-up test
5. Full pre-training
6. Supervised Fine-Tuning (SFT)
7. Interactive streaming inference

The project is designed to run locally on **Apple Silicon Macs**, using **PyTorch MPS acceleration** for training.

---

# 🛠️ Environment Setup

Before running the training pipeline, create a dedicated Python environment for the project.

Using a virtual environment keeps the project's dependencies isolated from the rest of your system.

## 1. Create the project directory

Create a directory where the project and its Python environment will be stored:

```bash
mkdir mon_ia_mac
cd mon_ia_mac

```

> You can replace `mon_ia_mac` with any directory name you prefer.

---

## 2. Create a virtual environment

Create a Python virtual environment named `.venv`:

```bash
python3 -m venv .venv

```

The `.venv` directory contains an isolated Python installation and the project's dependencies.

The leading `.` makes the directory hidden by default in Finder, keeping the project directory cleaner.

---

## 3. Activate the virtual environment

Activate the environment with:

```bash
source .venv/bin/activate

```

Once activated, you should see `(.venv)` at the beginning of your Terminal prompt:

```text
(.venv) user@Mac mon_ia_mac %

```

> **Important:** The virtual environment needs to be activated before installing dependencies or running the Python scripts.

---

## 4. Install the required Python packages

With the virtual environment activated, install the main dependencies:

```bash
pip install numpy torch torchvision tqdm safetensors

```

This installs:

* **NumPy** — numerical operations and data processing
* **PyTorch** — model implementation and training
* **Torchvision** — additional PyTorch utilities
* **Safetensors** — fast and safe model weight serialization

PyTorch can use Apple's **MPS (Metal Performance Shaders)** backend on compatible Apple Silicon Macs.

---

## 🔁 Using the environment later

You only need to create the virtual environment once.

The next time you restart your Mac and want to work on the project:

### 1. Open Terminal

### 2. Go to the project directory

```bash
cd mon_ia_mac

```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate

```

You should once again see:

```text
(.venv)

```

at the beginning of your Terminal prompt.

You can then run the project normally.

---

# 📋 Requirements

Before starting, make sure you have the following installed:

* **macOS**
* **Python 3**
* **Rust / Cargo**
* **PyTorch** with MPS support
* **NumPy**
* **Torchvision**

You can check your installations with:

```bash
python --version
cargo --version

```

To check whether PyTorch can access Apple's MPS backend:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"

```

If MPS is available, the command should return:

```text
True

```

---

# 📁 Project Structure

The project is organized as follows:

```text
.
├── data/
│   ├── vocab/
│   │   └── mon_vocabulaire_bpe.json
│   ├── processed/
│   │   └── train.bin
│   └── trainedmodel/
│       ├── model.safetensors
│       ├── model_instruct.safetensors
│       └── train_meta.json
│
├── src/
│   ├── gpt01_tokenizer/
│   │   └── bpe/
│   ├── gpt02_data_prep/
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── gpt03_pretraining/
│   │   ├── config.py
│   │   ├── gpt.py
│   │   └── train.py
│   ├── gpt04_finetuning/
│   │   └── finetune.py
│   └── gpt05_inference/
│       └── generate.py
│
├── .venv/
│
└── README.md

```

> The `.venv/` directory should normally **not be committed to Git**. Add it to your `.gitignore` file.

Example:

```gitignore
.venv/
__pycache__/
*.pyc

```

---

# 🚀 Full Pipeline

Once the environment is activated and the dependencies are installed, you can start the training pipeline.

## 1. Generate the BPE Vocabulary

The first step compiles the Rust tokenizer and generates the **BPE (Byte Pair Encoding)** vocabulary and merge rules using multi-threading (`rayon`).

From the project root:

```bash
cd src/gpt01_tokenizer/bpe
cargo run --release
cd ../../..

```

### ⏱️ Estimated time

Approximately **5–15 minutes**, depending on your machine and corpus size.

### ✅ Verification

The following file should be generated:

```text
data/vocab/mon_vocabulaire_bpe.json

```

The file should contain a non-empty JSON dictionary containing the tokenizer's merge rules.

---

# 2. Prepare and Binarize the Corpus

The raw text corpus must then be converted into numerical data that can be efficiently loaded during training.

Run:

```bash
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

The script converts the corpus into an array of **`uint16` integers**, stored in a binary file that can be memory-mapped during training.

### ⏱️ Estimated time

The execution time depends mainly on the size of the corpus.

### ✅ Verification

The following file should be created:

```text
data/processed/train.bin

```

Its size should be non-zero and proportional to the size of the input corpus.

---

# 🔥 3. Run a Warm-Up Test

Before starting a full training run that can take several hours, it is recommended to perform a short warm-up test.

Open:

```text
src/gpt03_pretraining/config.py

```

Temporarily change the following parameters:

```python
max_iters = 50
eval_interval = 10

```

Then start the training:

```bash
python -m src.gpt03_pretraining.train

```

### ⏱️ Estimated time

Approximately **2 minutes**.

### 🔎 What to check

While the model is training, make sure that:

* The **loss** is displayed correctly.
* The **MPS backend** is being used.
* No CUDA-related errors are reported.
* Memory usage remains reasonable.
* macOS memory pressure stays in the green.

You can monitor memory usage using:

**Activity Monitor → Memory**

> 💡 This warm-up test is useful for detecting configuration or memory issues before starting a long training run.

---

# 🏋️ 4. Run Full Pre-Training

Once the warm-up test has completed successfully, you can start the full pre-training process.

Open:

```text
src/gpt03_pretraining/config.py

```

and restore the target number of iterations:

```python
max_iters = 9000

```

You can adjust this value depending on your training target.

Then run:

```bash
python -m src.gpt03_pretraining.train

```

### ⏱️ Estimated time

**Several hours**, depending on:

* Number of iterations
* Model size
* Corpus size
* Training configuration
* Apple Silicon performance

### 📦 Expected output

At the end of training, the base model checkpoint should be available at:

```text
data/trainedmodel/model.safetensors

```

### ✅ Verification

The training should be considered successful if:

* The checkpoint is generated correctly.
* Training completes without errors (or can be paused/resumed smoothly via `Ctrl+C` thanks to `train_meta.json`).
* The loss decreases during training following the cosine decay schedule.
* The final validation loss stabilizes around **3.95**.

---

# 🎯 5. Supervised Fine-Tuning (SFT)

The pre-trained model can now be fine-tuned to improve its ability to follow instructions.

This stage uses an **Alpaca-style instruction dataset** to teach the model how to respond to prompts in a question-and-answer format, applying a `-100` target mask on prompts so gradients only update on responses.

Run:

```bash
python -m src.gpt04_finetuning.finetune

```

### ⏱️ Estimated time

Approximately **1 hour**.

The actual duration depends on early stopping criteria, batch accumulation, and hardware.

### 📦 Expected output

The script loads the base checkpoint:

```text
data/trainedmodel/model.safetensors

```

and generates the instruction-tuned model:

```text
data/trainedmodel/model_instruct.safetensors

```

---

# 💬 6. Run Interactive Inference

Once fine-tuning is complete, you can interact with the model directly from the terminal.

Run:

```bash
python -m src.gpt05_inference.generate

```

This launches an interactive command-line interface with real-time token streaming and repetition penalties.

Example:

```text
=======================================================
Alpaca instruction assistant (type 'exit' to quit)
=======================================================

Your instruction: List 3 advantages of renewable energy.

Response: Finding electricity. 
A new solar system is a system that can be used to generate electricity resources and the electricity needed by reducing the costs of its efficient speed...

```

### ✅ Verification

The model should:

* Accept user prompts.
* Stream generated responses in real-time.
* Stop cleanly upon generating `<|endoftext|>`.
* Respect the configured context window (**1,024 tokens**).

---

# 🔄 Pipeline Overview

The complete workflow can be summarized as:

```text
                  Raw Text Corpus
                         │
                         ▼
               ┌───────────────────┐
               │ 1. BPE Tokenizer  │
               │    Rust / Cargo   │
               └─────────┬─────────┘
                         │
                         ▼
            mon_vocabulaire_bpe.json
                         │
                         ▼
               ┌───────────────────┐
               │ 2. Data Prep &    │
               │    Binarization   │
               │    Rust / Cargo   │
               └─────────┬─────────┘
                         │
                         ▼
                      train.bin
                         │
                         ▼
               ┌───────────────────┐
               │  3. Warm-Up Test  │
               │   50 iterations   │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ 4. Pre-Training   │
               │  RMSNorm + RoPE   │
               │  SwiGLU + Cosine  │
               └─────────┬─────────┘
                         │
                         ▼
                 model.safetensors
                         │
                         ▼
               ┌───────────────────┐
               │ 5. SFT / Fine-    │
               │      Tuning       │
               │  Alpaca (-100)    │
               └─────────┬─────────┘
                         │
                         ▼
            model_instruct.safetensors
                         │
                         ▼
               ┌───────────────────┐
               │ 6. Interactive    │
               │ Streaming CLI     │
               └─────────┬─────────┘
                         │
                         ▼
                     💬 Chat

```

---

# ⚡ Quick Start

If everything is already configured, the complete pipeline is:

### Environment

```bash
cd mon_ia_mac
source .venv/bin/activate

```

### BPE Tokenizer

```bash
cd src/gpt01_tokenizer/bpe
cargo run --release
cd ../../..

```

### Data Preparation

```bash
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

### Warm-Up Test

Set:

```python
max_iters = 50
eval_interval = 10

```

Then run:

```bash
python -m src.gpt03_pretraining.train

```

### Full Pre-Training

Set:

```python
max_iters = 9000

```

Then run:

```bash
python -m src.gpt03_pretraining.train

```

### Fine-Tuning

```bash
python -m src.gpt04_finetuning.finetune

```

### Inference

```bash
python -m src.gpt05_inference.generate

```

---

# 📊 Generated Files

| File | Description |
| --- | --- |
| `data/vocab/mon_vocabulaire_bpe.json` | BPE vocabulary and merge rules (32k tokens) |
| `data/processed/train.bin` | Tokenized and binarized uint16 training corpus |
| `data/trainedmodel/model.safetensors` | Pre-trained base model weights (val loss ~3.95) |
| `data/trainedmodel/model_instruct.safetensors` | Instruction-tuned model weights (val loss ~3.32) |
| `data/trainedmodel/train_meta.json` | Step index and resumption metadata |

---

# 🛠️ Troubleshooting

## `mon_vocabulaire_bpe.json` was not generated

Run the tokenizer again:

```bash
cd src/gpt01_tokenizer/bpe
cargo run --release
cd ../../..

```

Then check:

```text
data/vocab/mon_vocabulaire_bpe.json

```

---

## `train.bin` was not generated

Run the data preparation script again:

```bash
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

---

## MPS is not available

Check the MPS backend with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"

```

If the result is:

```text
False

```

check that you are using a compatible Apple Silicon Mac and a recent version of PyTorch.

---

## Memory usage is too high

Start with the warm-up configuration:

```python
max_iters = 50

```

Then monitor memory usage using **Activity Monitor → Memory**.

It is much safer to identify memory problems during the warm-up test than during a multi-hour training run.

---

## The checkpoint was not generated

Make sure that training reaches the end of the training loop and that no error occurs in the terminal.

The expected checkpoint is:

```text
data/trainedmodel/model.safetensors

```

---

# 🧠 Project Goal

The goal of this project is to understand and implement the main stages involved in building a modern small Language Model from scratch:

**Tokenization → Data Preparation → Pre-Training → Fine-Tuning → Inference**

The entire pipeline is designed to run **locally on Apple Silicon**, using the MPS backend to accelerate training.