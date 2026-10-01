# 🧠 Large Language Model from Scratch

**[ English ](#-english) | [ 日本語 ](#-日本語)**

A modular, production-grade Language Model training pipeline written from scratch in **Rust** and **PyTorch**.  
RustとPyTorchを用いてフルスクラッチで実装された、モジュール式の大規模言語モデル（LLM）学習・推論パイプライン。

</div>

---

# 🇬🇧 English

A modular, production-grade Language Model training pipeline written from scratch in **Rust** and **PyTorch**.

This repository provides an end-to-end framework to build, train, align, and serve modern causal Transformers on consumer workstations, portable devices, or enterprise GPU clusters.

---

## 📌 Executive Overview

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

---

# 🇯🇵 日本語

**Rust** と **PyTorch** を用いてゼロから構築された、実用的かつモジュール式の大規模言語モデル（LLM）学習パイプライン。

ノートPCや一般的なワークステーションから、エンタープライズ向けのGPUクラスタに至るまで、最新の因果的Transformer（Causal Transformer）の構築、事前学習、指示調整（SFT）、推論実行を一気通貫で完結できるフレームワークです。

---

## 📌 プロジェクト概要

多くの教育用LLM実装は2019年の初期GPT-2設計（LayerNorm、学習済み絶対位置埋め込み、Python依存の低速トークナイザ、マスクなし自己回帰損失など）で止まっています。

本プロジェクトでは、近年の最先端オープンモデル（Llama 3、Mistral、Gemmaなど）で標準となっている**最新のアーキテクチャ**を採用しています。

* **コアエンジン:** 純粋なPyTorchによる事前学習、SFT、ストリーミング推論。デバイス（`CUDA`、`MPS`、`CPU`）の自動切り替えに対応。
* **データエンジニアリング:** **Rust** と `rayon` を活用した並列処理による、高速BPE（Byte Pair Encoding）トークナイザ生成およびゼロコピー・バイナリパッキング。
* **ハードウェア適応設計:** 物理マイクロバッチと勾配累積（Gradient Accumulation）を分離することで、限られたメモリ（VRAM/RAM）環境でもOOM（メモリ不足）を起こさず大規模コーパスの学習を実現。

---

## 🔬 詳細技術アーキテクチャ

本コードベースはスケールを考慮して設計されており、ハイパーパラメータの調整だけで小型モデルの検証から数十億パラメータ規模の基底モデル構築まで拡張可能です。

```text
                  生テキストコーパス (~40 GB)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 1. BPEトークナイザ │ ──► Rustマルチスレッド (`rayon`)
                   │    エンジン (Rust) │     32k語彙の高速抽出
                   └─────────┬─────────┘
                             │
                             ▼
                mon_vocabulaire_bpe.json
                             │
                             ▼
                   ┌───────────────────┐
                   │ 2. バイナリ変換   │ ──► 並列メモリーマッピング
                   │    データパッキング│     連続uint16バイナリ (`train.bin`)
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ 3. 最新LLM        │ ──► RMSNorm + RoPE + SwiGLU + Weight Tying
                   │    事前学習       │     コサインLR減衰 + 勾配クリッピング
                   └─────────┬─────────┘     マイクロバッチ + 勾配累積
                             │
                             ▼
                     model.safetensors (基底モデル)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 4. マスク付きSFT  │ ──► Alpaca指示データセット
                   │    ファインチューン│     プロンプトの厳密マスク (`target = -100`)
                   └─────────┬─────────┘
                             │
                             ▼
                 model_instruct.safetensors (指示追従モデル)
                             │
                             ▼
                   ┌───────────────────┐
                   │ 5. リアルタイムCLI │ ──► 自己回帰ストリーミング生成
                   │    推論エンジン   │     反復ペナルティ + 動的サンプリング
                   └───────────────────┘

```

### 1. Rustによる高スループット・トークナイゼーション

Pythonによる数十GB単位のテキスト処理は、GIL（グローバルインタプリタロック）やシリアライズのオーバーヘッドがボトルネックになります。

* トークナイゼーションサブシステム（`src/gpt01_tokenizer/bpe` および `src/gpt02_data_prep`）はネイティブRustで完全実装。
* `rayon` を使用し、利用可能なすべてのCPUコアでバイトペアの出現頻度を並列カウント。
* 32,768語彙の辞書をコンパクトなJSON形式で出力。
* 学習データを符号なし16ビット整数（`uint16`）の連続バイナリ列として保存し、学習時はOSのメモリーマッピング（`np.memmap`）によりゼロコピーで高速ロード。

### 2. Transformerアーキテクチャの最新化

`src/gpt03_pretraining/gpt.py` において、以下の最新コンポーネントを実装しています：

* **RMSNorm (Root Mean Square Normalization):** 各レイヤーの正規化処理において従来のLayerNormを置き換え。平均の減算を省略し二乗平均平方根のみで正規化することで、勾配の安定性を保ちながら逆伝播の計算コストを約10〜15%削減。
* **RoPE (Rotary Position Embeddings / 回転位置埋め込み):** 絶対位置埋め込みテーブルを廃止し、Query（$Q$）およびKey（$K$）ベクトルに回転行列を適用。距離に応じた自然な減衰特性を与え、学習時のコンテキスト長を超えた汎化性能を向上。
* **SwiGLU活性化関数:** 従来のGELU MLPをゲート付き線形ユニットに刷新：

$$\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{gate}})) \otimes (x W_{\text{up}}) W_{\text{down}}$$

従来の2層MLPに比べ、パラメータあたりの表現能力および勾配伝播効率が大幅に向上。

* **Weight Tying (重み共有):** 入力埋め込み層と最終線形プロジェクション層の重みを共有（$W_{\text{embed}} = W_{\text{head}}^T$）。パープレキシティを維持したままモデル総パラメータ数を約25〜30%削減（数百MBのメモリを節約）。

### 3. 学習のダイナミクスとメモリ管理

* **マイクロバッチ勾配累積:** 大量の物理メモリを必要とせず安定した学習を行うため、順伝播（`BATCH_SIZE = 1`）と重み更新（`GRAD_ACCUM_STEPS = 64–72`）を分離。物理VRAM使用量を**3 GB未満**に抑えつつ、仮想バッチサイズ65k〜75kトークンを達成。
* **コサイン学習率減衰 (Cosine Learning Rate Decay):** $1 \times 10^{-4}$ から $1 \times 10^{-5}$ への動的減衰スケジュールを導入。局所的なプラトーを回避し損失関数の滑らかな収束を実現。
* **状態保持型の中断・再開機能:** `Ctrl+C`（`SIGINT`）による中断を検知し、現在のステップ数、オプティマイザ状態を `train_meta.json` に保存。チェックポイント（`.safetensors`）から即座に学習を再開可能。

### 4. マスク付き教師あり微調整 (SFT)

指示追従アライメント（`src/gpt04_finetuning/finetune.py`）では、Alpaca形式の対話テンプレートを採用しています：

```text
### Instruction:
{user_query}

### Response:
{assistant_completion}<|endoftext|>

```

事前学習済み知識の破壊的忘却（Catastrophic Forgetting）を防ぐため、プロンプト（指示文）部分のターゲットトークンをすべて `-100` でマスク。PyTorchの `CrossEntropyLoss` がこれらを無視するため、勾配計算は**アシスタントの応答部分および完了トークンにのみ適用**されます。

---

## 📁 ディレクトリ構成

```text
.
├── data/
│   ├── raw/                       # 生テキストコーパス & alpaca_data.json
│   ├── vocab/                     # BPEマージルール & 32k語彙辞書
│   ├── processed/                 # メモリーマップ用バイナリデータ (train.bin)
│   └── trainedmodel/              # モデル重み & 学習メタデータ
│       ├── model.safetensors          # 事前学習済み基底モデル
│       ├── model_instruct.safetensors # SFT微調整済みモデル
│       └── train_meta.json            # ステップ数・再開用メタデータ
│
├── src/
│   ├── gpt01_tokenizer/
│   │   └── bpe/                   # Rust実装のBPEトークナイザ
│   ├── gpt02_data_prep/
│   │   └── src/                   # Rust実装の高速バイナリ変換クレート
│   ├── gpt03_pretraining/
│   │   ├── config.py              # ハイパーパラメータ & デバイス設定
│   │   ├── gpt.py                 # 最新Transformer (RMSNorm, RoPE, SwiGLU)
│   │   └── train.py               # コサインスケジューラ付き事前学習ループ
│   ├── gpt04_finetuning/
│   │   └── finetune.py            # -100マスク付きAlpaca SFT
│   └── gpt05_inference/
│       └── generate.py            # 反復ペナルティ付きCLIストリーミング推論
│
├── .gitignore
└── README.md

```

---

## ⚙️ 環境構築とセットアップ

### 1. 必要要件

* **Python 3.10+**
* **Rust & Cargo**（データ前処理・BPE生成用）
* **PyTorch 2.2+**（NVIDIA CUDA、Apple Silicon MPS、またはCPU対応）

インストールの確認：

```bash
python3 --version
cargo --version
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | MPS: {torch.backends.mps.is_available()}')"

```

### 2. 仮想環境の作成

```bash
# リポジトリのクローン
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
pip install numpy torch torchvision tqdm safetensors

```

---

## 🎛️ ハードウェア別スケーリング設定

実行前に `src/gpt03_pretraining/config.py` を開き、使用するハードウェア環境に応じてバッチ設定を調整します：

| 対象環境 | メモリ目安 | 推奨 `BATCH_SIZE` | 推奨 `GRAD_ACCUM_STEPS` | 想定VRAMフットプリント |
| --- | --- | --- | --- | --- |
| **ノートPC / Mac / CPU** | 8 – 16 GB | `1` | `64 – 72` | **~2.5 – 3.0 GB** |
| **コンシューマGPU (RTX 4070 / 5070)** | 12 – 16 GB | `4` | `16` | **~6.0 – 8.0 GB** |
| **データセンターGPU (A100 / H100)** | 40 – 80 GB | `16 – 32` | `2 – 4` | **~24 – 40 GB** |

デバイスは実行時に自動選択されます：

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

```

---

## 🚀 実行手順

### ステップ 1: BPE語彙辞書の生成 (Rust)

生テキストコーパスからネイティブRustを用いて辞書を高速生成します：

```bash
cd src/gpt01_tokenizer/bpe
cargo run --release
cd ../../..

```

*生成ファイル: `data/vocab/mon_vocabulaire_bpe.json*`

### ステップ 2: コーパスのバイナリ化 (Rust)

テキストデータを16ビット整数（`uint16`）配列のバイナリファイルへ変換します：

```bash
cargo run --release --manifest-path src/gpt02_data_prep/Cargo.toml

```

*生成ファイル: `data/processed/train.bin*`

### ステップ 3: 基底モデルの事前学習

バイナリトークン列を用いて事前学習を開始します：

```bash
python -m src.gpt03_pretraining.train

```

* ターミナル上にステップ数、学習率、Train Loss、Validation Lossがリアルタイム表示されます。
* いつでも `Ctrl+C` で安全に一時停止できます。再度スクリプトを実行すると、`train_meta.json` を検知して自動で直前のステップから再開します。
*生成ファイル: `data/trainedmodel/model.safetensors*`

### ステップ 4: 指示微調整 (SFT)

事前学習済みモデルにAlpaca形式の指示追従チューニングを行います：

```bash
python -m src.gpt04_finetuning.finetune

```

* 定期的に評価を行い、Validation Lossを監視します。
* Early Stopping機能により、過学習が発生する前に最適な重みを保持して終了します。
*生成ファイル: `data/trainedmodel/model_instruct.safetensors*`

### ステップ 5: インタラクティブ推論 (CLI)

ストリーミング出力に対応したCLIを起動し、モデルと対話します：

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

## 📊 実験結果・ベンチマーク

約125Mパラメータ設定（`block_size = 1024`, `n_embd = 768`, `n_layer = 12`, `n_head = 12`, `vocab_size = 32768`）における測定値：

| フェーズ | データセット / 規模 | ステップ数 | 収束指標 | 最大メモリ消費 |
| --- | --- | --- | --- | --- |
| **BPEトークナイズ** | 約40 GB生コーパス | マルチスレッド処理 | 32,768マージ | CPUマルチコア依存 |
| **事前学習** | 約6億トークン | 8,598ステップ | **Validation Loss: 3.9556** | **3.0 GB未満** |
| **指示微調整 (SFT)** | Alpaca (5.2万件) | 1,200ステップ (Early Stop) | **Validation Loss: 3.3217** | **2.5 GB未満** |

---

## 💡 今後の拡張性

本フレームワークは現代的なTransformer規格に準拠して設計されているため、以下の高度な研究・開発のベースラインとしても活用可能です：

* **Grouped-Query Attention (GQA):** Key/Valueヘッドを共有することで、推論時のKVキャッシュ消費量を大幅に削減。
* **Mixture of Experts (MoE):** SwiGLU層を疎結合なTop-kルーター制御エキスパート群に置き換え、計算量を抑えつつ総パラメータ数をスケール。
* **Direct Preference Optimization (DPO):** SFT完了後のモデルに対し、報酬モデルを介さずペアデータから直接人間の好みをアライメント。

```
