"""Shared paths and training hyperparameters.

Every Python stage imports this module, so file locations are defined once and
the rest of the project can refer to them by name instead of duplicating paths.
The active profile below is the large-context configuration intended for an
Apple Silicon machine with an MPS-compatible PyTorch build.
"""

# -----------------------------------------------------------------------------
# Project files
# -----------------------------------------------------------------------------
# These are intentionally plain strings.  They are resolved relative to the
# project root when the scripts are launched from the root directory.
text_path = "data/raw/openwebtext_complet.txt"
alpaca_data_path = "data/raw/alpaca_data.json"
train_data_path = "data/processed/train.bin"
vocab_path = "data/vocab/bpeVocabulary.json"
model_path = "data/trainedmodel/model.safetensors"
instruct_model_path = "data/trainedmodel/model_instruct.safetensors"

# -----------------------------------------------------------------------------
# Model and optimization profile
# -----------------------------------------------------------------------------
# The model predicts one token at a time from a vocabulary of this size.
vocab_size = 32000
block_size = 1024       # Maximum number of tokens processed in one context.
n_embd = 1024           # Width of each token representation.
n_layer = 12            # Number of Transformer blocks stacked sequentially.
n_head = 16             # Attention heads; each head has 1024 / 16 = 64 features.

# A physical batch contains 12 sequences.  Six micro-batches are accumulated
# before one optimizer update, giving an effective batch size of 12 * 6 = 72.
batch_size = 1
gradient_accumulation_steps = 72

# AdamW learning rate and training schedule.
learning_rate = 1e-4
min_lr = 1e-5         # Reserved for a learning-rate decay schedule.
max_iters = 9000     # Number of optimizer updates.
eval_interval = 100    # Evaluate train/validation loss every 500 updates.
dropout = 0.1          # Probability of dropping activations during training.
device = "mps"         # Apple GPU backend; use "cuda" or "cpu" when needed.

