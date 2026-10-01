"""Supervised fine-tuning for instruction-following behavior.

The base language model is trained on raw text first.  This script then trains
it on Alpaca-style instruction/response pairs.  The instruction tokens are
masked with ``-100`` so the loss teaches the model to generate the response,
not to reproduce the prompt.
"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PRETRAINING_DIR = CURRENT_DIR.parent / "gpt03_pretraining"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))

from config import *
from gpt import GPTLanguageModel
from train import encode, decode


# Fine-tuning usually needs a smaller learning rate than pre-training: the
# model should adapt its behavior without overwriting its language knowledge.
FT_LR = 1e-5
FT_ITERS = 1500
EVAL_STEP = 100
EVAL_ITERS = 20
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# Early stopping and validation tracking (identical security logic to step 03)
best_val_loss = float("inf")
patience = 5  # Max consecutive evaluations without improvement
no_improvement_count = 0
OVERFIT_THRESHOLD = 0.8  # Trigger stop if validation loss diverges from best score

try:
    eos_token_id = encode("<|endoftext|>")[0]
except Exception:
    eos_token_id = 0

# Load the same architecture and the weights produced by pre-training.
model = GPTLanguageModel().to(device)
try:
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    print("Base model loaded from 'model.safetensors'")
except FileNotFoundError:
    print("Error: 'model.safetensors' was not found.")
    exit(1)

# Load and tokenize the instruction dataset.
print("Loading alpaca_data.json...")
with open(alpaca_data_path, "r", encoding="utf-8") as f:
    alpaca_raw = json.load(f)

# Each training example is stored as (input_token_ids, target_token_ids).
tokenized_examples = []
print("Encoding instructions with the BPE tokenizer...")

for item in alpaca_raw:
    # Keep the prompt format identical to the format used during inference.
    if item.get("input", "").strip():
        user_prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
    else:
        user_prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"

    bot_response = f"{item['output']}"

    prompt_ids = encode(user_prompt)
    response_ids = encode(bot_response) + [eos_token_id]

    # The model sees prompt followed by response and learns next-token shifts.
    sequence = prompt_ids + response_ids
    
    # Keep every example within the model's maximum context window.
    if len(sequence) > block_size + 1:
        sequence = sequence[:block_size + 1]
        prompt_length = min(len(prompt_ids), block_size)
    else:
        prompt_length = len(prompt_ids)

    # CrossEntropyLoss ignores targets equal to -100.  Therefore only response
    # positions contribute gradients; prompt tokens provide context but are not
    # treated as text the model must reproduce.
    targets = [-100] * prompt_length + sequence[prompt_length:]

    if len(sequence) > 1:
        tokenized_examples.append((sequence, targets))

print(f"{len(tokenized_examples)} examples ready for fine-tuning.")

# Split dataset into training (90%) and validation (10%) splits
random.seed(1337)
random.shuffle(tokenized_examples)
split_idx = int(len(tokenized_examples) * 0.9)
train_examples = tokenized_examples[:split_idx]
val_examples = tokenized_examples[split_idx:]
print(f"Dataset split: {len(train_examples)} train examples, {len(val_examples)} validation examples.")

# Build fixed-size batches by padding shorter examples.
def get_batch(split="train"):
    """Sample a batch and align inputs/targets for next-token prediction."""
    dataset = train_examples if split == "train" else val_examples
    indices = torch.randint(len(dataset), (BATCH_SIZE,))
    batch_inputs, batch_targets = [], []

    for index in indices:
        token_ids, target_ids = dataset[index]
        # Input position t predicts target position t+1.
        sequence_inputs = token_ids[:-1]
        sequence_targets = target_ids[1:]

        # Clamp max sequence to context window
        if len(sequence_inputs) > block_size:
            sequence_inputs = sequence_inputs[:block_size]
            sequence_targets = sequence_targets[:block_size]

        padding_length = block_size - len(sequence_inputs)
        sequence_inputs = sequence_inputs + [0] * padding_length
        sequence_targets = sequence_targets + [-100] * padding_length

        batch_inputs.append(torch.tensor(sequence_inputs, dtype=torch.long))
        batch_targets.append(torch.tensor(sequence_targets, dtype=torch.long))

    return torch.stack(batch_inputs).to(device), torch.stack(batch_targets).to(device)


@torch.no_grad()
def estimate_val_loss():
    """Evaluate validation loss across multiple mini-batches without gradient computation."""
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        input_batch, target_batch = get_batch("val")
        logits, _ = model(input_batch)
        batch_size_current, time_steps, vocabulary_size = logits.shape
        flattened_logits = logits.view(batch_size_current * time_steps, vocabulary_size)
        flattened_targets = target_batch.view(batch_size_current * time_steps)
        loss = F.cross_entropy(flattened_logits, flattened_targets, ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# Optimize the model with masked next-token cross-entropy.
optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR)
model.train()

print("\nStarting fine-tuning...")
pbar = tqdm(range(FT_ITERS), desc="Fine-tuning Alpaca", unit="step")
last_train_loss = float("nan")

for step in pbar:
    # Validation evaluation and checkpointing (aligned with step 03 logic)
    if step % EVAL_STEP == 0:
        val_loss = estimate_val_loss()
        tqdm.write(f"Step {step:4d} / {FT_ITERS} | Validation loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

        # Save checkpoint only on validation improvements
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            save_file(model.state_dict(), instruct_model_path)
            tqdm.write(f"--> Best model checkpoint saved to '{instruct_model_path}' (val_loss: {val_loss:.4f})")
        else:
            no_improvement_count += 1
            tqdm.write(f"--> No validation improvement ({no_improvement_count}/{patience})")

        # Overfitting alert: halt if validation loss drifts significantly above best record
        if val_loss > best_val_loss + OVERFIT_THRESHOLD:
            tqdm.write(f"\n[Overfitting Alert] val_loss ({val_loss:.4f}) exceeded threshold ({best_val_loss:.4f} + {OVERFIT_THRESHOLD}).")
            tqdm.write("The model is starting to memorize answers. Halting fine-tuning early.")
            break

        # Early stopping if loss plateaus
        if no_improvement_count >= patience:
            tqdm.write(f"\n[Early Stopping] No improvement for {patience} evaluations. Halting fine-tuning early.")
            break

    optimizer.zero_grad(set_to_none=True)
    total_step_loss = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        input_batch, target_batch = get_batch("train")
        logits, _ = model(input_batch)

        batch_size_current, time_steps, vocabulary_size = logits.shape
        flattened_logits = logits.view(batch_size_current * time_steps, vocabulary_size)
        flattened_targets = target_batch.view(batch_size_current * time_steps)

        loss = F.cross_entropy(flattened_logits, flattened_targets, ignore_index=-100)
        
        loss = loss / GRAD_ACCUM_STEPS

        total_step_loss += loss.item() * GRAD_ACCUM_STEPS
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if device == "mps":
        torch.mps.empty_cache()

    last_train_loss = total_step_loss / GRAD_ACCUM_STEPS
    pbar.set_postfix(train_loss=f"{last_train_loss:.4f}", val_loss=f"{best_val_loss:.4f}")

pbar.close()
print(f"\nFine-tuning session ended. Optimal instruction model preserved at '{instruct_model_path}' with best val_loss: {best_val_loss:.4f}")