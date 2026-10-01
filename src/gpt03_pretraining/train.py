"""Pre-training loop and the Python implementation of the BPE tokenizer.

The tokenizer must remain identical during data preparation, fine-tuning, and
inference.  It loads the merge table learned by the Rust tokenizer, applies
the merges in rank order, and can reverse token IDs back into UTF-8 text.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import regex as re
import numpy as np
import sys
import os
import math
from pathlib import Path
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import *
from gpt import GPTLanguageModel
from safetensors.torch import load_file, save_file

# =============================================================================
# 1. TOKENIZER
# =============================================================================

# This pattern splits text into pieces before byte-level BPE merges.  Keeping
# punctuation, spaces, numbers, and words in predictable pieces makes the
# merge vocabulary reusable across languages and input formats.
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

# The JSON maps strings such as ``"97,98"`` to the ID assigned when bytes 97
# and 98 were merged.  Sorting by ID reconstructs the original merge order.
with open(vocab_path, "r", encoding="utf-8") as f:
    raw_merges = json.load(f)

sorted_merges = sorted(raw_merges.items(), key=lambda item: item[1])

merge_ranks = {}
decoder_vocabulary = {i: bytes([i]) for i in range(256)}

for pair_text, merged_id in sorted_merges:
    first_id, second_id = map(int, pair_text.split(','))
    merge_ranks[(first_id, second_id)] = merged_id
    
    if first_id in decoder_vocabulary and second_id in decoder_vocabulary:
        decoder_vocabulary[merged_id] = decoder_vocabulary[first_id] + decoder_vocabulary[second_id]
    else:
        decoder_vocabulary[merged_id] = (
            decoder_vocabulary.get(first_id, b'?')
            + decoder_vocabulary.get(second_id, b'?')
        )

def encode(text):
    """Convert UTF-8 text into the integer IDs used by the model.

    Each pre-tokenized piece starts as one integer per UTF-8 byte.  At every
    iteration the pair with the smallest merge ID is selected, because lower
    IDs represent merges learned earlier.  All non-overlapping occurrences of
    that pair are replaced by its merged ID.  The process stops when no pair is
    present in the learned merge table.
    """
    pieces = re.findall(compiled_pattern, text)
    final_tokens = []

    for piece in pieces:
        tokens = list(piece.encode("utf-8"))
        while len(tokens) >= 2:
            pairs = list(zip(tokens, tokens[1:]))
            pair_to_merge = min(pairs, key=lambda pair: merge_ranks.get(pair, float('inf')))

            if pair_to_merge not in merge_ranks:
                break

            merged_id = merge_ranks[pair_to_merge]
            merged_tokens = []
            index = 0
            while index < len(tokens):
                if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == pair_to_merge:
                    merged_tokens.append(merged_id)
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            tokens = merged_tokens
        final_tokens.extend(tokens)
    return final_tokens

def decode(token_ids):
    """Convert model token IDs back into UTF-8 text.

    Each merged token stores the byte sequence it represents.  Concatenating
    those byte sequences before decoding is important: decoding each token
    independently could split a multi-byte UTF-8 character.
    """
    byte_sequence = b"".join(decoder_vocabulary[token_id] for token_id in token_ids)
    return byte_sequence.decode("utf-8", errors="replace")


# =============================================================================
# 2. TRAINING EXECUTION
# =============================================================================
if __name__ == "__main__":

    data = np.memmap(train_data_path, dtype=np.uint16, mode='r')

    def get_batch(split):
        n = int(0.9 * len(data))
        d = data[:n] if split == 'train' else data[n:]

        max_start = len(d) - block_size
        if max_start <= 0:
            raise ValueError(
                f"The {split} split needs more than {block_size} tokens; got {len(d)}."
            )

        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy((d[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((d[i+1:i+block_size+1]).astype(np.int64)) for i in ix])

        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(10)
            for k in range(10):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        """Cosine learning rate schedule with warmup."""
        warmup_iters = 200
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > max_iters, return min learning rate
        if it > max_iters:
            return min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    model = GPTLanguageModel()
    m = model.to(device)

    # Assure l'existence du dossier de destination du modèle
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    checkpoint_meta_path = Path(model_path).parent / "train_meta.json"
    start_iter = 0

    # Checkpoint resumption if weights exist
    if Path(model_path).exists():
        print(f"Resuming training: loading weights from '{model_path}'...")
        try:
            state_dict = load_file(model_path)
            m.load_state_dict(state_dict)
            print("Successfully restored model weights.")

            if checkpoint_meta_path.exists():
                with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    start_iter = meta.get("last_iter", 0) + 1
                print(f"Reprise exacte à l'étape {start_iter}/{max_iters}")
        except Exception as e:
            print(f"Warning: Could not load existing checkpoint ({e}). Starting fresh.")
    else:
        print("No previous checkpoint found. Starting training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Starting training on {len(data)} tokens...")

    # Paramètres pour la détection du surapprentissage et l'arrêt précoce
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Initial baseline evaluation if resuming
    if Path(model_path).exists():
        initial_losses = estimate_loss(m)
        best_val_loss = initial_losses['val'].item()
        print(f"Baseline resumed validation loss: {best_val_loss:.4f}")

    # --- TRAINING LOOP ---
    progress_bar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters, desc="Pré-entraînement", dynamic_ncols=True)
    current_iter = start_iter

    try:
        for iter in progress_bar:
            current_iter = iter

            # Calcul et mise à jour automatique du Learning Rate (Cosine Decay)
            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Periodically measure average loss without tracking gradients.
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(m)
                val_loss = losses['val'].item()
                tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f} (lr: {lr:.2e})")

                # Sauvegarde du meilleur modèle dès que la val_loss s'améliore
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_file(m.state_dict(), model_path)
                    with open(checkpoint_meta_path, "w", encoding="utf-8") as f:
                        json.dump({"last_iter": iter, "best_val_loss": best_val_loss}, f)
                    tqdm.write(f"--> Meilleur modèle mis à jour (val_loss: {best_val_loss:.4f}) dans '{model_path}'")
                else:
                    patience_counter += 1
                    tqdm.write(f"--> Pas d'amélioration ({patience_counter}/{patience})")

                    # Déclenchement de l'arrêt anticipé
                    if patience_counter >= patience:
                        tqdm.write(f"\nArrêt précoce déclenché à l'étape {iter} : la validation ne s'améliore plus.")
                        break

            optimizer.zero_grad(set_to_none=True)
            total_micro_loss = 0.0 

            # Accumulate several micro-batch gradients before updating weights.
            for micro_step in range(gradient_accumulation_steps):
                xb, yb = get_batch('train')
                logits, loss = m(xb, yb)
                
                loss = loss / gradient_accumulation_steps
                total_micro_loss += loss.item()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

            optimizer.step()

            if device == "mps":
                torch.mps.empty_cache()

            progress_bar.set_postfix(train_loss=f"{total_micro_loss:.4f}", val_loss=f"{best_val_loss:.4f}", lr=f"{lr:.1e}")

    except KeyboardInterrupt:
        print("\n\nEntraînement interrompu par l'utilisateur (Ctrl+C).")
        print("Sauvegarde d'urgence des poids actuels...")
        save_file(m.state_dict(), model_path)
        with open(checkpoint_meta_path, "w", encoding="utf-8") as f:
            json.dump({"last_iter": current_iter, "best_val_loss": best_val_loss}, f)
        print(f"--> Poids et état (étape {current_iter}) sauvegardés dans '{model_path}'. Tu pourras relancer plus tard !")

    # --- SAVE THE PRE-TRAINED MODEL ---
    print(f"\nEntraînement terminé. Le meilleur modèle reste sauvegardé dans '{model_path}' avec une val_loss de {best_val_loss:.4f}")

    # Generate a short sample as a final sanity check.
    m.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    print("\nGenerating a short sample:")
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))