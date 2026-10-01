"""Inference script for the fine-tuned Alpaca instruction model.

Interactive CLI session allowing users to submit prompts and observe the model's
instruction-following capability.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

CURRENT_DIR = Path(__file__).resolve().parent
PRETRAINING_DIR = CURRENT_DIR.parent / "gpt03_pretraining"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))

from config import *
from gpt import GPTLanguageModel
from train import encode, decode


# Determine the end-of-sequence token ID from the tokenizer.
try:
    eos_token_id = encode("<|endoftext|>")[0]
except Exception:
    eos_token_id = 0

# Load model weights: prefer instruction-tuned checkpoint, fallback to base model.
model = GPTLanguageModel().to(device)
weights_path = instruct_model_path if Path(instruct_model_path).exists() else model_path

try:
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from '{weights_path}'")
except FileNotFoundError:
    print(f"Error: No model weights found at '{weights_path}'. Run pretraining or fine-tuning first.")
    exit(1)

model.eval()

@torch.no_grad()
def generate_response(
    history_ids: list,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
) -> tuple[str, list]:
    """Format user instruction and generate autoregressive model continuation."""
    # Use standard Alpaca prompt template identical to fine-tuning phase
    formatted_prompt = f"### Instruction:\n{prompt_text}\n\n### Response:\n"
    new_prompt_tokens = encode(formatted_prompt)

    # Append to rolling conversation history
    full_context_ids = history_ids + new_prompt_tokens

    # Crop history if context exceeds model block size
    if len(full_context_ids) > block_size:
        full_context_ids = full_context_ids[-block_size:]

    idx = torch.tensor(full_context_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = []

    print("\nResponse: ", end="", flush=True)

    for _ in range(max_new_tokens):
        # Crop context if sequence exceeds block size
        idx_cond = idx[:, -block_size:]

        # Forward pass to retrieve logits for the last token position
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply repetition penalty to discourage repeating already generated tokens
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Apply temperature scaling and top-k truncation
        if temperature > 0.0:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        next_token = idx_next.item()

        # Halt generation immediately when the model outputs the EOS token
        if next_token == eos_token_id:
            break

        generated_tokens.append(next_token)
        idx = torch.cat((idx, idx_next), dim=1)

        # Stream the token immediately to the terminal
        token_text = decode([next_token])
        sys.stdout.write(token_text)
        sys.stdout.flush()

    print("\n")
    response_text = decode(generated_tokens)
    
    # Return updated conversation history including the answer and trailing token
    updated_history = full_context_ids + generated_tokens + [eos_token_id]
    return response_text, updated_history

def main():
    print("\n" + "=" * 55)
    print("Alpaca instruction assistant (type 'exit' to quit, 'clear' to reset history)")
    print("=" * 55 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("Your instruction: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting...")
                break
            if user_input.lower() == "clear":
                conversation_history = []
                print("Conversation history cleared.\n" + "-" * 55)
                continue

            _, conversation_history = generate_response(
                conversation_history,
                user_input,
                max_new_tokens=256,
                temperature=0.7,
                top_k=40,
                repetition_penalty=1.2,
            )
            print("-" * 55)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()