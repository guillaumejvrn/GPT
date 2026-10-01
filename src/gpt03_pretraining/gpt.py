"""Decoder-only Transformer used for next-token language modeling.

The model receives integer token IDs with shape ``(batch, time)`` and returns
one vocabulary-sized logit vector for every position.  During training, the
logit at position ``t`` is compared with the token at position ``t + 1``.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. SELF-ATTENTION
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Vectorized multi-head causal attention using PyTorch's native kernel."""

    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        batch_size_current, time_steps, channels = x.size()
        queries, keys, values = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        keys = keys.view(batch_size_current, time_steps, self.n_head, head_size).transpose(1, 2)
        queries = queries.view(batch_size_current, time_steps, self.n_head, head_size).transpose(1, 2)
        values = values.view(batch_size_current, time_steps, self.n_head, head_size).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=dropout if self.training else 0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size_current, time_steps, channels)
        return self.resid_dropout(self.c_proj(attended))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. FEED-FORWARD NETWORK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise nonlinear transformation applied after attention."""

    def __init__(self, n_embd):
        """Build an expansion, ReLU, contraction, and dropout pipeline."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Transform each time position independently; shape is preserved."""
        return self.net(x)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. TRANSFORMER BLOCK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """One pre-normalized Transformer block with residual connections."""

    def __init__(self, n_embd, n_head):
        """Create attention and feed-forward sublayers."""
        super().__init__()
        self.sa = CausalSelfAttention()
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Apply normalized attention and feed-forward residual updates.

        The equations are ``x <- x + Attention(LayerNorm(x))`` followed by
        ``x <- x + FeedForward(LayerNorm(x))``.  Residual additions preserve
        an information path through all layers and make optimization easier.
        """
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. COMPLETE GPT MODEL
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    """Small decoder-only GPT model for autoregressive token prediction."""

    def __init__(self):
        """Create token/position embeddings, Transformer blocks, and output head."""
        super().__init__()
        # Token IDs become vectors in R^n_embd.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position IDs provide order information, which attention alone lacks.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # Converts each hidden vector into one score per vocabulary token.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Return logits and optionally the next-token cross-entropy loss.

        ``idx`` has shape ``(B, T)``.  Token and position embeddings are added
        to form ``(B, T, n_embd)``.  The output logits have shape
        ``(B, T, vocab_size)``.  When targets are supplied, both tensors are
        flattened to ``(B*T, ...)`` so PyTorch can compute one classification
        loss for every position.
        """
        batch_size_current, time_steps = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_ids = torch.arange(time_steps, device=device)
        position_embeddings = self.position_embedding_table(position_ids)
        x = token_embeddings + position_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            flattened_logits = logits.view(batch_size_current * time_steps, vocab_size)
            flattened_targets = targets.view(batch_size_current * time_steps)
            loss = F.cross_entropy(flattened_logits, flattened_targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively append sampled tokens to an initial context.

        At each step only the last ``block_size`` tokens are evaluated.  The
        final-position logits are converted to probabilities with softmax, one
        token is sampled from that categorical distribution, and it is appended
        to the sequence.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx