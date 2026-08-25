"""
Module 4 Solution: LLM Architectures — Assemble GPT-2 and Generate Text

Complete reference implementation of a decoder-only transformer that
loads real GPT-2 weights and generates text via greedy and sampling decoding.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Provided: multi-head causal self-attention (the engine from Module 3)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention matching GPT-2's design."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


# ---------------------------------------------------------------------------
# Step 1: Embedding lookup
# ---------------------------------------------------------------------------


class EmbeddingLayer(nn.Module):
    """Map token IDs to vectors and add positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, max_pos: int = 1024) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_pos, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token and positional embeddings and add them."""
        b, t = token_ids.size()
        position = torch.arange(t, dtype=torch.long, device=token_ids.device)
        position = position.unsqueeze(0).expand(b, t)
        return self.token_embed(token_ids) + self.pos_embed(position)


# ---------------------------------------------------------------------------
# Step 2: The feed-forward network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network: linear, GELU, linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer FFN with GELU nonlinearity."""
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Step 3: A transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm attention + pre-norm FFN, each with residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with residual connections."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Step 4: Stack the blocks and add the output head
# ---------------------------------------------------------------------------


class GPT2Model(nn.Module):
    """Complete decoder-only model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_pos: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = EmbeddingLayer(vocab_size, d_model, max_pos)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass from token IDs to logits."""
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Step 5: Load pretrained GPT-2 weights from HuggingFace
# ---------------------------------------------------------------------------


def load_gpt2_weights(model: GPT2Model) -> None:
    """Load pretrained GPT-2 small weights into the custom model."""
    from transformers import GPT2LMHeadModel

    print("Downloading GPT-2 weights from HuggingFace ...")
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    pretrained_sd = pretrained.state_dict()

    custom_names = [n for n, _ in model.named_parameters()]
    mapping = {}
    for cn in custom_names:
        if cn == "embed.token_embed.weight":
            pn = "transformer.wte.weight"
        elif cn == "embed.pos_embed.weight":
            pn = "transformer.wpe.weight"
        elif cn.startswith("blocks."):
            rest = cn[len("blocks."):]
            pn = "transformer.h." + rest
        elif cn == "ln_f.weight":
            pn = "transformer.ln_f.weight"
        elif cn == "ln_f.bias":
            pn = "transformer.ln_f.bias"
        elif cn == "lm_head.weight":
            pn = "lm_head.weight"
        else:
            raise ValueError(f"Unmapped custom parameter: {cn}")
        mapping[cn] = pn

    custom_sd = model.state_dict()
    new_sd = {}
    for cn, pn in mapping.items():
        pt = pretrained_sd[pn]
        ct = custom_sd[cn]
        if pt.shape != ct.shape:
            raise ValueError(f"Shape mismatch for {cn}: custom {ct.shape} vs pretrained {pt.shape}")
        new_sd[cn] = pt

    model.load_state_dict(new_sd)
    print(f"Loaded {len(new_sd)} parameter tensors from pretrained GPT-2.")


# ---------------------------------------------------------------------------
# Step 6: Greedy decoding
# ---------------------------------------------------------------------------


def greedy_decode(model: GPT2Model, tokenizer, prompt: str, max_new: int = 10) -> str:
    """Generate text by always picking the most likely next token."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])


# ---------------------------------------------------------------------------
# Step 7: Temperature and top-k sampling
# ---------------------------------------------------------------------------


def _sample_topk_token(next_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Sample one token from the top-k logits."""
    topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
    threshold = topk_vals[:, -1].unsqueeze(-1)
    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
    probs = F.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_with_temperature_topk(
    model: GPT2Model,
    tokenizer,
    prompt: str,
    max_new: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate text by sampling from a temperature-scaled, top-k truncated distribution."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # Scale logits by temperature before top-k filtering.
            next_logits = next_logits / temperature

            next_token = _sample_topk_token(next_logits, top_k)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])


# ---------------------------------------------------------------------------
# Extra credit: top-p (nucleus) sampling
# ---------------------------------------------------------------------------


def sample_topp(
    model: GPT2Model,
    tokenizer,
    prompt: str,
    max_new: int = 10,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    """Generate text with nucleus (top-p) sampling."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature

            # Sort logits descending and compute cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens where cumulative probability exceeds top_p
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

            # Scatter back to original index order
            full_logits = torch.empty_like(next_logits)
            full_logits.scatter_(-1, sorted_indices, sorted_logits)

            probs = F.softmax(full_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])
