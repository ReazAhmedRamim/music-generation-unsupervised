# src/models/transformer.py
"""
Task 3 – Transformer Decoder for Long-Horizon Music Generation
==============================================================

Architecture:
  Token Embedding  +  Genre Embedding  +  Positional Encoding
       ↓
  N × Transformer Decoder Layer (masked self-attention + FF)
       ↓
  Linear → logits

Autoregressive probability:
  p(X) = ∏_t p(x_t | x_{<t})

Training loss (negative log-likelihood):
  L_TR = -Σ_t log p_θ(x_t | x_{<t})

Perplexity:
  PPL = exp(1/T · L_TR)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES,
    TF_D_MODEL, TF_NHEAD, TF_NUM_LAYERS, TF_DIM_FF, TF_DROPOUT, TF_MAX_SEQ_LEN
)


# ──────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. 2017:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = TF_MAX_SEQ_LEN, dropout: float = TF_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                 # (max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                  # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────
# Music Transformer Decoder
# ──────────────────────────────────────────────────

class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive music generation.

    Conditioning on genre is done via an additive genre embedding:
        h_t = Emb(x_t) + Emb(genre)   (following project spec)
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        d_model:     int   = TF_D_MODEL,
        nhead:       int   = TF_NHEAD,
        num_layers:  int   = TF_NUM_LAYERS,
        dim_ff:      int   = TF_DIM_FF,
        dropout:     float = TF_DROPOUT,
        max_seq_len: int   = TF_MAX_SEQ_LEN,
        num_genres:  int   = NUM_GENRES
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.genre_emb  = nn.Embedding(num_genres + 1, d_model, padding_idx=0)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder stack
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True          # Pre-LN for stability
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Weight tying (token embedding ↔ output projection)
        self.fc_out.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def _padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """True where token is PAD (id=0)."""
        return x == 0                                      # (B, T)

    def forward(self, x, genre_ids=None):
        """
        Args:
            x:         (B, T) long tensor (input sequence, shifted right)
            genre_ids: (B,)   genre labels (optional)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape
        device = x.device

        # Token embedding + genre embedding + positional encoding
        tok_emb = self.token_emb(x) * math.sqrt(self.d_model)  # (B, T, d)
        if genre_ids is not None:
            g_emb    = self.genre_emb(genre_ids).unsqueeze(1)   # (B, 1, d)
            tok_emb  = tok_emb + g_emb                          # broadcast over T
        h = self.pos_enc(tok_emb)                               # (B, T, d)

        # Causal mask
        causal_mask  = self._causal_mask(T, device)             # (T, T)
        padding_mask = self._padding_mask(x)                    # (B, T)

        # Decoder-only: memory = h itself (self-attention only)
        out = self.transformer(
            tgt=h,
            memory=h,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )                                                        # (B, T, d)

        logits = self.fc_out(out)                               # (B, T, V)
        return logits

    # ── Loss ──────────────────────────────────────────────────────

    @staticmethod
    def autoregressive_loss(logits, target, pad_token=0):
        """
        L_TR = -Σ_t log p_θ(x_t | x_{<t})

        Shifts target by 1: model input is x[:-1], target is x[1:].

        Args:
            logits: (B, T, V)  – model output for input x[:-1]
            target: (B, T)     – full sequence (including BOS)
        Returns:
            scalar NLL loss
        """
        # Input is x[0..T-2], target is x[1..T-1]
        B, T, V = logits.shape
        # logits already computed on x[:-1] slice → target is x[1:]
        tgt = target[:, 1:1 + T]                               # (B, T)
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=pad_token
        )
        return loss

    @staticmethod
    def perplexity(loss: float) -> float:
        """PPL = exp(1/T · L_TR) ≈ exp(mean NLL)."""
        return math.exp(loss)

    # ── Generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        genre_id: int,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        self.eval()
        device    = prompt.device
        seq       = prompt.clone()
        genre_ids = torch.tensor([genre_id], device=device)

        from src.config import EOS_TOKEN, VOCAB_SIZE

        for _ in range(max_new_tokens):

            # Need at least 2 tokens for seq[:, :-1] to be non-empty
            if seq.size(1) < 2:
                next_tok = torch.tensor([[1]], device=device)  # BOS as filler
                seq = torch.cat([seq, next_tok], dim=1)
                continue

            # Crop context to max_seq_len
            seq_crop = seq[:, -self.max_seq_len:]
            inp      = seq_crop[:, :-1]                        # (1, T-1)

            if inp.size(1) == 0:
                next_tok = torch.tensor([[1]], device=device)
                seq = torch.cat([seq, next_tok], dim=1)
                continue

            logits = self.forward(inp, genre_ids)              # (1, T-1, V)
            logits = logits[:, -1, :].float()                  # (1, V) — last position only

            # Apply temperature
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            # ── Top-k filtering ───────────────────────────────
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                min_val = topk_vals[:, -1].unsqueeze(-1)
                logits  = logits.masked_fill(logits < min_val, float("-inf"))

            # ── Top-p (nucleus) filtering ─────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cum_probs    = torch.cumsum(probs_sorted, dim=-1)

                # Remove tokens with cumulative prob above threshold
                # Shift right so first token above threshold is kept
                remove = cum_probs - probs_sorted > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

                # Scatter back to original order
                logits = torch.zeros_like(sorted_logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            # ── Safe sampling ─────────────────────────────────
            # Guard: if everything is -inf, reset to uniform over pitch tokens
            if torch.all(logits == float("-inf")) or torch.any(torch.isnan(logits)):
                logits = torch.zeros(1, VOCAB_SIZE, device=device)
                logits[:, :4] = float("-inf")               # mask special tokens

            probs    = F.softmax(logits, dim=-1)             # (1, V)

            # Final safety clamp — no zeros, no NaNs
            probs    = torch.clamp(probs, min=1e-9)
            probs    = probs / probs.sum(dim=-1, keepdim=True)  # renormalise

            next_tok = torch.multinomial(probs, num_samples=1)   # (1, 1)
            seq      = torch.cat([seq, next_tok], dim=1)

            if next_tok.item() == EOS_TOKEN:
                break

        return seq

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return (f"MusicTransformer("
                f"d_model={self.d_model}, "
                f"layers={len(self.transformer.layers)}, "
                f"params={total:,})")
