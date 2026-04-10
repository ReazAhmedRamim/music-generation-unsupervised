# src/models/autoencoder.py
"""
Task 1 – LSTM Autoencoder for Single-Genre Music Generation
============================================================

Architecture:
  Encoder: Embedding → LSTM (stacked) → Linear → latent z
  Decoder: z → Linear → LSTM (stacked) → Linear → logits

Mathematical model:
  z     = f_φ(X)          (encode)
  X_hat = g_θ(z)          (decode)
  L_AE  = Σ_t ‖x_t - x̂_t‖²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    VOCAB_SIZE, SEQUENCE_LENGTH,
    AE_HIDDEN_DIM, AE_LATENT_DIM, AE_NUM_LAYERS, AE_DROPOUT
)


class LSTMEncoder(nn.Module):
    """Encodes a token sequence into a fixed-size latent vector z."""

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = 128,
        hidden_dim:  int = AE_HIDDEN_DIM,
        latent_dim:  int = AE_LATENT_DIM,
        num_layers:  int = AE_NUM_LAYERS,
        dropout:     float = AE_DROPOUT
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T) long tensor of token ids
        Returns:
            z: (B, latent_dim) latent embedding
            hidden: LSTM hidden states (for decoder initialisation)
        """
        emb = self.dropout(self.embedding(x))           # (B, T, embed_dim)
        out, (h_n, c_n) = self.lstm(emb)                # h_n: (layers, B, H)
        # Use last-layer hidden state as sequence summary
        z = self.fc_latent(h_n[-1])                     # (B, latent_dim)
        return z, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """Decodes a latent vector z back into a token sequence."""

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = 128,
        hidden_dim:  int = AE_HIDDEN_DIM,
        latent_dim:  int = AE_LATENT_DIM,
        num_layers:  int = AE_NUM_LAYERS,
        dropout:     float = AE_DROPOUT,
        seq_len:     int = SEQUENCE_LENGTH
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_init   = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, z):
        """Initialise LSTM hidden state from latent vector z."""
        B = z.size(0)
        # Project z → (num_layers * hidden_dim), then split per layer
        h = self.fc_init(z)                              # (B, num_layers * H)
        h = h.view(B, self.num_layers, self.hidden_dim)  # (B, layers, H)
        h = h.permute(1, 0, 2).contiguous()              # (layers, B, H)
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, target=None):
        """
        Teacher-forced forward pass.

        Args:
            z:      (B, latent_dim)
            target: (B, T) ground-truth token ids for teacher forcing
                    If None, greedy auto-regressive decoding is used.
        Returns:
            logits: (B, T, vocab_size)
        """
        B = z.size(0)
        h, c = self._init_hidden(z)

        if target is not None:
            # Teacher forcing: shift target right → input to decoder
            dec_input = target[:, :-1]                   # (B, T-1)
            emb = self.dropout(self.embedding(dec_input)) # (B, T-1, E)
            out, _ = self.lstm(emb, (h, c))              # (B, T-1, H)
            logits = self.fc_out(out)                    # (B, T-1, V)
        else:
            # Greedy decoding
            from src.config import BOS_TOKEN
            tok = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=z.device)
            logits_list = []
            for _ in range(self.seq_len):
                emb = self.dropout(self.embedding(tok))  # (B, 1, E)
                out, (h, c) = self.lstm(emb, (h, c))
                step_logits = self.fc_out(out)           # (B, 1, V)
                logits_list.append(step_logits)
                tok = step_logits.argmax(dim=-1)         # (B, 1)
            logits = torch.cat(logits_list, dim=1)       # (B, T, V)

        return logits


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder (Encoder + Decoder).

    Loss:
        L_AE = Σ_t ‖x_t - x̂_t‖²  (MSE on logits vs one-hot)
        or equivalently cross-entropy on token predictions.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = LSTMEncoder(**kwargs)
        self.decoder = LSTMDecoder(**kwargs)

    def forward(self, x):
        """
        Args:
            x: (B, T) long tensor
        Returns:
            logits: (B, T-1, vocab_size)
            z:      (B, latent_dim)
        """
        z, _ = self.encoder(x)
        logits = self.decoder(z, target=x)
        return logits, z

    def encode(self, x):
        """Return latent embedding only."""
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        """Greedy decode from latent z."""
        return self.decoder(z, target=None)

    def reconstruct(self, x):
        """Encode then decode (with teacher forcing)."""
        z, _ = self.encoder(x)
        logits = self.decoder(z, target=x)
        return logits

    # ── Loss ──────────────────────────────────────────────────────

    @staticmethod
    def reconstruction_loss(logits, target, pad_token=0):
        """
        L_AE = Σ_t ‖x_t - x̂_t‖² approximated as cross-entropy
               (equivalent when target is one-hot).

        Args:
            logits: (B, T-1, V)
            target: (B, T) ground-truth token ids
        Returns:
            scalar loss
        """
        B, T_minus1, V = logits.shape
        # decoder predicts tokens 1..T, target provides 1..T
        tgt = target[:, 1:1 + T_minus1]                 # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=pad_token
        )
        return loss

    def __repr__(self):
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total      = enc_params + dec_params
        return (f"LSTMAutoencoder(\n"
                f"  encoder_params={enc_params:,}\n"
                f"  decoder_params={dec_params:,}\n"
                f"  total_params={total:,}\n"
                f")")
