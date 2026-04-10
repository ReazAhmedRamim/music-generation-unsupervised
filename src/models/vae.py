# src/models/vae.py
"""
Task 2 – Variational Autoencoder (VAE) for Multi-Genre Music Generation
=======================================================================

Architecture:
  Encoder : Embedding → BiLSTM → Linear → (µ, log σ²)
  Decoder : z ⊕ genre_emb → LSTM → Linear → logits
  Sampling: z = µ + σ ⊙ ε,  ε ~ N(0, I)   (reparameterisation trick)

Loss:
  L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))
  D_KL  = -½ Σ (1 + log σ² - µ² - σ²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES,
    VAE_HIDDEN_DIM, VAE_LATENT_DIM, VAE_NUM_LAYERS, VAE_DROPOUT, VAE_BETA
)


class VAEEncoder(nn.Module):
    """
    Bidirectional LSTM encoder that outputs µ and log σ² for the
    latent Gaussian: q_φ(z|X) = N(µ(X), σ(X)).
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        embed_dim:   int   = 128,
        hidden_dim:  int   = VAE_HIDDEN_DIM,
        latent_dim:  int   = VAE_LATENT_DIM,
        num_layers:  int   = VAE_NUM_LAYERS,
        dropout:     float = VAE_DROPOUT
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Bidirectional → 2*hidden_dim
        self.fc_mu     = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T) long
        Returns:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        emb = self.dropout(self.embedding(x))            # (B, T, E)
        out, (h_n, _) = self.lstm(emb)                   # h_n: (2*L, B, H)
        # Concatenate last-layer forward & backward hidden states
        h_fwd = h_n[-2]                                  # (B, H)
        h_bwd = h_n[-1]                                  # (B, H)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)        # (B, 2H)
        mu     = self.fc_mu(h_cat)
        logvar = self.fc_logvar(h_cat)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    LSTM decoder conditioned on latent z and an optional genre embedding.
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        embed_dim:   int   = 128,
        hidden_dim:  int   = VAE_HIDDEN_DIM,
        latent_dim:  int   = VAE_LATENT_DIM,
        num_layers:  int   = VAE_NUM_LAYERS,
        dropout:     float = VAE_DROPOUT,
        num_genres:  int   = NUM_GENRES,
        genre_dim:   int   = 32,
        seq_len:     int   = SEQUENCE_LENGTH
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.genre_embedding = nn.Embedding(num_genres + 1, genre_dim, padding_idx=0)
        fused_dim = latent_dim + genre_dim

        self.fc_init   = nn.Linear(fused_dim, hidden_dim * num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim + fused_dim,  # input-feeding: z ⊕ genre at every step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, z_genre):
        B = z_genre.size(0)
        h = self.fc_init(z_genre)
        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, genre_ids, target=None):
        """
        Args:
            z:         (B, latent_dim)
            genre_ids: (B,) genre label integers
            target:    (B, T) for teacher forcing; None for autoregressive
        Returns:
            logits: (B, T-1, vocab_size) or (B, T, vocab_size)
        """
        B = z.size(0)
        g_emb  = self.genre_embedding(genre_ids)          # (B, genre_dim)
        z_cond = torch.cat([z, g_emb], dim=-1)            # (B, fused_dim)
        h, c   = self._init_hidden(z_cond)

        # Expand z_cond to concatenate at each timestep
        if target is not None:
            T = target.size(1) - 1
            dec_input = target[:, :-1]
            emb = self.dropout(self.embedding(dec_input))  # (B, T, E)
            z_expand = z_cond.unsqueeze(1).expand(-1, T, -1)
            lstm_in  = torch.cat([emb, z_expand], dim=-1) # (B, T, E+fused)
            out, _   = self.lstm(lstm_in, (h, c))
            logits   = self.fc_out(out)
        else:
            from src.config import BOS_TOKEN
            tok = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=z.device)
            logits_list = []
            for _ in range(self.seq_len):
                emb      = self.dropout(self.embedding(tok))
                z_expand = z_cond.unsqueeze(1)
                lstm_in  = torch.cat([emb, z_expand], dim=-1)
                out, (h, c) = self.lstm(lstm_in, (h, c))
                step_logits  = self.fc_out(out)
                logits_list.append(step_logits)
                tok = step_logits.argmax(dim=-1)
            logits = torch.cat(logits_list, dim=1)

        return logits


class MusicVAE(nn.Module):
    """
    β-VAE for multi-genre music generation.

    L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))
    """

    def __init__(self, beta: float = VAE_BETA, **kwargs):
        super().__init__()
        self.beta    = beta
        self.encoder = VAEEncoder(**{k: v for k, v in kwargs.items()
                                     if k in VAEEncoder.__init__.__code__.co_varnames})
        self.decoder = VAEDecoder(**{k: v for k, v in kwargs.items()
                                     if k in VAEDecoder.__init__.__code__.co_varnames})

    def reparameterise(self, mu, logvar):
        """
        z = µ + σ ⊙ ε,  ε ~ N(0, I)
        σ = exp(½ log σ²)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu   # deterministic at inference

    def forward(self, x, genre_ids):
        """
        Args:
            x:         (B, T) token ids
            genre_ids: (B,)   genre labels
        Returns:
            logits: (B, T-1, V)
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
            z:      (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        logits     = self.decoder(z, genre_ids, target=x)
        return logits, mu, logvar, z

    def generate(self, z, genre_ids):
        """Generate token sequence from a sampled latent z."""
        return self.decoder(z, genre_ids, target=None)

    def sample(self, n: int, genre_id: int, device):
        """Sample n new music sequences for a given genre."""
        z          = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        genre_ids  = torch.full((n,), genre_id, dtype=torch.long, device=device)
        logits     = self.generate(z, genre_ids)
        tokens     = logits.argmax(dim=-1)               # (n, T)
        return tokens

    # ── Loss ──────────────────────────────────────────────────────

    @staticmethod
    def kl_divergence(mu, logvar):
        """
        D_KL(q_φ(z|X) ‖ p(z)) = -½ Σ (1 + log σ² - µ² - σ²)
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def loss(self, logits, target, mu, logvar, pad_token=0):
        """
        L_VAE = L_recon + β · D_KL
        """
        B, T_minus1, V = logits.shape
        tgt = target[:, 1:1 + T_minus1]
        l_recon = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=pad_token
        )
        l_kl   = self.kl_divergence(mu, logvar)
        l_vae  = l_recon + self.beta * l_kl
        return l_vae, l_recon, l_kl

    def interpolate(self, x1, x2, genre_ids, steps=8):
        """
        Latent space interpolation between two music sequences.
        Returns a list of generated token sequences.
        """
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)
        results = []
        for alpha in torch.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            logits   = self.generate(z_interp, genre_ids)
            tokens   = logits.argmax(dim=-1)
            results.append(tokens)
        return results

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return f"MusicVAE(beta={self.beta}, latent_dim={self.encoder.fc_mu.out_features}, params={total:,})"
