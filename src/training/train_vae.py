# src/training/train_vae.py
"""
Task 2 – Training Script for β-VAE Multi-Genre Music Generator
==============================================================

Algorithm 2 (from project spec):
  Require: Multi-genre dataset D={X_i, y_i}, β, Epochs E
  1. Init params φ, θ
  2. for epoch = 1 to E:
       for each batch X in D:
         (µ, σ) = Encoder_φ(X)
         z = µ + σ ⊙ ε,  ε ~ N(0,I)
         X̂ = Decoder_θ(z)
         L_recon = ‖X - X̂‖²
         L_KL    = D_KL(q_φ(z|X) ‖ p(z))
         L_VAE   = L_recon + β L_KL
         (φ,θ) ← (φ,θ) - η ∇L_VAE
  3. Generate diverse multi-genre music by sampling z ~ N(0,I)

Usage:
    python src/training/train_vae.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    SPLIT_DIR, CHECKPOINTS_DIR, PLOTS_DIR,
    BATCH_SIZE, VAE_EPOCHS, VAE_LEARNING_RATE, VAE_BETA, DEVICE, SEED,
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES, VAE_LATENT_DIM
)
from src.models.vae import MusicVAE

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Dataset (multi-genre)
# ──────────────────────────────────────────────────

class MultiGenreMIDIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr, genre = self.data[idx]
        return (torch.tensor(arr,   dtype=torch.long),
                torch.tensor(genre, dtype=torch.long))


def load_split(name: str):
    path = os.path.join(SPLIT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        from src.preprocessing.midi_parser import _synthetic_segments
        data = []
        for g in range(NUM_GENRES):
            data.extend(_synthetic_segments(g, n=100))
        return data
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, beta_schedule=None, epoch=1):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    # Optional beta annealing (KL warm-up)
    beta = beta_schedule(epoch) if beta_schedule else model.beta

    for x, genre_ids in loader:
        x, genre_ids = x.to(device), genre_ids.to(device)
        optimizer.zero_grad()

        logits, mu, logvar, z = model(x, genre_ids)
        l_vae, l_recon, l_kl = model.loss(logits, x, mu, logvar)

        # Apply annealed beta
        l_vae_annealed = l_recon + beta * (l_vae - l_recon)
        l_vae_annealed.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += l_vae.item()
        total_recon += l_recon.item()
        total_kl    += l_kl.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    for x, genre_ids in loader:
        x, genre_ids = x.to(device), genre_ids.to(device)
        logits, mu, logvar, z = model(x, genre_ids)
        l_vae, l_recon, l_kl = model.loss(logits, x, mu, logvar)
        total_loss  += l_vae.item()
        total_recon += l_recon.item()
        total_kl    += l_kl.item()
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def linear_beta_schedule(warmup_epochs=10, max_beta=VAE_BETA):
    """KL warm-up: linearly increase β from 0 to max_beta over warmup_epochs."""
    def schedule(epoch):
        return min(max_beta, max_beta * epoch / warmup_epochs)
    return schedule


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = load_split("train")
    val_data   = load_split("val")

    train_ds = MultiGenreMIDIDataset(train_data)
    val_ds   = MultiGenreMIDIDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = MusicVAE(beta=VAE_BETA).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)
    beta_sch  = linear_beta_schedule(warmup_epochs=10)

    history = {"train_loss": [], "val_loss": [], "recon": [], "kl": []}
    best_val = float("inf")

    print("\n" + "=" * 50)
    print("  Training β-VAE (Task 2)")
    print("=" * 50)

    for epoch in range(1, VAE_EPOCHS + 1):
        tr_loss, tr_recon, tr_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta_sch, epoch
        )
        val_loss, val_recon, val_kl = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["recon"].append(tr_recon)
        history["kl"].append(tr_kl)

        print(f"  Epoch {epoch:3d}/{VAE_EPOCHS} | "
              f"Total: {tr_loss:.4f} | Recon: {tr_recon:.4f} | "
              f"KL: {tr_kl:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": val_loss},
                       os.path.join(CHECKPOINTS_DIR, "vae_best.pt"))

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(history["val_loss"],   label="Val",   color="#FF5722")
    axes[0].set_title("Total VAE Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["recon"], color="#4CAF50")
    axes[1].set_title("Reconstruction Loss"); axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["kl"], color="#9C27B0")
    axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Task 2 – VAE Training Curves")
    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "vae_loss_curves.png")
    fig.savefig(plot_path, dpi=150); plt.close(fig)
    print(f"\n  ✓ Loss curves saved → {plot_path}")

    # ── Latent interpolation experiment ───────────────────────────
    print("  Running latent interpolation experiment …")
    model.eval()
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "vae_best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    from src.generation.generate_music import generate_from_vae
    generate_from_vae(model, n_samples=8, device=device, tag="task2")

    # Visualise 2-D latent space via PCA
    _plot_latent_space(model, val_loader, device)

    print(f"\n✓ Task 2 complete. Best val loss: {best_val:.4f}")


@torch.no_grad()
def _plot_latent_space(model, loader, device, max_batches=20):
    """Project latent µ vectors onto 2-D with PCA and colour by genre."""
    from sklearn.decomposition import PCA
    from src.config import GENRES

    model.eval()
    all_mu, all_genres = [], []
    for i, (x, g) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        mu, _ = model.encoder(x)
        all_mu.append(mu.cpu().numpy())
        all_genres.append(g.numpy())

    mu_arr = np.concatenate(all_mu, axis=0)
    g_arr  = np.concatenate(all_genres, axis=0)

    pca = PCA(n_components=2)
    z2d = pca.fit_transform(mu_arr)

    colours = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for g_idx, (gname, col) in enumerate(zip(GENRES, colours)):
        mask = g_arr == g_idx
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=col, label=gname, alpha=0.6, s=15)
    ax.set_title("VAE Latent Space (PCA)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "vae_latent_pca.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ Latent space PCA saved → {path}")


if __name__ == "__main__":
    main()
