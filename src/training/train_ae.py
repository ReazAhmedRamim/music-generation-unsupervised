# src/training/train_ae.py
"""
Task 1 – Training Script for LSTM Autoencoder
==============================================

Algorithm 1 (from project spec):
  Require: MIDI dataset D, Encoder f_φ, Decoder g_θ, Epochs E, lr η
  1. Init params φ, θ
  2. for epoch = 1 to E:
       for each batch X in D:
         z    = f_φ(X)
         X̂   = g_θ(z)
         L_AE = ‖X - X̂‖²
         (φ,θ) ← (φ,θ) - η ∇L_AE
  3. Generate music by sampling latent codes z

Usage:
    python src/training/train_ae.py
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
    BATCH_SIZE, AE_EPOCHS, AE_LEARNING_RATE, DEVICE, SEED,
    VOCAB_SIZE, SEQUENCE_LENGTH
)
from src.models.autoencoder import LSTMAutoencoder

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────

class MIDIDataset(Dataset):
    def __init__(self, data, single_genre: int = None):
        """
        data: list of (token_array, genre_label)
        single_genre: if set, filter to that genre only (Task 1)
        """
        if single_genre is not None:
            data = [(arr, g) for arr, g in data if g == single_genre]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr, genre = self.data[idx]
        return torch.tensor(arr, dtype=torch.long), torch.tensor(genre, dtype=torch.long)


def load_split(name: str):
    path = os.path.join(SPLIT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"  [warn] {path} not found – generating synthetic data …")
        from src.preprocessing.midi_parser import _synthetic_segments
        return _synthetic_segments(genre_label=0, n=500)
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        logits, z = model(x)
        loss = LSTMAutoencoder.reconstruction_loss(logits, x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        logits, _ = model(x)
        loss = LSTMAutoencoder.reconstruction_loss(logits, x)
        total_loss += loss.item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (Task 1: single genre = classical = 0)
    train_data = load_split("train")
    val_data   = load_split("val")

    train_ds = MIDIDataset(train_data, single_genre=0)
    val_ds   = MIDIDataset(val_data,   single_genre=0)

    if len(train_ds) == 0:
        print("  No single-genre data – falling back to all genres.")
        train_ds = MIDIDataset(train_data)
        val_ds   = MIDIDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model
    model = LSTMAutoencoder().to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val = float("inf")

    print("\n" + "=" * 50)
    print("  Training LSTM Autoencoder (Task 1)")
    print("=" * 50)

    for epoch in range(1, AE_EPOCHS + 1):
        tr_loss  = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:3d}/{AE_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(CHECKPOINTS_DIR, "ae_best.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": val_loss}, ckpt_path)

    # ── Plot reconstruction loss curve ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="#2196F3")
    ax.plot(val_losses,   label="Val Loss",   color="#FF5722")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Task 1 – LSTM Autoencoder: Reconstruction Loss Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(PLOTS_DIR, "ae_loss_curve.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"\n  ✓ Loss curve saved → {curve_path}")

    # ── Generate 5 sample sequences ───────────────────────────────
    print("\n  Generating 5 music samples …")
    model.eval()
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "ae_best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    from src.generation.generate_music import generate_from_autoencoder
    generate_from_autoencoder(model, n_samples=5, device=device, tag="task1")

    print(f"\n✓ Task 1 training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
