# src/training/train_transformer.py
"""
Task 3 – Training Script for Transformer Music Generator
=========================================================

Algorithm 3 (from project spec):
  Require: Tokenized MIDI dataset D={x_1,...,x_T}, params θ, Epochs E
  1. Init Transformer p_θ
  2. for epoch = 1 to E:
       for each sequence X in D:
         for t = 1 to T:
           predict: p_θ(x_t | x_{<t})
         L_TR = -Σ_t log p_θ(x_t | x_{<t})
         θ ← θ - η ∇L_TR
  3. Generate long compositions by iterative sampling:
       x_t ~ p_θ(x_t | x_{<t})

Usage:
    python src/training/train_transformer.py
"""

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    SPLIT_DIR, CHECKPOINTS_DIR, PLOTS_DIR,
    BATCH_SIZE, TF_EPOCHS, TF_LEARNING_RATE, DEVICE, SEED,
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES, BOS_TOKEN
)
from src.models.transformer import MusicTransformer

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────

class TransformerMIDIDataset(Dataset):
    """Returns (x_input, x_target, genre_id) for autoregressive training."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr, genre = self.data[idx]
        x   = torch.tensor(arr,        dtype=torch.long)
        gen = torch.tensor(genre,       dtype=torch.long)
        # Input: x[0..T-2], Target: x[1..T-1]
        return x[:-1], x[1:], gen


def load_split(name):
    path = os.path.join(SPLIT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        from src.preprocessing.midi_parser import _synthetic_segments
        data = []
        for g in range(NUM_GENRES):
            data.extend(_synthetic_segments(g, n=200))
        return data
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────
# Learning-rate scheduler: Noam (Transformer warm-up)
# ──────────────────────────────────────────────────

class NoamScheduler:
    """
    L_rate = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer   = optimizer
        self.d_model     = d_model
        self.warmup      = warmup_steps
        self.step_num    = 0
        self._lr         = 0.0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup ** -1.5
        )
        self._lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @property
    def lr(self):
        return self._lr


# ──────────────────────────────────────────────────
# Training / eval loops
# ──────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for x_in, x_tgt, genre_ids in loader:
        x_in, x_tgt, genre_ids = (
            x_in.to(device), x_tgt.to(device), genre_ids.to(device)
        )
        optimizer.zero_grad()

        logits = model(x_in, genre_ids)           # (B, T-1, V)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            x_tgt[:, :T].reshape(-1),
            ignore_index=0
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x_in, x_tgt, genre_ids in loader:
        x_in, x_tgt, genre_ids = (
            x_in.to(device), x_tgt.to(device), genre_ids.to(device)
        )
        logits = model(x_in, genre_ids)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            x_tgt[:, :T].reshape(-1),
            ignore_index=0
        )
        total_loss += loss.item()
    avg = total_loss / len(loader)
    return avg, math.exp(avg)   # (NLL, perplexity)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = load_split("train")
    val_data   = load_split("val")

    train_ds = TransformerMIDIDataset(train_data)
    val_ds   = TransformerMIDIDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = MusicTransformer().to(device)
    print(model)

    # Separate weight-decay groups
    decay_params     = [p for n, p in model.named_parameters() if "bias" not in n and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if "bias" in n or p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 1e-2},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=TF_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9
    )
    from src.config import TF_D_MODEL
    scheduler = NoamScheduler(optimizer, d_model=TF_D_MODEL, warmup_steps=4000)

    history = {"train_loss": [], "val_loss": [], "perplexity": []}
    best_ppl = float("inf")

    print("\n" + "=" * 50)
    print("  Training Transformer (Task 3)")
    print("=" * 50)

    for epoch in range(1, TF_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, ppl = evaluate(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["perplexity"].append(ppl)

        print(f"  Epoch {epoch:3d}/{TF_EPOCHS} | "
              f"Train NLL: {tr_loss:.4f} | Val NLL: {val_loss:.4f} | "
              f"PPL: {ppl:.2f} | LR: {scheduler.lr:.6f}")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": val_loss, "perplexity": ppl},
                       os.path.join(CHECKPOINTS_DIR, "transformer_best.pt"))

    # ── Plots ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train NLL", color="#2196F3")
    ax1.plot(history["val_loss"],   label="Val NLL",   color="#FF5722")
    ax1.set_title("Autoregressive NLL Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["perplexity"], color="#9C27B0")
    ax2.set_title("Validation Perplexity"); ax2.set_xlabel("Epoch")
    ax2.axhline(y=best_ppl, linestyle="--", color="#FF5722", label=f"Best: {best_ppl:.2f}")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Task 3 – Transformer Training")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "transformer_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\n  ✓ Plots saved → {path}")

    # ── Generate 10 long compositions ─────────────────────────────
    model.eval()
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "transformer_best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    from src.generation.generate_music import generate_from_transformer
    generate_from_transformer(model, n_samples=10, device=device, tag="task3")

    print(f"\n✓ Task 3 complete. Best perplexity: {best_ppl:.2f}")


if __name__ == "__main__":
    main()
