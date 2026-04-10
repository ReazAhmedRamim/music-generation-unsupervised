# src/evaluation/pitch_histogram.py
"""
Pitch Histogram Analysis for Music Generation Evaluation
=========================================================

Project formula:
  H(p, q) = Σ_{i=1}^{12} |p_i - q_i|

Provides:
  - Per-sequence pitch class distribution
  - Cross-model histogram comparison
  - Visualisation of pitch distributions by genre
"""

import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PLOTS_DIR, GENRES

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_class_histogram(token_ids: list) -> np.ndarray:
    """
    Build a normalised 12-bin pitch class histogram from token ids.
    Only pitch tokens (ids 4–91) are counted.

    Returns:
        hist: (12,) float array summing to 1.0 (or zeros if no pitches)
    """
    counts = np.zeros(12, dtype=np.float64)
    for tid in token_ids:
        if 4 <= tid < 92:
            midi_pitch  = (tid - 4) + 21
            pitch_class = midi_pitch % 12
            counts[pitch_class] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def histogram_distance(tokens_a: list, tokens_b: list) -> float:
    """
    H(p, q) = Σ_{i=1}^{12} |p_i - q_i|  (L1 distance)

    Args:
        tokens_a, tokens_b: token id sequences

    Returns:
        distance ∈ [0, 2]  (0 = identical, 2 = maximally different)
    """
    p = pitch_class_histogram(tokens_a)
    q = pitch_class_histogram(tokens_b)
    return float(np.sum(np.abs(p - q)))


def histogram_similarity(tokens_a: list, tokens_b: list) -> float:
    """Normalised similarity ∈ [0,1] (1 = identical pitch distributions)."""
    return 1.0 - histogram_distance(tokens_a, tokens_b) / 2.0


def compute_mean_histogram(sequences: list) -> np.ndarray:
    """Compute mean pitch class histogram over a list of sequences."""
    hists = np.array([pitch_class_histogram(seq) for seq in sequences])
    return hists.mean(axis=0)


# ──────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────

def plot_pitch_histograms_by_model(model_sequences: dict, save: bool = True):
    """
    Plot pitch class histograms for multiple models side-by-side.

    Args:
        model_sequences: {model_name: [token_sequence, ...]}
        save:            whether to save the figure
    """
    model_names = list(model_sequences.keys())
    n_models    = len(model_names)
    colours     = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, axes = plt.subplots(2, max(n_models // 2, 1) + 1,
                              figsize=(4 * n_models, 6), constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, (name, seqs), col in zip(axes, model_sequences.items(), colours):
        hist = compute_mean_histogram(seqs)
        bars = ax.bar(PITCH_CLASSES, hist, color=col, alpha=0.85, width=0.7)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pitch Class")
        ax.set_ylabel("Relative Frequency")
        ax.set_ylim(0, max(hist.max() * 1.3, 0.2))
        ax.grid(True, alpha=0.3, axis="y")

    # Hide extra axes
    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Pitch Class Distribution – Model Comparison", fontsize=13, fontweight="bold")

    if save:
        path = os.path.join(PLOTS_DIR, "pitch_histograms.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Pitch histogram plot saved → {path}")
    else:
        plt.show()

    return fig


def plot_pitch_histograms_by_genre(genre_sequences: dict, save: bool = True):
    """
    Plot pitch class histograms per genre (for multi-genre VAE evaluation).

    Args:
        genre_sequences: {genre_name: [token_sequence, ...]}
    """
    genre_names = list(genre_sequences.keys())
    n = len(genre_names)
    colours = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, gname, col in zip(axes, genre_names, colours):
        seqs = genre_sequences[gname]
        hist = compute_mean_histogram(seqs)
        ax.bar(PITCH_CLASSES, hist, color=col, alpha=0.85, width=0.7)
        ax.set_title(gname.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Pitch Class")
        ax.set_ylabel("Rel. Frequency")
        ax.set_ylim(0, max(hist.max() * 1.3, 0.15))
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Pitch Class Distribution by Genre (VAE Task 2)", fontsize=13, fontweight="bold")

    if save:
        path = os.path.join(PLOTS_DIR, "pitch_histograms_genre.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Genre pitch histogram saved → {path}")
    else:
        plt.show()

    return fig


def cross_model_similarity_matrix(model_sequences: dict, save: bool = True):
    """
    Compute and visualise a pairwise pitch-similarity matrix between models.
    """
    model_names = list(model_sequences.keys())
    n = len(model_names)
    matrix = np.zeros((n, n))

    hists = {name: compute_mean_histogram(seqs)
             for name, seqs in model_sequences.items()}

    for i, n1 in enumerate(model_names):
        for j, n2 in enumerate(model_names):
            dist = float(np.sum(np.abs(hists[n1] - hists[n2])))
            matrix[i, j] = 1.0 - dist / 2.0   # similarity

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(model_names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label="Pitch Similarity")
    ax.set_title("Cross-Model Pitch Class Similarity Matrix")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "pitch_similarity_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Similarity matrix saved → {path}")
    else:
        plt.show()

    return matrix


if __name__ == "__main__":
    # Demo with synthetic data
    rng = np.random.default_rng(42)
    synthetic = {
        "Random":      [[int(rng.integers(4, 92)) for _ in range(128)] for _ in range(50)],
        "Autoencoder": [[int(rng.integers(30, 70)) for _ in range(128)] for _ in range(50)],
        "VAE":         [[int(rng.integers(25, 80)) for _ in range(128)] for _ in range(50)],
        "Transformer": [[int(rng.integers(35, 75)) for _ in range(128)] for _ in range(50)],
    }
    plot_pitch_histograms_by_model(synthetic)
    cross_model_similarity_matrix(synthetic)
    print("Pitch histogram analysis complete.")
