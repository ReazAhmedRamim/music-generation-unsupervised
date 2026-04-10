# notebooks/preprocessing.py
"""
Preprocessing Notebook – Interactive walkthrough of the full
MIDI preprocessing pipeline with visualisations.

Equivalent to preprocessing.ipynb (run as a script for automation).

Usage:
    python notebooks/preprocessing.py
"""

import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    RAW_MIDI_DIR, PROCESSED_DIR, SPLIT_DIR, PLOTS_DIR,
    SEQUENCE_LENGTH, PITCH_RANGE, NUM_PITCHES, STEPS_PER_BAR,
    VOCAB_SIZE, GENRES, SEED
)
from src.preprocessing.tokenizer import MusicTokenizer

np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Section 1: Tokenizer walkthrough
# ──────────────────────────────────────────────────

def demo_tokenizer():
    print("\n" + "=" * 55)
    print("  Section 1: MusicTokenizer")
    print("=" * 55)

    tok = MusicTokenizer()
    print(tok)

    # Encode a simple C-major arpeggio
    notes = [(60, 80, 2), (64, 75, 2), (67, 70, 2), (72, 80, 4)]
    tokens = tok.encode_sequence(notes)
    print(f"\n  C-major arpeggio → tokens: {tokens[:20]} …")

    # Decode back
    decoded = tok.decode_sequence(tokens)
    print(f"  Decoded events:  {decoded}")

    # Padding
    padded = tok.pad_sequence(tokens, SEQUENCE_LENGTH)
    print(f"  Padded to {SEQUENCE_LENGTH}: first 10 = {padded[:10]}, last 5 = {padded[-5:]}")


# ──────────────────────────────────────────────────
# Section 2: Piano roll demo
# ──────────────────────────────────────────────────

def demo_piano_roll():
    print("\n" + "=" * 55)
    print("  Section 2: Synthetic Piano Roll")
    print("=" * 55)

    # Generate a synthetic C-major scale piano roll
    roll = np.zeros((NUM_PITCHES, 64), dtype=np.float32)
    c_major = [0, 2, 4, 5, 7, 9, 11, 12]  # intervals from C4 (MIDI 60)
    c4_idx  = 60 - PITCH_RANGE[0]

    for i, interval in enumerate(c_major):
        p_idx = c4_idx + interval
        if 0 <= p_idx < NUM_PITCHES:
            roll[p_idx, i*8:(i+1)*8] = 0.8

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(roll[c4_idx-2:c4_idx+16, :], aspect="auto", origin="lower",
              cmap="Blues", interpolation="nearest")
    ax.set_title("Synthetic C-Major Scale – Piano Roll (MIDI 58–75, 64 steps)")
    ax.set_xlabel("Time Step (1/16 bar)")
    ax.set_ylabel("Relative Pitch Index")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "preprocessing_piano_roll_demo.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ Piano roll demo saved → {path}")


# ──────────────────────────────────────────────────
# Section 3: Token distribution analysis
# ──────────────────────────────────────────────────

def demo_token_distribution():
    print("\n" + "=" * 55)
    print("  Section 3: Token Distribution (Synthetic Data)")
    print("=" * 55)

    import pickle
    path = os.path.join(SPLIT_DIR, "train.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_tokens = []
        for arr, _ in data[:500]:
            all_tokens.extend(arr.tolist())
        print(f"  Loaded {len(data)} sequences, {len(all_tokens)} tokens.")
    else:
        print("  No preprocessed data. Generating synthetic tokens …")
        all_tokens = np.random.randint(0, VOCAB_SIZE, size=50000).tolist()

    token_arr = np.array(all_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Full vocab distribution
    counts = np.bincount(token_arr, minlength=VOCAB_SIZE)
    axes[0].bar(range(VOCAB_SIZE), counts, width=1.0, color="#2196F3", alpha=0.75)
    axes[0].set_xlabel("Token ID"); axes[0].set_ylabel("Count")
    axes[0].set_title("Full Token Vocabulary Distribution")
    axes[0].axvline(x=3.5, color="red",    linestyle="--", label="Special|Pitch boundary")
    axes[0].axvline(x=91.5, color="orange", linestyle="--", label="Pitch|Velocity boundary")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis="y")

    # Pitch-only distribution
    pitch_tokens = token_arr[(token_arr >= 4) & (token_arr < 92)]
    midi_pitches  = pitch_tokens - 4 + PITCH_RANGE[0]
    axes[1].hist(midi_pitches, bins=88, color="#4CAF50", alpha=0.8,
                 range=(PITCH_RANGE[0], PITCH_RANGE[1]))
    axes[1].set_xlabel("MIDI Pitch"); axes[1].set_ylabel("Count")
    axes[1].set_title("Pitch Token Distribution (MIDI range)")
    axes[1].axvline(x=60, color="red", linestyle="--", label="C4 (Middle C)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Token Distribution Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "preprocessing_token_dist.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ Token distribution saved → {path}")


# ──────────────────────────────────────────────────
# Section 4: Genre distribution
# ──────────────────────────────────────────────────

def demo_genre_distribution():
    print("\n" + "=" * 55)
    print("  Section 4: Genre Distribution in Dataset")
    print("=" * 55)

    import pickle
    path = os.path.join(SPLIT_DIR, "train.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        genre_counts = np.zeros(len(GENRES))
        for _, g in data:
            genre_counts[g] += 1
    else:
        genre_counts = np.random.randint(100, 500, size=len(GENRES))

    colours = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    bars = axes[0].bar(GENRES, genre_counts, color=colours, width=0.6)
    axes[0].set_title("Training Samples per Genre", fontweight="bold")
    axes[0].set_ylabel("Number of Sequences")
    for bar, v in zip(bars, genre_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{int(v)}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Pie chart
    axes[1].pie(genre_counts, labels=GENRES, colors=colours, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Genre Proportion", fontweight="bold")

    fig.suptitle("Dataset Genre Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "preprocessing_genre_dist.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ Genre distribution saved → {path}")


# ──────────────────────────────────────────────────
# Section 5: Sequence length distribution
# ──────────────────────────────────────────────────

def demo_sequence_stats():
    print("\n" + "=" * 55)
    print("  Section 5: Sequence Statistics")
    print("=" * 55)

    import pickle
    path = os.path.join(SPLIT_DIR, "train.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # All sequences should be exactly SEQUENCE_LENGTH due to windowing
        lengths = [len(arr) for arr, _ in data[:2000]]
        pitch_densities = [
            sum(1 for t in arr if 4 <= t < 92) / SEQUENCE_LENGTH
            for arr, _ in data[:500]
        ]
    else:
        lengths = [SEQUENCE_LENGTH] * 1000
        pitch_densities = np.random.uniform(0.1, 0.6, 500).tolist()

    print(f"  Sequence length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}")
    print(f"  Pitch density:   mean={np.mean(pitch_densities):.3f}, "
          f"std={np.std(pitch_densities):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(lengths, bins=20, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].set_title("Sequence Length Distribution"); axes[0].set_xlabel("Length (tokens)")
    axes[0].set_ylabel("Count"); axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(pitch_densities, bins=30, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].set_title("Note Density Distribution"); axes[1].set_xlabel("Pitch Tokens / Total Tokens")
    axes[1].set_ylabel("Count"); axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sequence Statistics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "preprocessing_seq_stats.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ Sequence stats saved → {path}")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Preprocessing Notebook")
    print("=" * 55)

    demo_tokenizer()
    demo_piano_roll()
    demo_token_distribution()
    demo_genre_distribution()
    demo_sequence_stats()

    print("\n✓ Preprocessing notebook complete. Plots saved to outputs/plots/")
