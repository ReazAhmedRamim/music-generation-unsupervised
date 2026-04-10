# notebooks/baseline_markov.py
"""
Baseline: Markov Chain Music Model
====================================

Implements a pitch-level n-gram Markov chain music generator.
This serves as Baseline 2 in the project comparison.

Compare against:
  - Baseline 1: Random Note Generator (Naive)
  - Baseline 2: Markov Chain (this file)
  - Task 1-4:   Neural models

Usage:
    python notebooks/baseline_markov.py
"""

import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    SPLIT_DIR, PLOTS_DIR, MIDI_OUT_DIR, SEQUENCE_LENGTH,
    BOS_TOKEN, EOS_TOKEN, REST_TOKEN, GENRES, SEED
)

np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Markov Chain Model
# ──────────────────────────────────────────────────

class MarkovChainMusicModel:
    """
    N-th order pitch Markov chain trained on token sequences.

    Transition: P(x_t | x_{t-1}, ..., x_{t-n+1})
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.start_states = []

    def train(self, sequences: list):
        """
        Train on a list of token id sequences.

        Args:
            sequences: list of lists of int token ids
        """
        for seq in sequences:
            pitch_seq = [t for t in seq if 4 <= t < 92]
            if len(pitch_seq) < self.order + 1:
                continue
            for i in range(len(pitch_seq) - self.order):
                state = tuple(pitch_seq[i:i + self.order])
                next_tok = pitch_seq[i + self.order]
                self.transitions[state][next_tok] += 1
            if len(pitch_seq) >= self.order:
                self.start_states.append(tuple(pitch_seq[:self.order]))

        print(f"  Markov chain trained: {len(self.transitions)} states, "
              f"{sum(len(v) for v in self.transitions.values())} transitions")

    def generate(self, length: int = SEQUENCE_LENGTH, temperature: float = 1.0) -> list:
        """
        Sample a pitch sequence of given length.

        Args:
            length:      number of pitches to generate
            temperature: softmax temperature (1.0 = uniform Markov)

        Returns:
            token_ids: list of int pitch token ids
        """
        if not self.start_states:
            # Random fallback if not trained
            return [int(np.random.randint(4, 92)) for _ in range(length)]

        state = list(self.start_states[np.random.randint(len(self.start_states))])
        seq   = list(state)

        for _ in range(length - self.order):
            key = tuple(state)
            if key not in self.transitions:
                # Restart from random known state
                new_state = list(self.start_states[np.random.randint(len(self.start_states))])
                state = new_state
                seq.extend(state)
                continue

            counts = self.transitions[key]
            tokens = list(counts.keys())
            freqs  = np.array(list(counts.values()), dtype=float)

            # Apply temperature
            log_freqs = np.log(freqs + 1e-9) / temperature
            log_freqs -= log_freqs.max()
            probs = np.exp(log_freqs)
            probs /= probs.sum()

            next_tok = np.random.choice(tokens, p=probs)
            seq.append(next_tok)
            state = state[1:] + [next_tok]

        return seq[:length]

    def perplexity(self, test_sequences: list) -> float:
        """
        Compute approximate perplexity on test sequences.
        """
        total_log_p = 0.0
        total_n     = 0

        for seq in test_sequences:
            pitch_seq = [t for t in seq if 4 <= t < 92]
            for i in range(self.order, len(pitch_seq)):
                state   = tuple(pitch_seq[i - self.order:i])
                next_t  = pitch_seq[i]
                counts  = self.transitions.get(state, {})
                total_c = sum(counts.values())
                if total_c == 0:
                    p = 1 / 88  # uniform fallback
                else:
                    p = counts.get(next_t, 0) / total_c
                    p = max(p, 1e-9)
                total_log_p += np.log(p)
                total_n += 1

        if total_n == 0:
            return float("inf")
        return float(np.exp(-total_log_p / total_n))


# ──────────────────────────────────────────────────
# Random Baseline
# ──────────────────────────────────────────────────

class RandomNoteGenerator:
    """Naive random note generator – Baseline 1."""

    def generate(self, length: int = SEQUENCE_LENGTH) -> list:
        return [int(np.random.randint(4, 92)) for _ in range(length)]

    def __repr__(self):
        return "RandomNoteGenerator(uniform over 88 pitches)"


# ──────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────

def load_training_data():
    import pickle
    path = os.path.join(SPLIT_DIR, "train.pkl")
    if not os.path.exists(path):
        print("  [warn] No preprocessed data found – generating synthetic sequences.")
        return [[int(np.random.randint(4, 92)) for _ in range(SEQUENCE_LENGTH)]
                for _ in range(1000)]
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [arr.tolist() for arr, _ in data]


def evaluate_baselines(sequences: list):
    from src.evaluation.metrics import (
        batch_rhythm_diversity, repetition_ratio, batch_pitch_similarity
    )

    rd = batch_rhythm_diversity(sequences)
    rr = float(np.mean([repetition_ratio(s) for s in sequences]))
    ref = [[int(np.random.randint(4, 92)) for _ in range(SEQUENCE_LENGTH)]
           for _ in range(len(sequences))]
    ps = batch_pitch_similarity(sequences, ref)
    return {"rhythm_diversity": rd, "repetition_ratio": rr, "pitch_similarity": ps}


def plot_baseline_comparison(random_metrics, markov_metrics, save: bool = True):
    metrics = ["rhythm_diversity", "repetition_ratio", "pitch_similarity"]
    labels  = ["Rhythm Diversity", "Repetition Ratio", "Pitch Similarity"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [random_metrics[metric], markov_metrics[metric]]
        bars = ax.bar(["Random", "Markov"], vals, color=["#FF5722", "#607D8B"], width=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 + 0.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Baseline Models – Metric Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "baseline_comparison.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  ✓ Baseline comparison saved → {path}")
    else:
        plt.show()


def plot_markov_transition_heatmap(model: MarkovChainMusicModel,
                                    pitch_range=(50, 70), save: bool = True):
    """Visualise pitch transition probabilities (bigram)."""
    if model.order != 1:
        print("  [info] Transition heatmap only shown for order-1 Markov chain.")
        return

    ps = range(pitch_range[0], pitch_range[1])
    n  = len(ps)
    mat = np.zeros((n, n))

    for i, p1 in enumerate(ps):
        state  = (p1,)
        counts = model.transitions.get(state, {})
        total  = sum(counts.values()) + 1e-9
        for j, p2 in enumerate(ps):
            mat[i, j] = counts.get(p2, 0) / total

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xlabel("Next Pitch Token"); ax.set_ylabel("Current Pitch Token")
    ax.set_title("Markov Chain Transition Probabilities", fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "markov_transition.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  ✓ Markov transition heatmap saved → {path}")


def main():
    print("=" * 60)
    print("  Baseline Models: Random + Markov Chain")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading training data …")
    train_seqs = load_training_data()
    print(f"      {len(train_seqs)} sequences loaded.")

    # Random baseline
    print("\n[2/4] Evaluating Random Note Generator …")
    random_gen  = RandomNoteGenerator()
    random_seqs = [random_gen.generate() for _ in range(200)]
    random_met  = evaluate_baselines(random_seqs)
    print(f"      Rhythm Diversity: {random_met['rhythm_diversity']:.3f}")
    print(f"      Repetition Ratio: {random_met['repetition_ratio']:.3f}")
    print(f"      Pitch Similarity: {random_met['pitch_similarity']:.3f}")

    # Markov chain
    print("\n[3/4] Training & evaluating Markov Chain (order=2) …")
    markov2 = MarkovChainMusicModel(order=2)
    markov2.train(train_seqs)
    markov_seqs = [markov2.generate() for _ in range(200)]
    markov_met  = evaluate_baselines(markov_seqs)
    print(f"      Rhythm Diversity: {markov_met['rhythm_diversity']:.3f}")
    print(f"      Repetition Ratio: {markov_met['repetition_ratio']:.3f}")
    print(f"      Pitch Similarity: {markov_met['pitch_similarity']:.3f}")

    # Perplexity approximation
    import pickle
    test_path = os.path.join(SPLIT_DIR, "test.pkl")
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)
        test_seqs = [arr.tolist() for arr, _ in test_data[:200]]
        ppl = markov2.perplexity(test_seqs)
        print(f"      Markov Chain Perplexity (approx.): {ppl:.2f}")

    print("\n[4/4] Generating comparison plots …")
    plot_baseline_comparison(random_met, markov_met)

    # Export sample MIDI files
    from src.generation.generate_music import _tokens_to_midi
    print("\n  Exporting baseline MIDI samples …")
    for i in range(3):
        path = os.path.join(MIDI_OUT_DIR, f"baseline_random_{i+1:02d}.mid")
        _tokens_to_midi(random_seqs[i], path, genre_label=0)
    for i in range(3):
        path = os.path.join(MIDI_OUT_DIR, f"baseline_markov_{i+1:02d}.mid")
        _tokens_to_midi(markov_seqs[i], path, genre_label=0)

    print("\n✓ Baseline evaluation complete.")
    return random_met, markov_met


if __name__ == "__main__":
    main()
