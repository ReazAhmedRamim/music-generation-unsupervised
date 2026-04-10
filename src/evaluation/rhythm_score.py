# src/evaluation/rhythm_score.py
"""
Rhythm Diversity & Pattern Analysis
=====================================

Project formula:
  D_rhythm = #unique_durations / #total_notes
  R        = #repeated_patterns / #total_patterns

Provides:
  - Per-sequence rhythm diversity
  - Repetition ratio
  - Inter-onset interval (IOI) analysis
  - Visualisation of rhythmic distributions
"""

import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PLOTS_DIR, GENRES, REST_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


# ──────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────

def extract_note_events(token_ids) -> List[dict]:
    """
    Parse a token sequence into note events with onset and duration.

    Returns:
        list of dicts with keys: onset, pitch, duration
    """
    # Accept numpy arrays, tensors, or plain lists
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, (list, tuple)):
        token_ids = list(token_ids)

    events      = []
    onset       = 0
    current_dur = 0
    current_pit = None

    for tid in token_ids:
        if 4 <= tid < 92:                          # pitch token
            if current_pit is not None:
                events.append({"onset": onset - current_dur,
                                "pitch": current_pit,
                                "duration": current_dur})
            current_pit = tid
            current_dur = 1
        elif tid == REST_TOKEN:
            if current_pit is not None:
                current_dur += 1
            onset += 1
            continue
        elif tid in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            if current_pit is not None:
                events.append({"onset": onset - current_dur,
                                "pitch": current_pit,
                                "duration": current_dur})
                current_pit = None
                current_dur = 0
        onset += 1

    if current_pit is not None:
        events.append({"onset": onset - current_dur,
                        "pitch": current_pit,
                        "duration": current_dur})
    return events


def rhythm_diversity_score(token_ids: List[int]) -> float:
    """
    D_rhythm = #unique_durations / #total_notes

    Higher is better – indicates more varied note lengths.
    """
    events = extract_note_events(token_ids)
    if not events:
        return 0.0
    durations = [e["duration"] for e in events]
    return len(set(durations)) / len(durations)


def repetition_ratio(token_ids: List[int], pattern_len: int = 4) -> float:
    """
    R = #repeated_patterns / #total_patterns

    A pitch n-gram of length `pattern_len` is counted as "repeated"
    if it appears more than once in the sequence.
    """
    pitch_seq = [t for t in token_ids if 4 <= t < 92]
    if len(pitch_seq) < pattern_len:
        return 0.0

    freq = Counter(
        tuple(pitch_seq[i:i + pattern_len])
        for i in range(len(pitch_seq) - pattern_len + 1)
    )
    total    = len(freq)
    repeated = sum(1 for c in freq.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def inter_onset_intervals(token_ids: List[int]) -> List[int]:
    """
    Compute the inter-onset interval (IOI) sequence.
    IOI[i] = onset[i+1] - onset[i]
    """
    events = extract_note_events(token_ids)
    if len(events) < 2:
        return []
    onsets = [e["onset"] for e in events]
    return [onsets[i+1] - onsets[i] for i in range(len(onsets) - 1)]


def mean_ioi_entropy(sequences: List[List[int]]) -> float:
    """
    Compute mean IOI entropy across sequences.
    Higher entropy → more rhythmically varied.
    """
    entropies = []
    for seq in sequences:
        iois = inter_onset_intervals(seq)
        if not iois:
            continue
        counts = np.array(list(Counter(iois).values()), dtype=float)
        probs  = counts / counts.sum()
        ent    = -np.sum(probs * np.log(probs + 1e-9))
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0


def density_score(token_ids: List[int]) -> float:
    """
    Note density = #notes / sequence_length.
    """
    n_notes = sum(1 for t in token_ids if 4 <= t < 92)
    return n_notes / max(len(token_ids), 1)


# ──────────────────────────────────────────────────
# Batch utilities
# ──────────────────────────────────────────────────

def evaluate_rhythm(sequences: List[List[int]]) -> Dict[str, float]:
    """
    Comprehensive rhythm evaluation for a list of generated sequences.

    Returns dict with:
      - mean_diversity:  mean D_rhythm
      - mean_repetition: mean R
      - mean_ioi_entropy
      - mean_density
      - std_diversity
    """
    diversities  = [rhythm_diversity_score(s)  for s in sequences]
    repetitions  = [repetition_ratio(s)         for s in sequences]
    densities    = [density_score(s)            for s in sequences]
    ioi_ent      = mean_ioi_entropy(sequences)

    return {
        "mean_diversity":   float(np.mean(diversities)),
        "std_diversity":    float(np.std(diversities)),
        "mean_repetition":  float(np.mean(repetitions)),
        "mean_density":     float(np.mean(densities)),
        "mean_ioi_entropy": ioi_ent,
    }


# ──────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────

def plot_rhythm_comparison(model_sequences: Dict[str, List[List[int]]], save: bool = True):
    """
    Bar chart comparing rhythm metrics across models.
    """
    model_names = list(model_sequences.keys())
    metrics_list = [evaluate_rhythm(seqs) for seqs in model_sequences.values()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colours = plt.cm.Set2(np.linspace(0, 0.9, len(model_names)))

    metric_keys   = ["mean_diversity", "mean_repetition", "mean_ioi_entropy"]
    metric_labels = ["Rhythm Diversity (D_rhythm)", "Repetition Ratio (R)", "IOI Entropy"]

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        vals = [m[mkey] for m in metrics_list]
        errs = [m.get("std_diversity", 0) if mkey == "mean_diversity" else 0
                for m in metrics_list]
        bars = ax.bar(model_names, vals, yerr=errs, color=colours,
                      width=0.6, capsize=4, error_kw={"linewidth": 1.5})
        ax.set_title(mlabel, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(vals) * 1.4 + 0.05)
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Rhythm Analysis – Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "rhythm_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Rhythm comparison plot saved → {path}")
    else:
        plt.show()

    return fig


def plot_duration_distribution(sequences: List[List[int]], title: str = "Duration Distribution",
                                save_name: str = "duration_dist.png", save: bool = True):
    """
    Plot histogram of note durations for a set of sequences.
    """
    all_durations = []
    for seq in sequences:
        events = extract_note_events(seq)
        all_durations.extend(e["duration"] for e in events)

    if not all_durations:
        print("  [warn] No note events found for duration distribution.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    max_dur = min(max(all_durations), 32)
    ax.hist(all_durations, bins=range(1, max_dur + 2), density=True,
            color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Duration (steps)")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, save_name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Duration distribution saved → {path}")
    else:
        plt.show()


def plot_piano_roll_preview(token_ids: List[int], title: str = "Piano Roll Preview",
                             save_name: str = "piano_roll_preview.png", save: bool = True):
    """
    Visualise a token sequence as a simple piano-roll image.
    """
    events = extract_note_events(token_ids)
    if not events:
        print("  [warn] No events to plot.")
        return

    max_time = max(e["onset"] + e["duration"] for e in events)
    min_pitch = min(e["pitch"] for e in events)
    max_pitch = max(e["pitch"] for e in events)
    pitch_range = max(max_pitch - min_pitch + 1, 10)

    roll = np.zeros((pitch_range, max(max_time, 1)))
    for e in events:
        p_idx  = e["pitch"] - min_pitch
        t_start = e["onset"]
        t_end   = e["onset"] + e["duration"]
        roll[p_idx, t_start:t_end] = 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(roll, aspect="auto", origin="lower", cmap="Blues",
              extent=[0, roll.shape[1], min_pitch, max_pitch + 1])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, save_name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Piano roll preview saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Demo with synthetic sequences
    rng = np.random.default_rng(42)

    def make_seq(structured=False):
        if structured:
            pitches = list(range(50, 70))
            tokens  = [BOS_TOKEN]
            for _ in range(30):
                tokens.append(np.random.choice(pitches))
                tokens.extend([REST_TOKEN] * np.random.choice([1, 2, 4]))
            tokens.append(EOS_TOKEN)
        else:
            tokens = [int(rng.integers(4, 92)) for _ in range(128)]
        return tokens

    model_seqs = {
        "Random":      [make_seq(False) for _ in range(50)],
        "Autoencoder": [make_seq(True)  for _ in range(50)],
        "Transformer": [make_seq(True)  for _ in range(50)],
    }
    plot_rhythm_comparison(model_seqs)
    plot_piano_roll_preview(make_seq(True), title="Sample Piano Roll – Transformer Output")
    print("Rhythm analysis complete.")
