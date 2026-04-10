# src/preprocessing/midi_parser.py
"""
MIDI Parser – converts raw MIDI files into token sequences and saves
preprocessed numpy arrays ready for training.

Supported datasets:
  • MAESTRO  (Classical Piano)
  • Lakh MIDI (Multi-Genre)
  • Groove   (Jazz / Drums / Rhythm)

Usage:
    python src/preprocessing/midi_parser.py
"""

import os
import sys
import glob
import numpy as np
import pretty_midi
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    RAW_MIDI_DIR, PROCESSED_DIR, SPLIT_DIR,
    SEQUENCE_LENGTH, PITCH_RANGE, VELOCITY_BINS,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, REST_TOKEN,
    NUM_PITCHES, VOCAB_SIZE, STEPS_PER_BAR, SEED
)

np.random.seed(SEED)

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def pitch_to_token(pitch: int) -> int:
    """Map MIDI pitch [21-108] → token [4, 91]."""
    return int(np.clip(pitch - PITCH_RANGE[0], 0, NUM_PITCHES - 1)) + 4  # offset past specials

def velocity_to_bin(velocity: int) -> int:
    """Quantise velocity [0-127] into VELOCITY_BINS bins."""
    return int(velocity / 128 * VELOCITY_BINS)

def midi_to_events(midi_path: str,allow_drums = False):
    """
    Parse a MIDI file and return a list of (pitch_token, velocity_bin, duration_steps) tuples.
    Duration is quantised to STEPS_PER_BAR resolution.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    tempo = pm.estimate_tempo()
    seconds_per_step = 60.0 / (tempo * STEPS_PER_BAR)
    events = []
    for instrument in pm.instruments:
        if instrument.is_drum and not allow_drums:
            continue
        for note in instrument.notes:
            pitch_tok  = pitch_to_token(note.pitch)
            vel_bin    = velocity_to_bin(note.velocity)
            dur_steps  = max(1, int(round((note.end - note.start) / seconds_per_step)))
            onset_step = int(round(note.start / seconds_per_step))
            events.append((onset_step, pitch_tok, vel_bin, dur_steps))

    # Sort by onset
    events.sort(key=lambda e: e[0])
    return events


def events_to_token_sequence(events):
    """
    Convert (onset, pitch_tok, vel_bin, dur) events to a flat token sequence.
    Each note → [pitch_tok, REST_TOKEN × (dur-1)] so timing is encoded implicitly.
    """
    if not events:
        return []
    tokens = [BOS_TOKEN]
    prev_onset = 0
    for onset, pitch_tok, vel_bin, dur in events:
        gap = onset - prev_onset
        # encode rests
        tokens.extend([REST_TOKEN] * min(gap, 16))
        tokens.append(pitch_tok)
        prev_onset = onset + dur
    tokens.append(EOS_TOKEN)
    return tokens


def segment_sequence(tokens, seq_len=SEQUENCE_LENGTH):
    """Slide a fixed-length window over the token sequence (no overlap)."""
    segments = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        seg = tokens[i:i + seq_len]
        segments.append(seg)
    return segments


# ──────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────

def process_dataset(genre_label: int, midi_glob: str, max_files: int = 500):
    """Process all MIDI files matching glob pattern for a given genre."""
    files = glob.glob(midi_glob, recursive=True)[:max_files]
    all_segments = []

    for fpath in tqdm(files, desc=f"Genre {genre_label}"):
        allow_drums = (genre_label == 1)  # jazz = 1
        events = midi_to_events(fpath, allow_drums=allow_drums)
        tokens = events_to_token_sequence(events)
        segs   = segment_sequence(tokens)
        for seg in segs:
            arr = np.array(seg, dtype=np.int32)
            # Pad if shorter
            if len(arr) < SEQUENCE_LENGTH:
                arr = np.pad(arr, (0, SEQUENCE_LENGTH - len(arr)), constant_values=PAD_TOKEN)
            all_segments.append((arr, genre_label))

    return all_segments


def build_dataset():
    from src.config import GENRES

    all_data = []
    for g_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(RAW_MIDI_DIR, genre)

        # Search both .mid and .midi extensions
        segs = []
        for ext in ["*.mid", "*.midi"]:
            midi_glob = os.path.join(genre_dir, "**", ext)
            found = process_dataset(g_idx, midi_glob, max_files=500)
            segs.extend(found)

        if segs:
            print(f"  [{genre}] {len(segs)} segments found.")
        else:
            print(f"  [{genre}] No MIDI found – generating synthetic placeholder data.")
            segs = _synthetic_segments(g_idx, n=200)
        all_data.extend(segs)

    np.random.shuffle(all_data)
    return all_data


def _synthetic_segments(genre_label: int, n: int = 200):
    """Generate simple synthetic token sequences (for demo / CI purposes)."""
    segs = []
    for _ in range(n):
        tokens = [BOS_TOKEN]
        for _ in range(SEQUENCE_LENGTH - 2):
            tokens.append(np.random.randint(4, VOCAB_SIZE))
        tokens.append(EOS_TOKEN)
        arr = np.array(tokens, dtype=np.int32)
        segs.append((arr, genre_label))
    return segs


def train_test_split(all_data, test_ratio=0.1, val_ratio=0.1):
    n = len(all_data)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test   = all_data[:n_test]
    val    = all_data[n_test:n_test + n_val]
    train  = all_data[n_test + n_val:]
    return train, val, test


def save_split(train, val, test):
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(SPLIT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(subset, f)
        print(f"  Saved {len(subset)} samples → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  MIDI Preprocessing Pipeline")
    print("=" * 60)

    print("\n[1/3] Building dataset from raw MIDI …")
    all_data = build_dataset()
    print(f"      Total segments: {len(all_data)}")

    print("\n[2/3] Splitting into train / val / test …")
    train, val, test = train_test_split(all_data)
    save_split(train, val, test)

    print("\n[3/3] Saving vocabulary metadata …")
    meta = {"vocab_size": VOCAB_SIZE, "seq_len": SEQUENCE_LENGTH,
            "pad": PAD_TOKEN, "bos": BOS_TOKEN, "eos": EOS_TOKEN, "rest": REST_TOKEN}
    with open(os.path.join(PROCESSED_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n✓ Preprocessing complete.")
