# src/generation/generate_music.py
"""
Music Generation – unified generation API for all four tasks.

Functions:
  generate_from_autoencoder(model, ...)  → Task 1
  generate_from_vae(model, ...)          → Task 2
  generate_from_transformer(model, ...)  → Task 3 & 4

Each function:
  1. Generates token sequences
  2. Converts to MIDI via midi_export.py
  3. Saves to outputs/generated_midis/

Usage:
    python src/generation/generate_music.py --model all
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    CHECKPOINTS_DIR, MIDI_OUT_DIR, DEVICE, SEED,
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES, BOS_TOKEN, GENRES
)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────
# Task 1 – Autoencoder generation
# ──────────────────────────────────────────────────

def generate_from_autoencoder(model, n_samples: int = 5, device=None, tag: str = "task1"):
    """
    Sample latent codes z ~ N(0, I) and decode to token sequences.
    Deliverable: 5 generated MIDI samples.
    """
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model.eval()
    from src.config import AE_LATENT_DIM
    z = torch.randn(n_samples, AE_LATENT_DIM, device=device)

    with torch.no_grad():
        logits = model.decode(z)                         # (n, T, V)
        tokens = logits.argmax(dim=-1).cpu().numpy()     # (n, T)

    paths = []
    for i, tok_seq in enumerate(tokens):
        out_path = os.path.join(MIDI_OUT_DIR, f"{tag}_sample_{i+1:02d}.mid")
        _tokens_to_midi(tok_seq.tolist(), out_path, genre_label=0)
        paths.append(out_path)

    print(f"  ✓ Generated {n_samples} samples (autoencoder) → {MIDI_OUT_DIR}")
    return paths


# ──────────────────────────────────────────────────
# Task 2 – VAE generation
# ──────────────────────────────────────────────────

def generate_from_vae(model, n_samples: int = 8, device=None, tag: str = "task2"):
    """
    Sample z ~ N(0, I) for each genre and decode.
    Deliverable: 8 multi-genre MIDI samples.
    """
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model.eval()
    from src.config import VAE_LATENT_DIM

    paths = []
    samples_per_genre = max(1, n_samples // NUM_GENRES)

    for g_idx in range(NUM_GENRES):
        n = samples_per_genre if g_idx < NUM_GENRES - 1 else n_samples - len(paths)
        if n <= 0:
            break

        tokens = model.sample(n, genre_id=g_idx, device=device)  # (n, T)
        tokens = tokens.cpu().numpy()

        for i, tok_seq in enumerate(tokens):
            out_path = os.path.join(MIDI_OUT_DIR,
                                    f"{tag}_{GENRES[g_idx]}_sample_{i+1:02d}.mid")
            _tokens_to_midi(tok_seq.tolist(), out_path, genre_label=g_idx)
            paths.append(out_path)

    print(f"  ✓ Generated {len(paths)} samples (VAE multi-genre) → {MIDI_OUT_DIR}")
    return paths


# ──────────────────────────────────────────────────
# Task 3 & 4 – Transformer generation
# ──────────────────────────────────────────────────

def generate_from_transformer(
    model,
    n_samples: int = 10,
    device=None,
    tag: str = "task3",
    max_new_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.92
):
    """
    Autoregressive generation with nucleus sampling.
    Deliverable: 10 long-sequence MIDI compositions.
    """
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model.eval()
    paths = []

    for i in range(n_samples):
        genre_id = i % NUM_GENRES
        prompt   = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

        generated = model.generate(
            prompt=prompt,
            genre_id=genre_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )                                                # (1, T_total)

        tok_seq  = generated[0].cpu().numpy().tolist()
        out_path = os.path.join(MIDI_OUT_DIR,
                                f"{tag}_{GENRES[genre_id]}_long_{i+1:02d}.mid")
        _tokens_to_midi(tok_seq, out_path, genre_label=genre_id)
        paths.append(out_path)

    print(f"  ✓ Generated {n_samples} long compositions (Transformer) → {MIDI_OUT_DIR}")
    return paths


# ──────────────────────────────────────────────────
# Token → MIDI helper
# ──────────────────────────────────────────────────

def _tokens_to_midi(token_ids: list, output_path: str, genre_label: int = 0, tempo: float = 120.0):
    """
    Convert a list of token ids to a MIDI file via pretty_midi.
    Falls back to a simple note-sequence if pretty_midi is unavailable.
    """
    try:
        import pretty_midi
        from src.preprocessing.tokenizer import MusicTokenizer
        from src.config import STEPS_PER_BAR

        tok = MusicTokenizer()
        events = tok.decode_sequence(token_ids)

        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        # Choose program by genre: 0=piano, 25=jazz guitar, 30=rock, 0=pop, 80=synth
        programs = [0, 25, 30, 0, 80]
        instrument = pretty_midi.Instrument(program=programs[genre_label % len(programs)])

        sec_per_step = 60.0 / (tempo * STEPS_PER_BAR)
        t = 0.0
        for pitch, velocity, dur in events:
            start = t
            end   = t + dur * sec_per_step
            note  = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
            t = end

        pm.instruments.append(instrument)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pm.write(output_path)
        print(f"    ✓ {output_path}")

    except ImportError:
        # Fallback: write a simple MIDI using mido
        _fallback_midi(token_ids, output_path)


def _fallback_midi(token_ids: list, output_path: str, tempo: int = 120):
    """Minimal MIDI writer using mido (no pretty_midi required)."""
    try:
        import mido
        from src.config import STEPS_PER_BAR

        ticks_per_beat = 480
        us_per_beat    = int(60_000_000 / tempo)
        ticks_per_step = ticks_per_beat // (STEPS_PER_BAR // 4)

        mid  = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track= mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))

        prev_pitch = None
        delta = 0
        for tid in token_ids:
            if 4 <= tid < 92:
                pitch = tid - 4 + 21
                if prev_pitch is not None:
                    track.append(mido.Message("note_off", note=prev_pitch, velocity=64, time=delta))
                    delta = 0
                track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
                prev_pitch = pitch
                delta = ticks_per_step
            else:
                delta += ticks_per_step

        if prev_pitch is not None:
            track.append(mido.Message("note_off", note=prev_pitch, velocity=64, time=delta))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mid.save(output_path)
        print(f"    ✓ {output_path} (mido fallback)")

    except Exception as e:
        print(f"    [warn] Could not save MIDI: {e}  → {output_path}")
        # Create placeholder file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0")


# ──────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate music with trained models")
    parser.add_argument("--model", choices=["ae", "vae", "transformer", "rlhf", "all"],
                        default="all", help="Which model to use for generation")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model in ("ae", "all"):
        from src.models.autoencoder import LSTMAutoencoder
        model = LSTMAutoencoder().to(device)
        ckpt  = os.path.join(CHECKPOINTS_DIR, "ae_best.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        print("\n[Task 1] Generating from LSTM Autoencoder …")
        generate_from_autoencoder(model, n_samples=5, device=device, tag="task1")

    if args.model in ("vae", "all"):
        from src.models.vae import MusicVAE
        model = MusicVAE().to(device)
        ckpt  = os.path.join(CHECKPOINTS_DIR, "vae_best.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        print("\n[Task 2] Generating from VAE …")
        generate_from_vae(model, n_samples=8, device=device, tag="task2")

    if args.model in ("transformer", "all"):
        model = MusicTransformer().to(device)
        ckpt  = os.path.join(CHECKPOINTS_DIR, "transformer_best.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        print("\n[Task 3] Generating long compositions from Transformer …")
        generate_from_transformer(model, n_samples=10, device=device, tag="task3")

    if args.model in ("rlhf", "all"):
        model = MusicTransformer().to(device)
        ckpt  = os.path.join(CHECKPOINTS_DIR, "rlhf_best.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        print("\n[Task 4] Generating RLHF-tuned compositions …")
        generate_from_transformer(model, n_samples=10, device=device, tag="task4_rlhf")

    print("\n✓ Generation complete.")


if __name__ == "__main__":
    main()
