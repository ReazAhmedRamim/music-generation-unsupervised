# src/preprocessing/piano_roll.py
"""
Piano-roll ↔ token sequence converters.

A piano-roll is a 2-D binary/continuous array of shape (NUM_PITCHES, T)
where axis-0 is pitch and axis-1 is time (quantised to STEPS_PER_BAR).
"""

import numpy as np
import pretty_midi

from src.config import (
    PITCH_RANGE, NUM_PITCHES, STEPS_PER_BAR, SEQUENCE_LENGTH
)


# ──────────────────────────────────────────────────
# MIDI → Piano Roll
# ──────────────────────────────────────────────────

def midi_to_piano_roll(midi_path: str, fs: float = None) -> np.ndarray:
    """
    Convert a MIDI file to a piano roll array.

    Args:
        midi_path: path to .mid file
        fs: frames per second (None → inferred from STEPS_PER_BAR + tempo)

    Returns:
        piano_roll: np.ndarray of shape (NUM_PITCHES, T) with values in [0, 1]
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if fs is None:
        tempo = pm.estimate_tempo()
        fs = tempo * STEPS_PER_BAR / 60.0  # steps per second

    roll = pm.get_piano_roll(fs=fs)          # shape: (128, T)
    # Crop to standard piano range
    roll = roll[PITCH_RANGE[0]:PITCH_RANGE[1] + 1, :]  # (NUM_PITCHES, T)
    # Normalise velocity to [0, 1]
    roll = roll / 127.0
    return roll.astype(np.float32)


def segment_piano_roll(roll: np.ndarray, seg_len: int = SEQUENCE_LENGTH):
    """
    Slice piano roll along time axis into non-overlapping segments.

    Args:
        roll:    (NUM_PITCHES, T)
        seg_len: number of time steps per segment

    Returns:
        List of (NUM_PITCHES, seg_len) arrays
    """
    T = roll.shape[1]
    segments = []
    for start in range(0, T - seg_len, seg_len):
        segments.append(roll[:, start:start + seg_len])
    return segments


# ──────────────────────────────────────────────────
# Piano Roll → MIDI
# ──────────────────────────────────────────────────

def piano_roll_to_midi(
    roll: np.ndarray,
    output_path: str,
    tempo: float = 120.0,
    program: int = 0,
    threshold: float = 0.3
):
    """
    Convert a piano roll array back to a MIDI file.

    Args:
        roll:        (NUM_PITCHES, T) float array in [0, 1]
        output_path: where to save the .mid file
        tempo:       BPM
        program:     General MIDI program number (0 = Acoustic Grand Piano)
        threshold:   minimum value to count a cell as a note-on
    """
    pm        = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)

    seconds_per_step = 60.0 / (tempo * STEPS_PER_BAR)
    pitch_offset     = PITCH_RANGE[0]
    T                = roll.shape[1]

    for p_idx in range(NUM_PITCHES):
        midi_pitch = p_idx + pitch_offset
        active     = False
        note_start = 0.0

        for t in range(T):
            on = roll[p_idx, t] >= threshold
            if on and not active:
                note_start = t * seconds_per_step
                active     = True
            elif not on and active:
                velocity = int(roll[p_idx, t - 1] * 127)
                note     = pretty_midi.Note(
                    velocity=max(1, velocity),
                    pitch=midi_pitch,
                    start=note_start,
                    end=t * seconds_per_step
                )
                instrument.notes.append(note)
                active = False

        # Close any open note at the end
        if active:
            velocity = int(roll[p_idx, T - 1] * 127)
            note     = pretty_midi.Note(
                velocity=max(1, velocity),
                pitch=midi_pitch,
                start=note_start,
                end=T * seconds_per_step
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_path)
    print(f"  ✓ Saved MIDI → {output_path}")


# ──────────────────────────────────────────────────
# Token sequence ↔ Piano Roll helpers
# ──────────────────────────────────────────────────

def tokens_to_piano_roll(token_ids, seq_len=SEQUENCE_LENGTH) -> np.ndarray:
    """
    Convert a flat token sequence to a simplified piano-roll for evaluation.
    (Assumes pitch tokens map pitch directly, rest tokens = silence.)
    """
    from src.config import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, REST_TOKEN
    from src.preprocessing.tokenizer import MusicTokenizer

    tokenizer = MusicTokenizer()
    roll = np.zeros((NUM_PITCHES, seq_len), dtype=np.float32)

    t = 0
    for tid in token_ids:
        if t >= seq_len:
            break
        if tokenizer.is_pitch(tid):
            p_idx = tokenizer.id_to_pitch(tid) - PITCH_RANGE[0]
            if 0 <= p_idx < NUM_PITCHES:
                roll[p_idx, t] = 1.0
        t += 1

    return roll
