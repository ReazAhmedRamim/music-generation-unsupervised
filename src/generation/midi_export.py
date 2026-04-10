# src/generation/midi_export.py
"""
MIDI export utilities – converts token sequences / piano rolls to .mid files.
"""

import os
import pretty_midi
import numpy as np

from src.config import STEPS_PER_BAR, PITCH_RANGE, NUM_PITCHES


def tokens_to_midi_file(
    token_ids: list,
    output_path: str,
    genre_label: int = 0,
    tempo: float = 120.0,
    ticks_per_beat: int = 480
) -> str:
    """
    Convert a token id sequence to a MIDI file.

    Args:
        token_ids:    list of int token ids
        output_path:  where to write the .mid file
        genre_label:  selects instrument program
        tempo:        BPM
        ticks_per_beat

    Returns:
        output_path
    """
    from src.preprocessing.tokenizer import MusicTokenizer

    tokenizer = MusicTokenizer()
    events    = tokenizer.decode_sequence(token_ids)   # [(pitch, vel, dur)]

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    programs = {0: 0, 1: 25, 2: 30, 3: 0, 4: 80}     # genre → GM program
    program  = programs.get(genre_label, 0)
    inst     = pretty_midi.Instrument(program=program)

    sec_per_step = 60.0 / (tempo * STEPS_PER_BAR)
    t = 0.0
    for pitch, velocity, dur in events:
        start = t
        end   = t + dur * sec_per_step
        note  = pretty_midi.Note(
            velocity=max(1, min(127, velocity)),
            pitch=max(0, min(127, pitch)),
            start=start,
            end=max(start + 0.01, end)
        )
        inst.notes.append(note)
        t = end

    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pm.write(output_path)
    return output_path


def piano_roll_to_midi_file(
    roll: np.ndarray,
    output_path: str,
    tempo: float = 120.0,
    threshold: float = 0.3,
    program: int = 0
) -> str:
    """
    Convert a (NUM_PITCHES, T) piano roll to MIDI.

    Args:
        roll:         (NUM_PITCHES, T) float [0,1]
        output_path:  output .mid path
        tempo:        BPM
        threshold:    minimum activation to count as note-on
        program:      GM instrument program

    Returns:
        output_path
    """
    pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program)

    sec_per_step = 60.0 / (tempo * STEPS_PER_BAR)
    pitch_offset = PITCH_RANGE[0]
    T = roll.shape[1]

    for p_idx in range(NUM_PITCHES):
        midi_pitch = p_idx + pitch_offset
        active     = False
        note_start = 0.0
        note_vel   = 64

        for t in range(T):
            on = roll[p_idx, t] >= threshold
            if on and not active:
                note_start = t * sec_per_step
                note_vel   = int(roll[p_idx, t] * 127)
                active     = True
            elif not on and active:
                inst.notes.append(
                    pretty_midi.Note(velocity=max(1, note_vel), pitch=midi_pitch,
                                     start=note_start, end=t * sec_per_step)
                )
                active = False
        if active:
            inst.notes.append(
                pretty_midi.Note(velocity=max(1, note_vel), pitch=midi_pitch,
                                 start=note_start, end=T * sec_per_step)
            )

    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pm.write(output_path)
    return output_path


def batch_export(token_sequences: list, out_dir: str, prefix: str = "sample", **kwargs):
    """Export a list of token sequences to numbered MIDI files."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, tokens in enumerate(token_sequences):
        path = os.path.join(out_dir, f"{prefix}_{i+1:03d}.mid")
        tokens_to_midi_file(tokens, path, **kwargs)
        paths.append(path)
        print(f"  Exported: {path}")
    return paths
