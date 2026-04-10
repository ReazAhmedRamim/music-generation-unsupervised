# src/preprocessing/tokenizer.py
"""
Token ↔ MIDI event mapping utilities.
Provides MusicTokenizer used by all models.
"""

import numpy as np
from src.config import (
    PITCH_RANGE, VELOCITY_BINS, NUM_PITCHES, VOCAB_SIZE,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, REST_TOKEN
)


class MusicTokenizer:
    """
    Encodes/decodes between (pitch, velocity, duration) tuples
    and integer token IDs.

    Token layout:
      0       → PAD
      1       → BOS
      2       → EOS
      3       → REST
      4..91   → pitch tokens  (MIDI 21-108, 88 pitches)
      92..123 → velocity bins (32 bins)
    """

    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self._pitch_offset = 4                          # first pitch token id
        self._vel_offset   = self._pitch_offset + NUM_PITCHES  # first velocity token id

    # ── encoding ──────────────────────────────────────────────────

    def pitch_to_id(self, midi_pitch: int) -> int:
        idx = int(np.clip(midi_pitch - PITCH_RANGE[0], 0, NUM_PITCHES - 1))
        return idx + self._pitch_offset

    def id_to_pitch(self, token_id: int) -> int:
        return (token_id - self._pitch_offset) + PITCH_RANGE[0]

    def velocity_to_id(self, velocity: int) -> int:
        bin_idx = int(velocity / 128 * VELOCITY_BINS)
        return bin_idx + self._vel_offset

    def id_to_velocity(self, token_id: int) -> int:
        bin_idx = token_id - self._vel_offset
        return int((bin_idx + 0.5) / VELOCITY_BINS * 128)

    def is_pitch(self, token_id: int) -> bool:
        return self._pitch_offset <= token_id < self._vel_offset

    def is_velocity(self, token_id: int) -> bool:
        return self._vel_offset <= token_id < self.vocab_size

    def is_special(self, token_id: int) -> bool:
        return token_id in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, REST_TOKEN)

    # ── sequence helpers ──────────────────────────────────────────

    def encode_sequence(self, note_events) -> list:
        """
        note_events: list of (pitch, velocity, duration_steps)
        Returns: list of token ids  [BOS, pitch, …, EOS]
        """
        tokens = [BOS_TOKEN]
        for pitch, velocity, duration in note_events:
            tokens.append(self.pitch_to_id(pitch))
            tokens.extend([REST_TOKEN] * max(0, duration - 1))
        tokens.append(EOS_TOKEN)
        return tokens

    def decode_sequence(self, token_ids) -> list:
        """
        Returns list of (pitch, velocity, duration_steps) tuples.
        Consecutive REST tokens after a pitch extend its duration.
        """
        events = []
        current_pitch    = None
        current_duration = 0
        default_velocity = 64

        for tid in token_ids:
            if tid in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                if current_pitch is not None:
                    events.append((current_pitch, default_velocity, current_duration))
                    current_pitch = None
                if tid == EOS_TOKEN:
                    break
            elif tid == REST_TOKEN:
                if current_pitch is not None:
                    current_duration += 1
                # else: leading rest, ignore
            elif self.is_pitch(tid):
                if current_pitch is not None:
                    events.append((current_pitch, default_velocity, current_duration))
                current_pitch    = self.id_to_pitch(tid)
                current_duration = 1
            # velocity tokens (future extension) – skip for now

        return events

    def pad_sequence(self, tokens: list, max_len: int) -> list:
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [PAD_TOKEN] * (max_len - len(tokens))

    def __repr__(self):
        return (f"MusicTokenizer(vocab_size={self.vocab_size}, "
                f"pitch_range={PITCH_RANGE}, velocity_bins={VELOCITY_BINS})")
