# src/config.py
"""
Global configuration for the Music Generation project.
Modify these settings before training.
"""

import os
import torch

# ──────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR         = os.path.join(BASE_DIR, "data")
RAW_MIDI_DIR     = os.path.join(DATA_DIR, "raw_midi")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")
SPLIT_DIR        = os.path.join(DATA_DIR, "train_test_split")

OUTPUTS_DIR      = os.path.join(BASE_DIR, "outputs")
MIDI_OUT_DIR     = os.path.join(OUTPUTS_DIR, "generated_midis")
PLOTS_DIR        = os.path.join(OUTPUTS_DIR, "plots")
SURVEY_DIR       = os.path.join(OUTPUTS_DIR, "survey_results")

CHECKPOINTS_DIR  = os.path.join(BASE_DIR, "checkpoints")

# ──────────────────────────────────────────────────
# MIDI / Tokenization
# ──────────────────────────────────────────────────
STEPS_PER_BAR    = 16          # temporal resolution
SEQUENCE_LENGTH  = 128         # tokens per training window
PITCH_RANGE      = (21, 108)   # standard piano range (MIDI)
VELOCITY_BINS    = 32          # number of velocity bins
NUM_PITCHES      = PITCH_RANGE[1] - PITCH_RANGE[0] + 1  # 88

# Genres supported
GENRES = ["classical", "jazz", "rock", "pop", "electronic"]
NUM_GENRES = len(GENRES)

# Vocabulary size  (pitch + velocity_bins + special tokens)
VOCAB_SIZE = NUM_PITCHES + VELOCITY_BINS + 4   # +4: PAD, BOS, EOS, REST

# Special token IDs
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
REST_TOKEN = 3

# ──────────────────────────────────────────────────
# Training (shared)
# ──────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_WORKERS  = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"           # change to "cpu" if no GPU
SEED         = 42

# ──────────────────────────────────────────────────
# Task 1 – LSTM Autoencoder
# ──────────────────────────────────────────────────
AE_HIDDEN_DIM   = 512
AE_LATENT_DIM   = 128
AE_NUM_LAYERS   = 2
AE_DROPOUT      = 0.2
AE_LEARNING_RATE = 1e-3
AE_EPOCHS       = 50

# ──────────────────────────────────────────────────
# Task 2 – VAE
# ──────────────────────────────────────────────────
VAE_HIDDEN_DIM   = 512
VAE_LATENT_DIM   = 256
VAE_NUM_LAYERS   = 2
VAE_DROPOUT      = 0.2
VAE_LEARNING_RATE = 1e-3
VAE_EPOCHS       = 60
VAE_BETA         = 4.0          # KL weight (β-VAE)

# ──────────────────────────────────────────────────
# Task 3 – Transformer
# ──────────────────────────────────────────────────
TF_D_MODEL      = 256
TF_NHEAD        = 8
TF_NUM_LAYERS   = 6
TF_DIM_FF       = 1024
TF_DROPOUT      = 0.1
TF_LEARNING_RATE = 5e-4
TF_EPOCHS       = 80
TF_MAX_SEQ_LEN  = 512

# ──────────────────────────────────────────────────
# Task 4 – RLHF
# ──────────────────────────────────────────────────
RLHF_LEARNING_RATE = 1e-5
RLHF_ITERATIONS    = 200
RLHF_SAMPLE_SIZE   = 8          # samples per RL step
REWARD_HIDDEN_DIM  = 256

# ──────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────
EVAL_NUM_SAMPLES   = 100
HUMAN_SCORE_SCALE  = (1, 5)

# ──────────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────────
for _dir in [RAW_MIDI_DIR, PROCESSED_DIR, SPLIT_DIR,
             MIDI_OUT_DIR, PLOTS_DIR, SURVEY_DIR, CHECKPOINTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
