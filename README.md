# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425 / EEE474 — Neural Networks  
**Semester:** Spring 2026

---

## Project Overview

This project implements four progressively complex unsupervised deep generative models for symbolic multi-genre MIDI music generation. Starting from a single-genre LSTM Autoencoder, we extend to a multi-genre β-Variational Autoencoder, a decoder-only Transformer with autoregressive sampling, and finally Reinforcement Learning from Human Feedback (RLHF) fine-tuning.

---

## Datasets Used

| Dataset | Genre | Link |
|---|---|---|
| MAESTRO v3 | Classical Piano | magenta.tensorflow.org/datasets/maestro |
| Lakh MIDI Dataset | Rock / Pop / Electronic | colinraffel.com/projects/lmd |
| Groove MIDI Dataset | Jazz / Drums | magenta.tensorflow.org/datasets/groove |

---

---

## Setup

### 1. Unzip the Project

```bash
cd music-generation-unsupervised
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Organise Datasets

[Download datasets](https://drive.google.com/drive/folders/1injSF7DvoqJapCeiVeX7GAfUXOLAERfL?usp=sharing  "Open Google Drive folder")

Download the three datasets from the drive link and run the setup script:

```bash
# Edit the 3 folder paths inside setup_datasets.py first, then:
python setup_datasets.py
```

This copies files into:
```
data/raw_midi/classical/    ← MAESTRO .midi files
data/raw_midi/jazz/         ← Groove .mid files
data/raw_midi/rock/         ← Lakh split 1
data/raw_midi/pop/          ← Lakh split 2
data/raw_midi/electronic/   ← Lakh split 3
```

---

## Running the Full Pipeline

Run each step in order from the project root:

```bash
# Step 1 — Preprocess MIDI data into token sequences
python src/preprocessing/midi_parser.py

# Step 2 — Run baseline models
python notebooks/baseline_markov.py

# Step 3 — Task 1: Train LSTM Autoencoder
python src/training/train_ae.py

# Step 4 — Task 2: Train VAE
python src/training/train_vae.py

# Step 5 — Task 3: Train Transformer
python src/training/train_transformer.py

# Step 5b — Generate Task 3 MIDI samples (if needed separately)
python src/generation/generate_task3.py

# Step 6 — Task 4: RLHF Fine-Tuning
python src/training/train_rlhf.py

# Step 7 — Generate music from all models
python src/generation/generate_music.py --model all

# Step 8 — Final evaluation and comparison table
python src/evaluation/metrics.py
```


## Task Summary

| Task | Model | Difficulty | Key Metric | Result |
|---|---|---|---|---|
| Task 1 | LSTM Autoencoder | Easy | Reconstruction Loss | 0.82 |
| Task 2 | β-VAE Multi-Genre | Medium | KL + Recon Loss | 0.65 |
| Task 3 | Transformer (Autoregressive) | Hard | Perplexity | 1.02 |
| Task 4 | RLHF Fine-Tuning | Advanced | Human Score | 3.91 / 5 |

---

## Results

| Model | Rhythm Div | Rep. Ratio | Pitch Sim | Perplexity | Human Score |
|---|---|---|---|---|---|
| Random Generator | 0.008 | 0.000 | 0.845 | – | 1.1 |
| Markov Chain | 0.008 | 0.068 | 0.746 | – | 2.3 |
| Task 1: LSTM AE | 0.096 | 0.113 | 0.388 | – | 3.1 |
| Task 2: VAE | 0.247 | 0.718 | 0.146 | – | 3.8 |
| Task 3: Transformer | 0.000 | 0.000 | 0.500 | **1.02** | 4.4 |
| Task 4: RLHF | 0.000 | 0.000 | 0.500 | – | **4.8** |

---

## Project Structure

```
music-generation-unsupervised/
│
├── README.md
├── requirements.txt
├── setup_datasets.py               ← dataset organisation script
│
├── data/
│   ├── raw_midi/                   ← place downloaded MIDI files here
│   │   ├── classical/              ← MAESTRO files
│   │   ├── jazz/                   ← Groove files
│   │   ├── rock/                   ← Lakh split 1
│   │   ├── pop/                    ← Lakh split 2
│   │   └── electronic/             ← Lakh split 3
│   ├── processed/                  ← auto-filled by midi_parser.py
│   └── train_test_split/           ← train.pkl / val.pkl / test.pkl
│
├── src/
│   ├── config.py                   ← all hyperparameters
│   ├── preprocessing/
│   │   ├── midi_parser.py          ← MIDI to token sequences
│   │   ├── tokenizer.py            ← token and note event converter
│   │   └── piano_roll.py           ← piano roll and MIDI converter
│   ├── models/
│   │   ├── autoencoder.py          ← Task 1: LSTM Autoencoder
│   │   ├── vae.py                  ← Task 2: beta-VAE
│   │   └── transformer.py          ← Task 3 and 4: Transformer
│   ├── training/
│   │   ├── train_ae.py             ← Task 1 training
│   │   ├── train_vae.py            ← Task 2 training
│   │   ├── train_transformer.py    ← Task 3 training
│   │   └── train_rlhf.py           ← Task 4 RLHF
│   ├── evaluation/
│   │   ├── metrics.py              ← all evaluation metrics and comparison
│   │   ├── pitch_histogram.py      ← H(p,q) pitch similarity
│   │   └── rhythm_score.py         ← D_rhythm and repetition ratio
│   └── generation/
│       ├── generate_music.py       ← unified generation for all tasks
│       ├── generate_task3.py       ← standalone Task 3 generation
│       ├── midi_export.py          ← token sequences to .mid files
│       └── sample_latent.py        ← latent interpolation and sampling
│
├── notebooks/
│   ├── preprocessing.py            ← data visualisation walkthrough
│   └── baseline_markov.py          ← Markov chain baseline
│
├── outputs/
│   ├── generated_midis/            ← all .mid files (33 total)
│   ├── plots/                      ← all training and evaluation plots
│   └── survey_results/
│       └── survey_data.json        ← human listening survey data
│
├── checkpoints/                    ← saved model weights
│   ├── ae_best.pt                  ← Task 1
│   ├── vae_best.pt                 ← Task 2
│   ├── transformer_best.pt         ← Task 3
│   ├── reward_model.pt             ← Task 4 reward model
│   └── rlhf_best.pt                ← Task 4 RLHF
│
└── report/
    ├── final_report.tex            ← LaTeX source (NeurIPS format)
    ├── final_report.pdf            ← compiled PDF
    └── references.bib              ← 13 academic citations
```

---

## Mathematical Formulation

### Task 1 — LSTM Autoencoder
```
z    = f_phi(X)               Encoder
X_hat = g_theta(z)            Decoder
L_AE = sum_t ||x_t - x_hat_t||^2    Reconstruction loss
```

### Task 2 — beta-VAE
```
q_phi(z|X) = N(mu(X), sigma(X))       Latent distribution
z = mu + sigma * eps, eps ~ N(0, I)   Reparameterisation trick
L_VAE = L_recon + beta * D_KL         Combined loss  (beta = 4.0)
```

### Task 3 — Transformer
```
p(X) = product_t p_theta(x_t | x_{<t})     Autoregressive factorisation
L_TR = -sum_t log p_theta(x_t | x_{<t})    Negative log-likelihood
PPL  = exp(1/T * L_TR) = 1.02              Perplexity achieved
```

### Task 4 — RLHF
```
X_gen ~ p_theta(X)
r = HumanScore(X_gen)
gradient_theta J(theta) = E[r * gradient_theta log p_theta(X)]
```

---

## Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| Pitch Histogram Similarity | H(p,q) = sum\|p_i - q_i\| | L1 distance between pitch class histograms |
| Rhythm Diversity | D = #unique_durations / #total_notes | Higher = more varied rhythms |
| Repetition Ratio | R = #repeated_patterns / #total_patterns | Lower = less repetitive |
| Perplexity | PPL = exp(1/T * L_TR) | Lower = better sequence model |
| Human Score | Score in [1, 5] | Subjective listening quality rating |

---

## Generated MIDI Output Files

All generated samples are saved in `outputs/generated_midis/`:

| File Pattern | Model | Count |
|---|---|---|
| task1_sample_01 to 05.mid | LSTM Autoencoder | 5 |
| task2_[genre]_sample_01.mid | VAE (multi-genre) | 8 |
| task3_[genre]_long_01 to 10.mid | Transformer | 10 |
| task4_rlhf_[genre]_long_01 to 10.mid | RLHF-Tuned | 10 |

---

## Training Plots

All plots are saved in `outputs/plots/`:

| File | Description |
|---|---|
| ae_loss_curve.png | Task 1 reconstruction loss curve |
| vae_loss_curves.png | Task 2 total / recon / KL loss curves |
| vae_latent_pca.png | Task 2 latent space PCA by genre |
| transformer_curves.png | Task 3 NLL loss and perplexity |
| rlhf_training.png | Task 4 reward before vs after |
| model_comparison.png | Final comparison across all models |

---

## Report

Written in NeurIPS 2024 LaTeX format.

- Source: `report/final_report.tex`
- Bibliography: `report/references.bib` (13 citations)
- Compiled PDF: `report/final_report.pdf`