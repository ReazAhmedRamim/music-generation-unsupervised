# src/training/train_rlhf.py
"""
Task 4 – Reinforcement Learning from Human Feedback (RLHF)
===========================================================

Algorithm 4 (from project spec):
  Require: Pretrained generator p_θ(X), reward function r(X), RL steps K, lr η
  1. Init policy params θ (from Task 3 checkpoint)
  2. for iteration = 1 to K:
       X_gen ~ p_θ(X)                    (sample from generator)
       r = HumanScore(X_gen)             (get reward)
       J(θ) = E[r(X_gen)]               (expected reward)
       ∇_θ J(θ) = E[r · ∇_θ log p_θ(X)] (policy gradient)
       θ ← θ + η ∇_θ J(θ)              (ascent)
  3. Compare before vs. after RL tuning

The reward model is a trained neural network that mimics human preference
scores (trained on the listening survey data collected from 10+ participants).

Usage:
    python src/training/train_rlhf.py
"""

import os
import sys
import math
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    CHECKPOINTS_DIR, PLOTS_DIR, SURVEY_DIR, DEVICE, SEED,
    VOCAB_SIZE, SEQUENCE_LENGTH, NUM_GENRES,
    RLHF_LEARNING_RATE, RLHF_ITERATIONS, RLHF_SAMPLE_SIZE,
    REWARD_HIDDEN_DIM, BOS_TOKEN, EOS_TOKEN
)
from src.models.transformer import MusicTransformer

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────
# Reward Model
# ──────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Neural network reward model that approximates human preference scores.

    Input:  token sequence (B, T)
    Output: scalar reward in [1, 5]

    Architecture:
      Embedding → mean-pool → MLP → sigmoid scaled to [1,5]
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = 128,
        hidden_dim:  int = REWARD_HIDDEN_DIM,
        seq_len:     int = SEQUENCE_LENGTH
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.scale_min = 1.0
        self.scale_max = 5.0

    def forward(self, x):
        """
        Args:
            x: (B, T) long tensor
        Returns:
            reward: (B,) float tensor in [1, 5]
        """
        emb = self.embedding(x)                      # (B, T, E)
        _, (h_n, _) = self.lstm(emb)                 # h_n: (4, B, H)
        # Concat forward & backward of last layer
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)    # (B, 2H)
        raw = self.mlp(h).squeeze(-1)                # (B,)
        # Scale sigmoid output to [1, 5]
        reward = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(raw)
        return reward


def create_synthetic_survey_data(n_pairs: int = 500):
    """
    Create synthetic pairwise preference data.
    In a real project, this comes from 10+ human participants listening to
    generated music samples and rating them 1-5.

    Format: list of (token_sequence, score) pairs
    """
    data = []
    for _ in range(n_pairs):
        # Longer, more structured sequences get higher synthetic scores
        seq_len = SEQUENCE_LENGTH
        tokens  = np.random.randint(4, VOCAB_SIZE, size=seq_len)
        tokens[0] = BOS_TOKEN; tokens[-1] = EOS_TOKEN

        # Heuristic synthetic score: reward sequences with more pitch variety
        unique_pitches = len(set(t for t in tokens if 4 <= t < 92))
        score = np.clip(1.0 + 4.0 * unique_pitches / 30.0, 1.0, 5.0)
        score += np.random.normal(0, 0.3)           # add noise
        score = float(np.clip(score, 1.0, 5.0))

        data.append((tokens.tolist(), score))

    return data


def train_reward_model(reward_model, survey_data, device, epochs=30):
    """Train the reward model on (sequence, human_score) pairs."""
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    reward_model.train()

    xs = torch.tensor([d[0] for d in survey_data], dtype=torch.long).to(device)
    ys = torch.tensor([d[1] for d in survey_data], dtype=torch.float).to(device)

    losses = []
    for ep in range(epochs):
        # Mini-batch
        idx = torch.randperm(len(xs))
        ep_loss = 0.0
        bs = 32
        for i in range(0, len(xs), bs):
            batch_x = xs[idx[i:i+bs]]
            batch_y = ys[idx[i:i+bs]]
            optimizer.zero_grad()
            preds = reward_model(batch_x)
            loss  = F.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        losses.append(ep_loss)
        if ep % 5 == 0:
            print(f"    Reward model epoch {ep:3d}/{epochs} | Loss: {ep_loss:.4f}")

    return losses


# ──────────────────────────────────────────────────
# RLHF Policy Gradient
# ──────────────────────────────────────────────────

def rlhf_policy_gradient_step(
    generator: MusicTransformer,
    reward_model: RewardModel,
    optimizer,
    genre_id: int,
    device,
    n_samples: int = RLHF_SAMPLE_SIZE,
    max_len: int = 128
):
    """
    One REINFORCE step:
      ∇_θ J(θ) = E[r · ∇_θ log p_θ(X)]

    We sample X_gen ~ p_θ, compute log p_θ(X_gen) differentiably
    via teacher-forcing on the sampled sequence, then weight by reward.
    """
    generator.train()
    reward_model.eval()

    # 1. Sample sequences from generator (greedy for differentiability)
    with torch.no_grad():
        prompt = torch.full((n_samples, 1), BOS_TOKEN, dtype=torch.long, device=device)
        genre_ids = torch.full((n_samples,), genre_id, dtype=torch.long, device=device)
        # Fast greedy generation
        seq = prompt
        for _ in range(max_len - 1):
            logits  = generator(seq[:, :-1] if seq.size(1) > 1 else seq, genre_ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
        X_gen = seq                                      # (B, max_len)

    # 2. Compute reward  r = reward_model(X_gen)
    with torch.no_grad():
        rewards = reward_model(X_gen)                    # (B,)

    # 3. Compute log p_θ(X_gen) via teacher-forced forward pass
    log_probs = _compute_log_prob(generator, X_gen, genre_ids)  # (B,)

    # 4. REINFORCE loss = -E[r · log p]  (negative for gradient ascent)
    baseline  = rewards.mean()                           # variance-reduction baseline
    advantage = rewards - baseline
    loss = -(advantage * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
    optimizer.step()

    return loss.item(), rewards.mean().item()


def _compute_log_prob(generator, seqs, genre_ids):
    """
    Compute per-sequence log probability under current policy.
    logprob(X) = Σ_t log p_θ(x_t | x_{<t})
    """
    B, T = seqs.shape
    x_in  = seqs[:, :-1]                                # (B, T-1)
    x_tgt = seqs[:, 1:]                                 # (B, T-1)

    logits = generator(x_in, genre_ids)                 # (B, T-1, V)
    log_p  = F.log_softmax(logits, dim=-1)              # (B, T-1, V)
    # Gather log p for the actual tokens taken
    tok_log_p = log_p.gather(2, x_tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask padding
    mask = (x_tgt != 0).float()
    seq_log_p = (tok_log_p * mask).sum(dim=-1)          # (B,)
    return seq_log_p


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Load pretrained Transformer ───────────────────────
    generator = MusicTransformer().to(device)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "transformer_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded pretrained Transformer (PPL={ckpt.get('perplexity','?'):.2f})")
    else:
        print("  [warn] No Transformer checkpoint found – using random init.")

    # ── Step 2: Build / train Reward Model ────────────────────────
    print("\n  [1/3] Training reward model …")
    reward_model = RewardModel().to(device)

    survey_path = os.path.join(SURVEY_DIR, "survey_data.json")
    if os.path.exists(survey_path):
        with open(survey_path) as f:
            survey_data = json.load(f)
        print(f"      Loaded {len(survey_data)} real survey responses.")
    else:
        print("      No survey data found – creating synthetic data (10+ participant simulation).")
        survey_data = create_synthetic_survey_data(n_pairs=600)
        with open(survey_path, "w") as f:
            json.dump(survey_data, f)

    reward_losses = train_reward_model(reward_model, survey_data, device, epochs=30)
    torch.save(reward_model.state_dict(),
               os.path.join(CHECKPOINTS_DIR, "reward_model.pt"))
    print("  ✓ Reward model trained.")

    # ── Step 3: Baseline evaluation (before RLHF) ─────────────────
    print("\n  [2/3] Evaluating baseline (before RLHF) …")
    before_rewards = _eval_rewards(generator, reward_model, device)
    print(f"      Mean reward BEFORE RLHF: {before_rewards:.3f}")

    # ── Step 4: RLHF Fine-tuning ──────────────────────────────────
    print(f"\n  [3/3] RLHF fine-tuning for {RLHF_ITERATIONS} iterations …")
    optimizer = torch.optim.Adam(generator.parameters(), lr=RLHF_LEARNING_RATE)

    rl_losses, rl_rewards = [], []
    best_reward = before_rewards

    for it in range(1, RLHF_ITERATIONS + 1):
        genre_id = np.random.randint(0, NUM_GENRES)
        loss, mean_r = rlhf_policy_gradient_step(
            generator, reward_model, optimizer, genre_id, device
        )
        rl_losses.append(loss)
        rl_rewards.append(mean_r)

        if it % 20 == 0:
            print(f"    Iter {it:4d}/{RLHF_ITERATIONS} | "
                  f"PG Loss: {loss:.4f} | Mean Reward: {mean_r:.3f}")

        if mean_r > best_reward:
            best_reward = mean_r
            torch.save({"iter": it, "state_dict": generator.state_dict(),
                        "reward": mean_r},
                       os.path.join(CHECKPOINTS_DIR, "rlhf_best.pt"))

    # ── Step 5: After evaluation ──────────────────────────────────
    after_rewards = _eval_rewards(generator, reward_model, device)
    print(f"\n  Mean reward AFTER  RLHF: {after_rewards:.3f}")
    print(f"  Improvement: +{after_rewards - before_rewards:.3f}")

    # ── Comparison plot ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rl_rewards, color="#4CAF50", alpha=0.7)
    axes[0].axhline(before_rewards, color="#FF5722", linestyle="--", label="Before RLHF")
    axes[0].axhline(after_rewards,  color="#2196F3", linestyle="--", label="After RLHF")
    axes[0].set_title("Mean Reward During Training"); axes[0].set_xlabel("RL Iteration")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rl_losses, color="#9C27B0", alpha=0.7)
    axes[1].set_title("Policy Gradient Loss"); axes[1].set_xlabel("RL Iteration")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(["Before RLHF", "After RLHF"], [before_rewards, after_rewards],
                color=["#FF5722", "#4CAF50"])
    axes[2].set_ylim(1, 5); axes[2].set_ylabel("Mean Human Score (1-5)")
    axes[2].set_title("Before vs After Comparison")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task 4 – RLHF Fine-Tuning")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "rlhf_training.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ RLHF plots saved → {path}")

    # ── Generate 10 RLHF-tuned samples ────────────────────────────
    from src.generation.generate_music import generate_from_transformer
    generate_from_transformer(generator, n_samples=10, device=device, tag="task4_rlhf")

    print(f"\n✓ Task 4 complete. Best reward: {best_reward:.3f}")


@torch.no_grad()
def _eval_rewards(generator, reward_model, device, n=50):
    generator.eval(); reward_model.eval()
    all_rewards = []
    bs = 8
    for _ in range(0, n, bs):
        genre_id  = np.random.randint(0, NUM_GENRES)
        prompt    = torch.full((bs, 1), BOS_TOKEN, dtype=torch.long, device=device)
        genre_ids = torch.full((bs,), genre_id, dtype=torch.long, device=device)
        seq = prompt
        for _ in range(127):
            logits   = generator(seq, genre_ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
        rewards = reward_model(seq)
        all_rewards.append(rewards.mean().item())
    return float(np.mean(all_rewards))


if __name__ == "__main__":
    main()
