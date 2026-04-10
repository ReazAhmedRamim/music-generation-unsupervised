# fix_generate.py
import torch
import os, sys
sys.path.insert(0, ".")

from src.config import CHECKPOINTS_DIR
from src.models.transformer import MusicTransformer
from src.generation.generate_music import generate_from_transformer

# Use CPU for generation to avoid CUDA assertion errors
device = torch.device("cpu")
print(f"Device: {device}")

# Load the saved Task 3 checkpoint
model = MusicTransformer().to(device)
ckpt  = torch.load(
    os.path.join(CHECKPOINTS_DIR, "transformer_best.pt"),
    map_location=device
)
model.load_state_dict(ckpt["state_dict"])
print(f"Loaded checkpoint — PPL: {ckpt.get('perplexity', '?'):.4f}")

# Generate 10 long compositions
generate_from_transformer(
    model,
    n_samples=10,
    device=device,
    tag="task3",
    max_new_tokens=256,   # reduced for speed on CPU
    temperature=1.0,      # safer than 0.9 for avoiding -inf
    top_k=40,
    top_p=0.95
)

print("\n✓ Done! Check outputs/generated_midis/")