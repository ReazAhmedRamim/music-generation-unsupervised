# src/generation/sample_latent.py
"""
Latent space sampling utilities for the VAE and Autoencoder models.

Provides:
  - random_latent_samples()       – sample z ~ N(0,I)
  - interpolate_latent()          – linear interpolation between two z vectors
  - grid_traversal()              – traverse 2 latent dimensions on a grid
"""

import numpy as np
import torch


def random_latent_samples(
    latent_dim: int,
    n: int = 8,
    device=None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample n latent vectors z ~ N(0, temperature²·I).

    Args:
        latent_dim: dimensionality of z
        n:          number of samples
        device:     torch device
        temperature: scaling of the standard normal

    Returns:
        z: (n, latent_dim) float tensor
    """
    z = torch.randn(n, latent_dim, device=device) * temperature
    return z


def interpolate_latent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 8,
    interp_type: str = "linear"
) -> torch.Tensor:
    """
    Interpolate between two latent vectors.

    Args:
        z1, z2:      (latent_dim,) or (1, latent_dim) tensors
        steps:       number of interpolation steps
        interp_type: "linear" or "spherical" (SLERP)

    Returns:
        z_path: (steps, latent_dim) tensor
    """
    z1 = z1.squeeze(0)
    z2 = z2.squeeze(0)

    alphas = torch.linspace(0, 1, steps, device=z1.device)

    if interp_type == "spherical":
        # SLERP: more natural traversal on hypersphere
        z1_n = z1 / (z1.norm() + 1e-8)
        z2_n = z2 / (z2.norm() + 1e-8)
        dot  = torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0)
        omega = torch.acos(dot)
        if omega.abs() < 1e-6:
            # Nearly parallel vectors – fall back to linear
            z_path = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        else:
            z_path = torch.stack([
                (torch.sin((1 - a) * omega) / torch.sin(omega)) * z1 +
                (torch.sin(a * omega)       / torch.sin(omega)) * z2
                for a in alphas
            ])
    else:
        # Linear interpolation
        z_path = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

    return z_path                                    # (steps, latent_dim)


def grid_traversal(
    latent_dim: int,
    dim1: int = 0,
    dim2: int = 1,
    grid_size: int = 5,
    range_std: float = 2.0,
    device=None
) -> torch.Tensor:
    """
    Create a grid of latent vectors by varying two dimensions
    while keeping others at zero.

    Args:
        latent_dim: total latent dimension
        dim1, dim2: which dimensions to traverse
        grid_size:  number of points per axis
        range_std:  ±range_std standard deviations to explore

    Returns:
        grid_z: (grid_size² , latent_dim) tensor
    """
    vals    = torch.linspace(-range_std, range_std, grid_size)
    base_z  = torch.zeros(latent_dim, device=device)
    grid_zs = []

    for v1 in vals:
        for v2 in vals:
            z = base_z.clone()
            z[dim1] = v1
            z[dim2] = v2
            grid_zs.append(z)

    return torch.stack(grid_zs)                      # (grid_size², latent_dim)


def decode_latent_batch(model, z_batch: torch.Tensor, genre_id: int = 0) -> list:
    """
    Decode a batch of latent vectors using a VAE or AE decoder.

    Args:
        model:    MusicVAE or LSTMAutoencoder instance
        z_batch:  (B, latent_dim) tensor
        genre_id: genre label for VAE

    Returns:
        List of token id lists, one per z vector
    """
    model.eval()
    device = next(model.parameters()).device
    z_batch = z_batch.to(device)

    with torch.no_grad():
        if hasattr(model, "decoder") and hasattr(model.decoder, "genre_embedding"):
            # VAE decoder
            genre_ids = torch.full((z_batch.size(0),), genre_id,
                                   dtype=torch.long, device=device)
            logits = model.decoder(z_batch, genre_ids, target=None)
        else:
            # AE decoder
            logits = model.decode(z_batch)

        tokens = logits.argmax(dim=-1).cpu().numpy()   # (B, T)

    return [tok.tolist() for tok in tokens]
