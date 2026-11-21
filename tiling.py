"""CUDANT_VQPC — tiling.py
Tiling generation and persistence."""

import torch
import os

_TILING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tilings")
os.makedirs(_TILING_DIR, exist_ok=True)


def flat_mirrors(N: int) -> torch.Tensor:
    """Every tile reflects along the face normal. Trivial periodic orbits."""
    T = torch.zeros(6, N, N, 3, dtype=torch.float32)
    # Face 0 (±X): normal = (1,0,0). Face 1 (±Y): (0,1,0). Face 2 (±Z): (0,0,1)
    T[0, :, :, 0] = 1.0
    T[1, :, :, 1] = 1.0
    T[2, :, :, 2] = 1.0
    T[3, :, :, 0] = -1.0
    T[4, :, :, 1] = -1.0
    T[5, :, :, 2] = -1.0
    return T


def random_tiling(N: int, seed: int = 42) -> torch.Tensor:
    """Random unit vectors. Chaotic orbits."""
    torch.manual_seed(seed)
    T = torch.randn(6, N, N, 3, dtype=torch.float32)
    T = T / T.norm(dim=-1, keepdim=True)
    return T


def save_tiling(T: torch.Tensor, name: str) -> str:
    path = os.path.join(_TILING_DIR, f"{name}.pt")
    torch.save({"tiling": T, "N": T.shape[1]}, path)
    return path


def load_tiling(name: str) -> torch.Tensor:
    path = os.path.join(_TILING_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tiling '{name}' not found in {_TILING_DIR}")
    return torch.load(path, weights_only=True)["tiling"]   