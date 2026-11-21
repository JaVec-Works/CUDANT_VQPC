"""CUDANT_VQPC — encode_algorithm.py
Convert a gate sequence into a tiling."""

import torch
import numpy as np
from tiling import flat_mirrors


def circuit_to_tiling(name: str, n_qubits: int = 4, N: int = 16,
                      target: int = 0) -> torch.Tensor:
    """
    Generate a tiling that encodes the given algorithm.
    v1: flat mirrors + one modified tile (the "oracle tile").
    Full implementation maps gate sequences to tiling geometry.
    """
    T = flat_mirrors(N)

    if name == "grover":
        # Mark one tile with a phase-flip vector (anti-parallel to face normal)
        tile_idx = target % (N * N)
        u, w = tile_idx // N, tile_idx % N
        T[0, u, w, 0] = -1.0  # anti-reflect on face 0

    elif name == "bell_pair":
        # Two tiles that create a correlated reflection pair
        T[0, 0, 0, :] = torch.tensor([1.0, 1.0, 0.0]) / np.sqrt(2)
        T[2, 0, 0, :] = torch.tensor([1.0, 0.0, 1.0]) / np.sqrt(2)

    elif name == "ghz_state":
        # All tiles on face 0 point toward the same corner
        corner = torch.tensor([1.0, 1.0, 1.0]) / np.sqrt(3)
        T[0, :, :, :] = corner.expand(N, N, 3)

    # qft, phase_estimation: TODO (frequency-encoded tiling)

    return T   