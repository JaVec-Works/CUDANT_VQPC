"""CUDANT_VQPC — box.py
The vector-tiled cube. The tiling IS the program. No gates, no rods."""

import torch
import numpy as np


class Box:
    def __init__(self, N: int, tiling: torch.Tensor, device="cuda"):
        self.N = N
        self.T = tiling.to(device)  # (6, N, N, 3)
        self.device = device

    def _ray_box(self, v: torch.Tensor):
        """Return (t, hit_point, face_index)."""
        inv = torch.where(v.abs() > 1e-8, 1.0 / v.abs(), torch.tensor(float('inf'), device=v.device))
        t = inv.min()
        h = t * v
        face = v.abs().argmax().item()
        return t, h, face

    def _tile_index(self, h: torch.Tensor, face: int):
        """Map hit point to (idx_u, idx_w) on the given face."""
        N = self.N
        if face == 0:   # X-face
            u, w = h[1], h[2]
        elif face == 1:  # Y-face
            u, w = h[0], h[2]
        else:            # Z-face
            u, w = h[0], h[1]
        idx_u = min(int((u.item() + 1.0) * 0.5 * N), N - 1)
        idx_w = min(int((w.item() + 1.0) * 0.5 * N), N - 1)
        return idx_u, idx_w

    def step(self, v: torch.Tensor):
        """One bounce. Returns (v_new, hit, (face, idx_u, idx_w))."""
        t, h, face = self._ray_box(v)
        idx_u, idx_w = self._tile_index(h, face)
        w = self.T[face, idx_u, idx_w]
        v_new = 2.0 * torch.dot(w, v) * w - v
        v_new = v_new / v_new.norm()
        return v_new, h, (face, idx_u, idx_w)

    def run(self, v0: torch.Tensor, K: int):
        """K bounces. Returns (v_final, orbit)."""
        v = v0.clone()
        orbit = []
        for _ in range(K):
            v, h, tile = self.step(v)
            orbit.append((h.cpu(), tile))
        return v, orbit

    def run_multi(self, v_batch: torch.Tensor, K: int):
        """GPU-parallel: (n_colors, 3) → list of orbits."""
        n = v_batch.shape[0]
        v = v_batch.clone()
        orbits = [[] for _ in range(n)]
        for _ in range(K):
            # Vectorized: all bounces in parallel
            inv = 1.0 / v.abs().max(dim=1, keepdim=True).values
            t = inv.expand(n, 3)
            h = t * v
            faces = v.abs().argmax(dim=1)
            # Tile lookup (vectorized)
            idx_u = torch.clamp(((h[:, 1] + 1) * 0.5 * self.N).long(), 0, self.N - 1)
            idx_w = torch.clamp(((h[:, 2] + 1) * 0.5 * self.N).long(), 0, self.N - 1)
            # (Simplified: use face-0 mapping; full version selects per-face)
            w = self.T[faces, idx_u, idx_w]
            v = 2.0 * (v * w).sum(dim=1, keepdim=True) * w - v
            v = v / v.norm(dim=1, keepdim=True)
            for i in range(n):
                orbits[i].append((h[i].cpu(), (faces[i].item(), idx_u[i].item(), idx_w[i].item())))
        return orbits, v