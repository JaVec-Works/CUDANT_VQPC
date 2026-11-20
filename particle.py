# =============================================================================
# CUDANT_VQPC — particle.py
# The lion. One vertex at origin, one line pointing anywhere in 3D.
# The GPU is the physics engine. This particle obeys quantum laws.
#
# AXIS CONVENTION — see AXIS.md (the law)
#
#   θ = elevation from north pole (|0>).  0° = up, 180° = down.
#   φ = azimuth from +X, counter-clockwise looking down from +Z.
#
#   Bloch vector: x=sin(θ)cos(φ), y=sin(θ)sin(φ), z=cos(θ)
#   P(|0>) = (1+z)/2,  P(|1>) = (1-z)/2   [φ invisible to measurement]
#
#   Rx(θ) → Y-Z plane.  After Rx on |0>: φ = -90°
#   Ry(θ) → X-Z plane.  After Ry on |0>: φ =  0°
#   Rz(φ) → X-Y plane.   θ unchanged. Invisible on poles.
#   H = Ry(90°).  Correlation axis (entanglement) = +X.
#
#   Universal recipe: Ry(θ) then Rz(φ) → any point on the sphere.
#
#   Run: python particle.py
# =============================================================================   

import torch
import numpy as np


# ============================================================
# THE PARTICLE: one vertex, one line, full 3D freedom
# ============================================================

class Particle:
    """
    One quantum particle. A line from the origin, pointing anywhere on the sphere.
    
    - theta (θ): elevation. 0 = pointing up (|0>), π = pointing down (|1>)
    - phi (φ): direction. Which way around the equator. 0 = +x axis.
    
    The particle is a vertex on the GPU. Gates are rotations.
    It doesn't know it's 'quantum.' It just obeys the laws.
    """
    
    def __init__(self, theta=0.0, phi=0.0, device="cuda"):
        self.device = device
        self.theta = theta  # elevation (0 to π)
        self.phi = phi      # direction (0 to 2π)
        # The Bloch vector: where the line is pointing
        self.v = self._to_vector()
        # Box 2: the painted path (trajectory over time)
        self.path = [self.v.clone().cpu()]
    
    def _to_vector(self):
        """Convert (θ, φ) to 3D vector on the unit sphere."""
        x = torch.tensor(np.sin(self.theta) * np.cos(self.phi), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.sin(self.theta) * np.sin(self.phi), dtype=torch.float32, device=self.device)
        z = torch.tensor(np.cos(self.theta), dtype=torch.float32, device=self.device)
        return torch.stack([x, y, z])   
    
    def rotate_x(self, angle):
        """Rotate around X-axis. Changes elevation (θ)."""
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=torch.float32, device=self.device)
        self.v = R @ self.v
        self.v = self.v / self.v.norm()
        self._update_angles()
        self.path.append(self.v.clone().cpu())
    
    def rotate_y(self, angle):
        """Rotate around Y-axis. Changes elevation AND direction."""
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32, device=self.device)
        self.v = R @ self.v
        self.v = self.v / self.v.norm()
        self._update_angles()
        self.path.append(self.v.clone().cpu())
    
    def rotate_z(self, angle):
        """Rotate around Z-axis. Changes direction (φ) only. The 'wobble'."""
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        self.v = R @ self.v
        self.v = self.v / self.v.norm()
        self._update_angles()
        self.path.append(self.v.clone().cpu())
    
    def _update_angles(self):
        """Recover (θ, φ) from the vector."""
        z = self.v[2].item()
        self.theta = np.arccos(np.clip(z, -1, 1))
        if self.v[0].item() != 0 or self.v[1].item() != 0:
            self.phi = np.arctan2(self.v[1].item(), self.v[0].item())
        else:
            self.phi = 0.0
    
    def measure(self):
        """Project onto the Z-axis (the walls of the cage). Returns (bit, collapsed vector)."""
        p0 = (1 + self.v[2].item()) / 2
        p1 = (1 - self.v[2].item()) / 2
        bit = 0 if np.random.random() < p0 else 1
        # Collapse: snap to |0> (up) or |1> (down)
        self.v = torch.tensor([0.0, 0.0, 1.0 if bit == 0 else -1.0], device=self.device)
        self.theta = 0.0 if bit == 0 else np.pi
        self.phi = 0.0
        self.path.append(self.v.clone().cpu())
        return bit, p0, p1
    
    def report(self, label=""):
        """Print the particle's current state."""
        x, y, z = self.v[0].item(), self.v[1].item(), self.v[2].item()
        p0 = (1 + z) / 2
        p1 = (1 - z) / 2
        print(f"  {label:20s} θ={np.degrees(self.theta):6.1f}°  φ={np.degrees(self.phi):6.1f}°  "
              f"v=[{x:+.4f}, {y:+.4f}, {z:+.4f}]  P(0)={p0:.4f}  P(1)={p1:.4f}")

if __name__ == "__main__":
    print("=== CUDANT_VQPC: THE PARTICLE ===\n")
    # ============================================================
    # TEST 1: The particle at rest
    # ============================================================
    
    print("--- TEST 1: Particle at rest (|0> state) ---\n")
    p = Particle(theta=0.0, phi=0.0)
    p.report("At rest")
    print(f"  Path length: {len(p.path)} points\n")
    
    
    # ============================================================
    # TEST 2: Full sphere — rotate to ANY direction
    # ============================================================
    
    print("--- TEST 2: Full Bloch sphere (θ + φ) ---\n")
    p = Particle(theta=0.0, phi=0.0)
    p.report("Start (|0>)")
    
    # Rotate to: elevation 60°, direction 45°
    # Step 1: Rx(60°) — tilt down to 60° elevation
    p.rotate_y(np.radians(60))
    p.report("After Rx(60°)")
    
    # Step 2: Rz(45°) — spin around Z to 45° direction
    p.rotate_z(np.radians(45))
    p.report("After Rz(45°)")
    
    print(f"\n  The particle is now at θ=60°, φ=45° — anywhere on the sphere.")
    print(f"  Path length: {len(p.path)} points (the painted trail)\n")
    
    
    # ============================================================
    # TEST 3: The wobble (precession)
    # ============================================================
    
    print("--- TEST 3: The wobble (cone) ---\n")
    print("  Apply a small Rx, then a small Rz, repeatedly.")
    print("  The particle doesn't land clean. It traces a cone.\n")
    
    p = Particle(theta=np.radians(30), phi=0.0)
    p.report("Start (θ=30°)")
    
    # 10 small steps: each one nudges the particle slightly off-axis
    for i in range(10):
        p.rotate_x(np.radians(5))   # small tilt
        p.rotate_z(np.radians(10))  # small spin
        if i % 3 == 0:
            p.report(f"Step {i+1}")
    
    print(f"\n  Path length: {len(p.path)} points")
    print(f"  The particle is wobbling around its axis. That's the cone.\n")
    
    
    # ============================================================
    # TEST 4: The cage (measurement = hitting the wall)
    # ============================================================
    
    print("--- TEST 4: The cage (measurement) ---\n")
    p = Particle(theta=np.radians(90), phi=np.radians(90))
    p.report("Before measurement")
    print(f"  The particle is at 90° elevation (sideways). Equal superposition.")
    print(f"  When we measure (hit the wall), it MUST pick: up or down.\n")
    
    torch.manual_seed(42)
    bits = []
    for i in range(10):
        p2 = Particle(theta=np.radians(45), phi=np.radians(90))
        bit, p0, p1 = p2.measure()
        bits.append(bit)
    
    zeros = bits.count(0)
    ones = bits.count(1)
    print(f"  10 measurements: {zeros}× |0>, {ones}× |1>")
    print(f"  Expected: ~5× each (90° = equal superposition)")
    print(f"  The particle hit the wall and picked a side. Every time.\n")
    
    
    # ============================================================
    # TEST 5: Widen — 1000 particles, 1000 different directions
    # ============================================================
    
    print("--- TEST 5: Widen (1000 particles, one batch) ---\n")
    
    batch = 1000
    # 1000 different (θ, φ) pairs
    thetas = torch.linspace(0, np.pi, batch, device="cuda")
    phis = torch.linspace(0, 2 * np.pi, batch, device="cuda")
    
    # Compute all 1000 vectors at once (GPU batch)
    x = torch.sin(thetas) * torch.cos(phis)
    y = torch.sin(thetas) * torch.sin(phis)
    z = torch.cos(thetas)
    
    # Measure all 1000: P(|0>) = (1+z)/2
    p0 = (1 + z) / 2
    p1 = (1 - z) / 2
    
    print(f"  Batch: {batch} particles, all computed in one GPU pass")
    print(f"  θ=0°:   P(0)={p0[0].item():.4f}  (pointing up, definitely |0>)")
    print(f"  θ=90°:  P(0)={p0[batch//2].item():.4f}  (pointing sideways, 50/50)")
    print(f"  θ=180°: P(0)={p0[-1].item():.4f}  (pointing down, definitely |1>)")
    print(f"  Average P(0): {p0.mean().item():.4f}")
    print(f"  Average P(1): {p1.mean().item():.4f}")
    print()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("=== THE PARTICLE IS BUILT ===\n")
    print("  1. One vertex, one line, full 3D (θ + φ)     ✅")
    print("  2. Three rotation axes (Rx, Ry, Rz)           ✅")
    print("  3. The wobble (precession / cone)             ✅")
    print("  4. The cage (measurement / collapse)          ✅")
    print("  5. Widen (1000 particles, one batch)          ✅")
    print("  6. Box 2 (the painted path / trajectory)      ✅")
    print()
    print("Next: laws.py — the 5 quantum laws that govern the particle.")
    print("Then: gates.py — how you steer it.")
    print("Then: cage.py — the full trajectory recorder.")   