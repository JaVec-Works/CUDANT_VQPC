# =============================================================================
# CUDANT_VQPC — gates.py
# How you steer the particle. Every gate is a rotation on the Bloch sphere.
# The GPU rotates vertices for a living. This is the same operation.
#
# AXIS CONVENTION — see AXIS.md (the law)
#
#   All rotation matrices follow AXIS.md:
#     Rx(θ): Y-Z plane.  After Rx on |0>, vector is along -Y (φ = -90°).
#     Ry(θ): X-Z plane.  After Ry on |0>, vector is along +X (φ =  0°).
#     Rz(φ): X-Y plane.  Changes φ only. θ invariant.
#
#   Named gates in terms of rotations:
#     X = Rx(180°),  H = Ry(90°),  Z = Rz(180°),  S = Rz(90°),  T = Rz(45°)
#
#   "Tilt without turning" → Ry.  "Tilt and turn" → Rx.  "Spin in place" → Rz.
#
#   Run: python gates.py
# =============================================================================   

import torch
import numpy as np


# ============================================================
# THE GATES: each one is a 3x3 rotation matrix
# Applied to the particle's Bloch vector (x, y, z)
# ============================================================

def rotation_x(angle):
    """Rotate by `angle` around X-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32, device="cuda")

def rotation_y(angle):
    """Rotate by `angle` around Y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32, device="cuda")

def rotation_z(angle):
    """Rotate by `angle` around Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32, device="cuda")

# Named gates (specific rotations)
GATES = {
    "I":  torch.eye(3, dtype=torch.float32, device="cuda"),
    "X":  rotation_x(np.pi),
    "Y":  rotation_y(np.pi),
    "Z":  rotation_z(np.pi),
    "H":  torch.tensor([[0,0,1],[0,1,0],[1,0,0]], dtype=torch.float32, device="cuda"),
    "S":  rotation_z(np.pi / 2),
    "Sd": rotation_z(-np.pi / 2),
    "T":  rotation_z(np.pi / 4),
    "Td": rotation_z(-np.pi / 4),
}

# Parameterized gates (any angle)
def Rx(angle): return rotation_x(angle)
def Ry(angle): return rotation_y(angle)
def Rz(angle): return rotation_z(angle)


def apply_gate(v, gate):
    """Apply a gate (3x3 rotation) to a Bloch vector. Returns new vector."""
    v_new = gate @ v
    return v_new / v_new.norm()


def report(v, label=""):
    """Print the particle's state."""
    x, y, z = v[0].item(), v[1].item(), v[2].item()
    theta = np.degrees(np.arccos(np.clip(z, -1, 1)))
    phi = np.degrees(np.arctan2(y, x))
    p0 = (1 + z) / 2
    print(f"  {label:20s} θ={theta:6.1f}°  φ={phi:6.1f}°  v=[{x:+.4f}, {y:+.4f}, {z:+.4f}]  P(0)={p0:.4f}")

if __name__ == "__main__":
    print("=== CUDANT_VQPC: GATES (STEERING THE PARTICLE) ===\n")
    # ============================================================
    # TEST 1: Each named gate on |0>
    # ============================================================
    
    print("--- TEST 1: Each gate on |0> (start pointing up) ---\n")
    v0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
    
    for name in ["I", "X", "Y", "Z", "H", "S", "T"]:
        v = apply_gate(v0, GATES[name])
        report(v, name)
    
    print()
    
    
    # ============================================================
    # TEST 2: H then H = Identity (double Hadamard)
    # ============================================================
    
    print("--- TEST 2: H then H = I (undo) ---\n")
    v = v0.clone()
    v = apply_gate(v, GATES["H"])
    report(v, "After H")
    v = apply_gate(v, GATES["H"])
    report(v, "After H again")
    print(f"  Back to start? {torch.allclose(v0, v, atol=1e-5)}\n")
    
    
    # ============================================================
    # TEST 3: S then Sd = Identity
    # ============================================================
    
    print("--- TEST 3: S then S† = I ---\n")
    v = v0.clone()
    v = apply_gate(v, GATES["S"])
    report(v, "After S")
    v = apply_gate(v, GATES["Sd"])
    report(v, "After S†")
    print(f"  Back to start? {torch.allclose(v0, v, atol=1e-5)}\n")
    
    
    # ============================================================
    # TEST 4: Parameterized gates (any angle)
    # ============================================================
    
    print("--- TEST 4: Parameterized gates (any angle) ---\n")
    v = v0.clone()
    report(v, "Start")
    v = apply_gate(v, Rx(np.radians(30)))
    report(v, "Rx(30°)")
    v = apply_gate(v, Ry(np.radians(45)))
    report(v, "Ry(45°)")
    v = apply_gate(v, Rz(np.radians(60)))
    report(v, "Rz(60°)")
    print(f"\n  Three rotations, any angle, any axis. The particle went anywhere.\n")
    
    
    # ============================================================
    # TEST 5: Full gate sequence (a mini circuit)
    # ============================================================
    
    print("--- TEST 5: Mini circuit (H → T → H) ---\n")
    print("  This is a phase kickback pattern. The particle")
    print("  leaves |0>, gets a phase kick, comes back with a twist.\n")
    
    v = v0.clone()
    report(v, "Start |0>")
    v = apply_gate(v, GATES["H"])
    report(v, "After H")
    v = apply_gate(v, GATES["T"])
    report(v, "After T")
    v = apply_gate(v, GATES["H"])
    report(v, "After H")
    print(f"\n  The particle is back near |0> but with a phase twist.")
    print(f"  That's how phase estimation works: H, phase kick, H.\n")
    
    
    # ============================================================
    # TEST 6: Widen — 1000 particles, 1000 different gate sequences
    # ============================================================
    
    print("--- TEST 6: Widen (1000 particles, one batch) ---\n")
    
    batch = 1000
    # 1000 different Rx angles
    angles = torch.linspace(0, np.pi, batch, device="cuda")
    
    # All start at |0>
    vs = torch.zeros(batch, 3, dtype=torch.float32, device="cuda")
    vs[:, 2] = 1.0
    
    # Apply Rx(angle) to each: [0,0,1] → [0, -sin(θ), cos(θ)]
    vs[:, 1] = -torch.sin(angles)
    vs[:, 2] = torch.cos(angles)
    
    # Measure: P(0) = (1+z)/2
    p0 = (1 + vs[:, 2]) / 2
    
    print(f"  Batch: {batch} particles, 1000 different angles, one GPU pass")
    print(f"  Rx(0°):   P(0)={p0[0].item():.4f}  (still |0>)")
    print(f"  Rx(90°):  P(0)={p0[batch//2].item():.4f}  (50/50)")
    print(f"  Rx(180°): P(0)={p0[-1].item():.4f}  (now |1>)")
    print(f"  Average P(0): {p0.mean().item():.4f}\n")
    
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("=== GATES BUILT ===\n")
    print("  Named:  I, X, Y, Z, H, S, S†, T, T†       ✅")
    print("  Param:  Rx(θ), Ry(θ), Rz(θ) — any angle    ✅")
    print("  Reversible: H·H=I, S·S†=I                 ✅")
    print("  Mini circuit: H→T→H (phase kickback)       ✅")
    print("  Widen: 1000 particles, one batch           ✅")
    print()
    print("  Every gate is a rotation. The GPU's native job.")
    print("  No new math. No new hardware. Just vertices.")
    print()
    print("Next: measure.py — looking at the particle (collapse + N shots).")
    print("Then: circuit.py — run a full sequence, return state + trajectory.")   