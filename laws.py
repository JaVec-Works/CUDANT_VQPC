# CUDANT_VQPC — laws.py
# The 5 laws that govern the particle.
# The particle doesn't know it's "quantum." It just obeys these rules.
# Quantum behavior is what EMERGES from the rules.
#
# Run: python laws.py

import torch
import numpy as np

print("=== CUDANT_VQPC: THE 5 LAWS ===\n")

# ============================================================
# LAW 1: SUPERPOSITION
# "The particle can be in multiple states at once."
# In code: the state is a vector, not a single point.
# ============================================================

print("--- LAW 1: SUPERPOSITION ---\n")
print("  A particle at θ=90° is NOT |0> AND NOT |1>.")
print("  It's a vector pointing sideways. Both at once.")
print("  You can't say 'it's 0' or 'it's 1' until you look.\n")

# A particle in superposition
theta = np.radians(90)
phi = 0.0
v = torch.tensor([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
], dtype=torch.float32, device="cuda")

p0 = (1 + v[2].item()) / 2
p1 = (1 - v[2].item()) / 2
print(f"  Vector: [{v[0].item():+.4f}, {v[1].item():+.4f}, {v[2].item():+.4f}]")
print(f"  P(|0>) = {p0:.4f}, P(|1>) = {p1:.4f}")
print(f"  It's BOTH. Until measured.\n")


# ============================================================
# LAW 2: UNITARY EVOLUTION
# "The particle moves smoothly. It never jumps. Every move is reversible."
# In code: gates are rotation matrices. Rotations are reversible.
# ============================================================

print("--- LAW 2: UNITARY EVOLUTION ---\n")
print("  Every gate is a rotation. Rotations are reversible.")
print("  Rotate 30° one way, rotate 30° the other, you're back.")
print("  The particle NEVER teleports. It always moves along the sphere.\n")

# Prove it: rotate +30°, then -30°, back to start
v_start = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")

c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
R_plus = torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32, device="cuda")
R_minus = torch.tensor([[1,0,0],[0,c,s],[0,-s,c]], dtype=torch.float32, device="cuda")

v_after = R_plus @ v_start
v_back = R_minus @ v_after

print(f"  Start:  [{v_start[0]:.4f}, {v_start[1]:.4f}, {v_start[2]:.4f}]")
print(f"  After +30°: [{v_after[0]:.4f}, {v_after[1]:.4f}, {v_after[2]:.4f}]")
print(f"  After -30°: [{v_back[0]:.4f}, {v_back[1]:.4f}, {v_back[2]:.4f}]")
print(f"  Back to start? {torch.allclose(v_start, v_back, atol=1e-5)}\n")


# ============================================================
# LAW 3: ENTANGLEMENT
# "Two particles can become one object. You can't describe them separately."
# In code: the CNOT "rod." After CNOT, the pair is one state, not two.
# ============================================================

print("--- LAW 3: ENTANGLEMENT ---\n")
print("  Two particles, connected by a rod (CNOT).")
print("  After the rod connects them, they're ONE object.")
print("  You can't say 'particle A is here' and 'particle B is there.'")
print("  You can only say 'the PAIR is in this state.'\n")

# Bell state: (|00> + |11>) / sqrt(2)
# In vector terms: neither particle has its own state.
# The pair has a joint state.
print("  Bell state |Φ+> = (|00> + |11>) / √2")
print("  Measure both: 50% → |00>, 50% → |11>")
print("  NEVER |01> or |10>. They're locked.\n")
print("  Particle A alone: 50% up, 50% down. (No information.)")
print("  Particle B alone: 50% up, 50% down. (No information.)")
print("  Particle A + B together: ALWAYS the same. (Perfect correlation.)\n")
print("  That's entanglement. The whole is more than the sum of parts.\n")


# ============================================================
# LAW 4: MEASUREMENT (COLLAPSE)
# "Looking at the particle forces it to pick one state."
# In code: project the vector onto the Z-axis. Snap to |0> or |1>.
# ============================================================

print("--- LAW 4: MEASUREMENT (COLLAPSE) ---\n")
print("  Before measurement: the particle is a vector (superposition).")
print("  After measurement: the particle is a POINT (|0> or |1>).")
print("  The vector SNAPS to the nearest wall of the cage.\n")
print("  This is the ONLY irreversible operation.")
print("  You can undo a rotation. You CANNOT undo a measurement.\n")

# Show it
v_super = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")  # θ=90°
p0 = (1 + v_super[2].item()) / 2
print(f"  Before: v=[{v_super[0]:.4f}, {v_super[1]:.4f}, {v_super[2]:.4f}]  P(0)={p0:.2f}  (superposition)")

# Collapse (deterministic for demo: pick |0>)
v_collapsed = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
print(f"  After:  v=[{v_collapsed[0]:.4f}, {v_collapsed[1]:.4f}, {v_collapsed[2]:.4f}]  P(0)=1.00  (collapsed)")
print(f"  The vector is GONE. Replaced by a point. Irreversible.\n")


# ============================================================
# LAW 5: INTERFERENCE
# "Paths add or cancel. Two particles can reinforce or destroy each other."
# In code: complex amplitudes add. Same phase = reinforce. Opposite = cancel.
# ============================================================

print("--- LAW 5: INTERFERENCE ---\n")
print("  Two paths to the same destination.")
print("  If they arrive in phase → they ADD (constructive).")
print("  If they arrive out of phase → they CANCEL (destructive).")
print("  This is how Grover's algorithm works: amplify the answer,")
print("  cancel everything else.\n")

# Demo: two amplitudes
# Path A: amplitude 0.5 (phase 0)
# Path B: amplitude 0.5 (phase 0) → constructive
# Path B': amplitude 0.5 (phase π) → destructive

amp_a = 0.5 + 0j
amp_b_same = 0.5 + 0j
amp_b_opp = -0.5 + 0j  # phase flip = π

result_same = amp_a + amp_b_same
result_opp = amp_a + amp_b_opp

print(f"  Path A: amplitude = {amp_a}")
print(f"  Path B (same phase): amplitude = {amp_b_same}")
print(f"  A + B (same phase) = {result_same}  → P = {abs(result_same)**2:.2f}  (REINFORCED)")
print()
print(f"  Path B' (opposite phase): amplitude = {amp_b_opp}")
print(f"  A + B' (opposite phase) = {result_opp}  → P = {abs(result_opp)**2:.2f}  (CANCELLED)")
print()
print("  Grover uses this: flip the phase of the wrong answers,")
print("  then reflect. Wrong answers cancel. Right answer amplifies.\n")


# ============================================================
# SUMMARY
# ============================================================

print("=== THE 5 LAWS ===\n")
print("  1. SUPERPOSITION    → The particle is a vector, not a point")
print("  2. UNITARY          → Moves are smooth, reversible rotations")
print("  3. ENTANGLEMENT     → Two particles can become one object")
print("  4. MEASUREMENT      → Looking snaps the vector to a wall (irreversible)")
print("  5. INTERFERENCE     → Paths add or cancel (the engine of Grover)")
print()
print("  These 5 rules are the ENTIRE physics engine.")
print("  Everything else — Grover, phase estimation, QFT —")
print("  is what EMERGES when you manipulate a particle that obeys them.")
print()
print("  The GPU is the physics engine.")
print("  The particle is the qubit.")
print("  The laws are quantum mechanics.")
print("  No cryogenics. No decoherence. No $10 billion.")   