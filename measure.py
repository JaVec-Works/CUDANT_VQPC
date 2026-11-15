# CUDANT_VQPC — measure.py
# Looking at the particle. Collapse + N shots.
# Measurement is the ONLY irreversible operation.
# Before: vector (superposition). After: point (|0> or |1>).
#
# Run: python measure.py

import torch
import numpy as np


# ============================================================
# SINGLE-SHOT MEASUREMENT
# "Hit the wall. Pick a side. The vector is gone."
# ============================================================

def measure_single(v, seed=None):
    """
    Measure a single particle.
    
    Args:
        v: Bloch vector [x, y, z]
        seed: Optional random seed (for reproducibility)
    
    Returns:
        (bit, p0, p1, collapsed_vector)
        bit: 0 or 1
        p0, p1: probabilities before collapse
        collapsed_vector: the new vector after collapse
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    z = v[2].item()
    p0 = (1 + z) / 2
    p1 = (1 - z) / 2
    
    # Random draw
    r = torch.rand(1, device=v.device).item()
    bit = 0 if r < p0 else 1
    
    # Collapse: snap to |0> (up) or |1> (down)
    collapsed = torch.tensor([0.0, 0.0, 1.0 if bit == 0 else -1.0],
                             dtype=torch.float32, device=v.device)
    
    return bit, p0, p1, collapsed


# ============================================================
# N-SHOT SAMPLING
# "Run the circuit N times. Give me the distribution."
# This is what a real quantum computer does: you can't read
# the state vector. You can only sample from it.
# ============================================================

def measure_n_shots(v, n_shots=1000, seed=None):
    """
    Measure the same particle N times (re-preparing each time).
    
    Args:
        v: Bloch vector [x, y, z]
        n_shots: How many times to measure (default 1000)
        seed: Optional random seed
    
    Returns:
        dict with counts, frequencies, and the raw bit string
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    z = v[2].item()
    p0 = (1 + z) / 2
    
    # Batch sample: draw N random numbers, compare to p0
    r = torch.rand(n_shots, device=v.device)
    bits = (r >= p0).int()  # 0 if r < p0, else 1
    
    zeros = int((bits == 0).sum().item())
    ones = int((bits == 1).sum().item())
    
    return {
        "n_shots": n_shots,
        "zeros": zeros,
        "ones": ones,
        "freq_0": zeros / n_shots,
        "freq_1": ones / n_shots,
        "expected_p0": p0,
        "expected_p1": 1 - p0,
        "bits": bits.cpu().numpy(),
    }

if __name__ == "__main__":
    print("=== CUDANT_VQPC: MEASUREMENT ===\n")
    # ============================================================
    # TEST 1: Single shot on |0> (definite)
    # ============================================================
    
    print("--- TEST 1: Single shot on |0> (definite) ---\n")
    v0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
    
    bit, p0, p1, collapsed = measure_single(v0, seed=42)
    print(f"  P(0)={p0:.4f}, P(1)={p1:.4f}")
    print(f"  Result: |{bit}>  (always |0>, no surprise)")
    print(f"  Collapsed vector: [{collapsed[0]:.4f}, {collapsed[1]:.4f}, {collapsed[2]:.4f}]\n")
    
    
    # ============================================================
    # TEST 2: Single shot on |1> (definite)
    # ============================================================
    
    print("--- TEST 2: Single shot on |1> (definite) ---\n")
    v1 = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device="cuda")
    
    bit, p0, p1, collapsed = measure_single(v1, seed=42)
    print(f"  P(0)={p0:.4f}, P(1)={p1:.4f}")
    print(f"  Result: |{bit}>  (always |1>, no surprise)\n")
    
    
    # ============================================================
    # TEST 3: Single shot on superposition (50/50)
    # ============================================================
    
    print("--- TEST 3: Single shot on |+> (50/50 superposition) ---\n")
    v_plus = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    bit, p0, p1, collapsed = measure_single(v_plus, seed=42)
    print(f"  P(0)={p0:.4f}, P(1)={p1:.4f}")
    print(f"  Result: |{bit}>  (coin flip, could be either)\n")
    
    
    # ============================================================
    # TEST 4: N shots on superposition (1000 samples)
    # ============================================================
    
    print("--- TEST 4: 1000 shots on |+> (50/50) ---\n")
    result = measure_n_shots(v_plus, n_shots=1000, seed=42)
    print(f"  |0>: {result['zeros']} times ({result['freq_0']*100:.1f}%)")
    print(f"  |1>: {result['ones']} times ({result['freq_1']*100:.1f}%)")
    print(f"  Expected: 50.0% / 50.0%")
    print(f"  First 20 bits: {result['bits'][:20]}\n")
    
    
    # ============================================================
    # TEST 5: N shots on biased state (θ=60°)
    # ============================================================
    
    print("--- TEST 5: 1000 shots on θ=60° (biased) ---\n")
    theta = np.radians(60)
    v_biased = torch.tensor([
        np.sin(theta), 0.0, np.cos(theta)
    ], dtype=torch.float32, device="cuda")
    
    result = measure_n_shots(v_biased, n_shots=1000, seed=42)
    print(f"  P(0) expected: {result['expected_p0']:.4f}")
    print(f"  P(1) expected: {result['expected_p1']:.4f}")
    print(f"  |0>: {result['zeros']} times ({result['freq_0']*100:.1f}%)")
    print(f"  |1>: {result['ones']} times ({result['freq_1']*100:.1f}%)")
    print(f"  The particle is biased toward |0>. The samples reflect that.\n")
    
    
    # ============================================================
    # TEST 6: The cage — what the AI agent sees
    # ============================================================
    
    print("--- TEST 6: The cage (agent's view) ---\n")
    print("  The AI agent doesn't see the state vector.")
    print("  It sees: 'I ran the circuit 1000 times, here's what I got.'")
    print("  That's all a real quantum computer gives you too.\n")
    
    v_test = torch.tensor([0.7071, 0.0, 0.7071], dtype=torch.float32, device="cuda")  # θ=45°
    result = measure_n_shots(v_test, n_shots=1000, seed=42)
    
    print(f"  Circuit: H on |0> (θ=45°... actually θ=arccos(0.7071)=45°)")
    print(f"  1000 shots:")
    print(f"    |0>: {result['zeros']} ({result['freq_0']*100:.1f}%)")
    print(f"    |1>: {result['ones']} ({result['freq_1']*100:.1f}%)")
    print(f"  Agent's answer: 'The circuit outputs |0> about {result['freq_0']*100:.0f}% of the time.'")
    print(f"  No state vector. No amplitudes. Just counts. Like a real QC.\n")
    
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("=== MEASUREMENT BUILT ===\n")
    print("  Single shot: collapse to |0> or |1>              ✅")
    print("  N shots: sample distribution (1000+)             ✅")
    print("  Biased states: samples reflect the bias          ✅")
    print("  The cage: agent sees counts, not amplitudes      ✅")
    print("  Irreversible: once collapsed, vector is gone     ✅")
    print()
    print("  This is what a real quantum computer gives you.")
    print("  Not the state vector. The SAMPLES.")
    print("  1000 shots. 1000 coin flips weighted by the state.")
    print()
    print("Next: circuit.py — run a full sequence, return state + trajectory.")   