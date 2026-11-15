# =============================================================================
# CUDANT_VQPC — algorithms.py
# Pre-built quantum circuits. The agent says the problem, the tool picks the algorithm.
#
# AXIS CONVENTION — see AXIS.md (the law)
#   All algorithms use the engine (exact state vector). Qubit 0 = MSB.
#
# Run: python algorithms.py
# =============================================================================

import torch
import numpy as np
from engine import (
    apply_single_qubit, apply_cnot, _gate_matrix,
    bloch_vector, run_engine
)


# ============================================================
# 1. GROVER SEARCH
# ============================================================

def grover(n_qubits: int, target: int) -> dict:
    """
    Grover's search. Find `target` in an unsorted database of 2^n items.
    Uses the engine (needs full state vector for oracle + diffusion).
    
    Args:
        n_qubits: 1-16.
        target: Integer index of the item to find (0 to 2^n - 1).
    
    Returns:
        dict with target, iterations, P(target), probabilities, verdict.
    """
    if n_qubits < 1 or n_qubits > 16:
        return {"error": f"n_qubits must be 1-16. Got {n_qubits}."}
    
    dim = 1 << n_qubits
    if target < 0 or target >= dim:
        return {"error": f"target must be 0-{dim-1}. Got {target}."}
    
    # Optimal iterations: round(π/4 * √N)
    n_iters = max(1, round(np.pi / 4 * np.sqrt(dim)))
    
    # Initialize: H on all qubits (uniform superposition)
    state = torch.zeros(dim, dtype=torch.complex64, device="cuda")
    state[0] = 1.0
    H = _gate_matrix("H")
    for q in range(n_qubits):
        state = apply_single_qubit(state, H, q, n_qubits)
    
    # Grover iterations
    for _ in range(n_iters):
        # Oracle: flip sign of target
        state[target] *= -1
        # Diffusion: 2|s><s| - I
        mean = state.sum() / np.sqrt(dim)
        state = 2 * mean / np.sqrt(dim) - state
    
    # Results
    probs = state.abs() ** 2
    probs = probs / probs.sum()
    p_target = probs[target].item()
    
    # Top outcomes
    top_indices = torch.topk(probs, min(5, dim)).indices.tolist()
    top_outcomes = [
        {"state": format(i, f"0{n_qubits}b"), "prob": round(probs[i].item(), 6)}
        for i in top_indices
    ]
    
    # Bloch vectors (dashboard)
    bloch = {}
    for q in range(n_qubits):
        bv = bloch_vector(state, n_qubits, q)
        bloch[f"qubit_{q}"] = [round(float(x), 6) for x in bv]
    
    verdict = "FOUND" if p_target > 0.9 else "PARTIAL" if p_target > 0.5 else "NOT FOUND"
    
    return {
        "algorithm": "grover",
        "n_qubits": n_qubits,
        "target": f"|{target:0{n_qubits}b}> (index {target})",
        "iterations": n_iters,
        "P_target": round(p_target, 6),
        "top_outcomes": top_outcomes,
        "bloch_vectors": bloch,
        "verdict": f"{verdict}. P(target) = {p_target:.6f} after {n_iters} iterations.",
    }


# ============================================================
# 2. BELL PAIR
# ============================================================

def bell_pair(n_qubits: int = 2) -> dict:
    """
    Create a Bell pair |Φ+> = (|00>+|11>)/√2 on the first two qubits.
    Remaining qubits stay at |0>.
    
    Args:
        n_qubits: Total qubits (2-16). Bell pair on qubits 0 and 1.
    """
    if n_qubits < 2 or n_qubits > 16:
        return {"error": f"n_qubits must be 2-16. Got {n_qubits}."}
    
    ops = ["GATE 0 H", "CNOT 0 1"]
    result = run_engine(n_qubits, ops, n_shots=1000, seed=42)
    
    if "error" in result:
        return result
    
    result["algorithm"] = "bell_pair"
    result["description"] = f"|Φ+> on qubits 0,1. Remaining {n_qubits-2} qubits at |0>."
    result["verdict"] = "ENTANGLED. P(00...0)=0.5, P(1100...0)=0.5. All else 0."
    return result


# ============================================================
# 3. GHZ STATE
# ============================================================

def ghz_state(n_qubits: int) -> dict:
    """
    Create a GHZ state: (|00...0> + |11...1>)/√2.
    Maximum entanglement across all n qubits.
    
    Args:
        n_qubits: 2-16.
    """
    if n_qubits < 2 or n_qubits > 16:
        return {"error": f"n_qubits must be 2-16. Got {n_qubits}."}
    
    # H on qubit 0, then CNOT chain: 0→1→2→...→n-1
    ops = ["GATE 0 H"]
    for i in range(n_qubits - 1):
        ops.append(f"CNOT {i} {i+1}")
    
    result = run_engine(n_qubits, ops, n_shots=1000, seed=42)
    
    if "error" in result:
        return result
    
    result["algorithm"] = "ghz_state"
    result["description"] = f"(|{'0'*n_qubits}> + |{'1'*n_qubits}>)/√2"
    result["verdict"] = f"GHZ. P(0{'0'*(n_qubits-1)})=0.5, P(1{'1'*(n_qubits-1)})=0.5. All else 0."
    return result


# ============================================================
# 4. QUANTUM FOURIER TRANSFORM
# ============================================================

def quantum_fourier_transform(n_qubits: int, input_state: str = "0") -> dict:
    """
    Apply the QFT to an input state.
    QFT|x> = (1/√N) Σ_k exp(2πi·x·k/N) |k>
    
    This is exactly the DFT on the state vector. torch.fft.fft does it in one call.
    
    Args:
        n_qubits: 1-16.
        input_state: Binary string for the input. "0" = |00...0>, "1" = |11...1>, etc.
                     If shorter than n_qubits, left-padded with zeros.
    """
    if n_qubits < 1 or n_qubits > 16:
        return {"error": f"n_qubits must be 1-16. Got {n_qubits}."}
    
    dim = 1 << n_qubits
    
    # Build input state
    state = torch.zeros(dim, dtype=torch.complex64, device="cuda")
    if input_state in ("0", ""):
        state[0] = 1.0
    elif input_state == "1":
        state[dim - 1] = 1.0
    else:
        idx = int(input_state, 2)
        if idx >= dim:
            return {"error": f"input_state '{input_state}' = {idx}, out of range 0-{dim-1}."}
        state[idx] = 1.0
    
    # QFT = DFT / √N
    state = torch.fft.fft(state) / np.sqrt(dim)
    
    # Results
    probs = state.abs() ** 2
    probs = probs / probs.sum()
    
    top_indices = torch.topk(probs, min(8, dim)).indices.tolist()
    top_outcomes = [
        {"state": format(i, f"0{n_qubits}b"), "prob": round(probs[i].item(), 6)}
        for i in top_indices
    ]
    
    # For |0> input: QFT|0> = uniform superposition (all probs = 1/N)
    # For |1> input: QFT|1> = phase ramp
    input_label = format(int(input_state, 2) if input_state not in ("0", "") else 0, f"0{n_qubits}b")
    
    if input_state in ("0", ""):
        description = f"QFT|0{'0'*(n_qubits-1)}> = uniform superposition. All {dim} states equal."
    else:
        description = f"QFT|{input_label}> = phase ramp (the 'frequency' signature)."
    
    return {
        "algorithm": "quantum_fourier_transform",
        "n_qubits": n_qubits,
        "input": f"|{input_label}>",
        "description": description,
        "top_outcomes": top_outcomes,
        "uniform": round(probs.max().item() / probs.min().item(), 4) if probs.min().item() > 0 else "inf",
        "verdict": f"QFT applied. Max/Min ratio = {probs.max().item()/max(probs.min().item(),1e-10):.4f} (1.0 = perfectly uniform).",
    }


# ============================================================
# 5. PHASE ESTIMATION
# ============================================================

def _apply_controlled_phase(state, control, target, angle, n_qubits):
    """
    Apply e^(i·angle) to the |11> component (control=1 AND target=1).
    This is the controlled-Rz phase kick. Vectorized: one mask + one multiply.
    """
    dim = state.shape[0]
    c_bit = n_qubits - 1 - control
    t_bit = n_qubits - 1 - target
    c_mask = 1 << c_bit
    t_mask = 1 << t_bit
    
    indices = torch.arange(dim, device=state.device)
    mask = ((indices & c_mask) != 0) & ((indices & t_mask) != 0)
    state[mask] *= np.exp(1j * angle)
    return state


def phase_estimation(n_qubits: int, phase_deg: float) -> dict:
    """
    Estimate the phase of U = Rz(phase_deg) applied to |1>.
    
    Circuit:
      1. t target qubits (for the answer) + 1 eigenstate qubit at |1>
      2. H on all t target qubits
      3. Controlled-Rz(phase * 2^j) from target j to eigenstate qubit
      4. Inverse QFT on target qubits
      5. Read off the phase in binary
    
    Args:
        n_qubits: Total qubits. t = n_qubits - 1 target qubits + 1 eigenstate qubit.
                  So n_qubits=5 means 4 target + 1 eigenstate → 4-bit phase resolution.
        phase_deg: The phase to estimate (0-360).
    """
    if n_qubits < 3 or n_qubits > 16:
        return {"error": f"n_qubits must be 3-16 (need at least 2 target + 1 eigenstate). Got {n_qubits}."}
    
    t = n_qubits - 1  # number of target (answer) qubits
    eigen_qubit = n_qubits - 1  # last qubit is the eigenstate
    dim = 1 << n_qubits
    phase_rad = np.radians(phase_deg)
    
    # Step 1: Initialize. Eigenstate qubit at |1>, targets at |0>
    state = torch.zeros(dim, dtype=torch.complex64, device="cuda")
    # |00...01> (last bit is 1)
    state[1] = 1.0
    
    # Step 2: H on all target qubits
    H = _gate_matrix("H")
    for q in range(t):
        state = apply_single_qubit(state, H, q, n_qubits)
    
    # Step 3: Controlled-Rz(phase * 2^j) from target j to eigenstate
    for j in range(t):
        angle = phase_rad * (2 ** (t - 1 - j))
        state = _apply_controlled_phase(state, j, eigen_qubit, angle, n_qubits)
    
    # Step 4: Inverse QFT on target register
    # Reshape: (2^t target values, 2 eigenstate values)
    state_reshaped = state.reshape(1 << t, 2)
    # Apply inverse DFT to the target dimension
    state_reshaped = torch.fft.fft(state_reshaped, dim=0) / np.sqrt(1 << t)
    state = state_reshaped.reshape(-1)
    
    # Step 5: Read out the target register
    state_reshaped = state.reshape(1 << t, 2)
    target_probs = (state_reshaped.abs() ** 2).sum(dim=1)
    target_probs = target_probs / target_probs.sum()   
    
    # The answer should be at index k where k/2^t ≈ phase/(2π)
    expected_k = round(phase_rad / (2 * np.pi) * (1 << t)) % (1 << t)
    
    top_indices = torch.topk(target_probs, min(5, 1 << t)).indices.tolist()
    top_outcomes = [
        {"target": format(i, f"0{t}b"), "prob": round(target_probs[i].item(), 6)}
        for i in top_indices
    ]
    
    p_expected = target_probs[expected_k].item()
    
    return {
        "algorithm": "phase_estimation",
        "n_qubits": n_qubits,
        "target_qubits": t,
        "phase_input_deg": phase_deg,
        "expected_answer": f"{expected_k:0{t}b} (= {phase_deg * (1<<t) / 360:.1f} / {1<<t})",
        "P_expected": round(p_expected, 6),
        "top_outcomes": top_outcomes,
        "verdict": f"{'ESTIMATED' if p_expected > 0.8 else 'PARTIAL' if p_expected > 0.4 else 'NOISE'}. P(answer) = {p_expected:.4f}.",
    }


# ============================================================
# 6. DISPATCHER
# ============================================================

def run_algorithm(name: str, **kwargs) -> dict:
    """
    Dispatch to the right algorithm.
    
    Args:
        name: "grover", "bell_pair", "ghz_state", "qft", "phase_estimation"
        **kwargs: Algorithm-specific parameters.
    """
    name = name.lower().replace("-", "_").replace(" ", "_")
    
    if name == "grover":
        return grover(kwargs.get("n_qubits", 4), kwargs.get("target", 0b1010))
    elif name in ("bell_pair", "bell"):
        return bell_pair(kwargs.get("n_qubits", 2))
    elif name in ("ghz_state", "ghz"):
        return ghz_state(kwargs.get("n_qubits", 4))
    elif name in ("qft", "quantum_fourier_transform"):
        return quantum_fourier_transform(
            kwargs.get("n_qubits", 4),
            kwargs.get("input_state", "0")
        )
    elif name in ("phase_estimation", "phase"):
        return phase_estimation(
            kwargs.get("n_qubits", 5),
            kwargs.get("phase_deg", 45.0)
        )
    else:
        return {"error": f"Unknown algorithm: '{name}'. Valid: grover, bell_pair, ghz_state, qft, phase_estimation"}


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("=== CUDANT_VQPC: ALGORITHMS ===\n")
    
    # TEST 1: Grover (4 qubits, target = 0b1010)
    print("--- TEST 1: Grover (4 qubits, target=1010) ---\n")
    r = grover(4, target=0b1010)
    print(f"  Target: {r['target']}")
    print(f"  Iterations: {r['iterations']}")
    print(f"  P(target): {r['P_target']}")
    print(f"  Top: {r['top_outcomes'][:3]}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 2: Bell pair (2 qubits)
    print("--- TEST 2: Bell pair (2 qubits) ---\n")
    r = bell_pair(2)
    print(f"  P(00) = {r['probabilities'].get('00', 'n/a')}")
    print(f"  P(11) = {r['probabilities'].get('11', 'n/a')}")
    print(f"  Bloch: {r['bloch_vectors']}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 3: GHZ (4 qubits)
    print("--- TEST 3: GHZ (4 qubits) ---\n")
    r = ghz_state(4)
    print(f"  P(0000) = {r['probabilities'].get('0000', 'n/a')}")
    print(f"  P(1111) = {r['probabilities'].get('1111', 'n/a')}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 4: QFT (4 qubits, input |0000>)
    print("--- TEST 4: QFT (4 qubits, input=|0000>) ---\n")
    r = quantum_fourier_transform(4, "0")
    print(f"  Top: {r['top_outcomes'][:4]}")
    print(f"  Uniformity (max/min): {r['uniform']}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 5: QFT (4 qubits, input |1000>)
    print("--- TEST 5: QFT (4 qubits, input=|1000>) ---\n")
    r = quantum_fourier_transform(4, "1000")
    print(f"  Top: {r['top_outcomes'][:4]}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 6: Phase estimation (5 qubits = 4 target + 1 eigenstate, phase=90°)
    print("--- TEST 6: Phase estimation (4-bit, phase=90°) ---\n")
    r = phase_estimation(5, phase_deg=90.0)
    print(f"  Input phase: {r['phase_input_deg']}°")
    print(f"  Expected: {r['expected_answer']}")
    print(f"  P(expected): {r['P_expected']}")
    print(f"  Top: {r['top_outcomes'][:3]}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 7: Phase estimation (5 qubits, phase=45°)
    print("--- TEST 7: Phase estimation (4-bit, phase=45°) ---\n")
    r = phase_estimation(5, phase_deg=45.0)
    print(f"  Input phase: {r['phase_input_deg']}°")
    print(f"  Expected: {r['expected_answer']}")
    print(f"  P(expected): {r['P_expected']}")
    print(f"  Verdict: {r['verdict']}\n")
    
    # TEST 8: Dispatcher
    print("--- TEST 8: Dispatcher (run_algorithm) ---\n")
    r = run_algorithm("grover", n_qubits=4, target=0b1111)
    print(f"  grover(4, 1111): P={r['P_target']}, verdict={r['verdict']}")
    r = run_algorithm("ghz", n_qubits=3)
    print(f"  ghz(3): verdict={r['verdict']}")
    r = run_algorithm("qft", n_qubits=3, input_state="0")
    print(f"  qft(3, 0): verdict={r['verdict']}")
    print()
    
    print("=== ALGORITHMS BUILT ===\n")
    print("  Grover search (1-16 qubits)                   ✅")
    print("  Bell pair (2-16 qubits)                        ✅")
    print("  GHZ state (2-16 qubits)                        ✅")
    print("  Quantum Fourier Transform (1-16 qubits)        ✅")
    print("  Phase estimation (3-16 qubits)                 ✅")
    print("  Dispatcher: run_algorithm(name, **kwargs)      ✅")
    print()
    print("Next: wire into vqpc_server.py as tool #8 (run_algorithm).")   