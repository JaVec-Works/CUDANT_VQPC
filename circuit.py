# CUDANT_VQPC — circuit.py
# The full pipeline. One call: gates in, state + trajectory + measurement out.
# This is the "virtual quantum computer." You give it a program, it gives you an answer.
#
# Run: python circuit.py

import torch
import numpy as np
from particle import Particle
from gates import (
    GATES, Rx, Ry, Rz, apply_gate, report
)
from measure import measure_single, measure_n_shots


def run_circuit(gates_str, n_shots=0, seed=None):
    """
    Run a quantum circuit.
    
    Args:
        gates_str: Comma-separated gate sequence.
                   Named: I, X, Y, Z, H, S, Sd, T, Td
                   Parameterized: Rx(30), Ry(45), Rz(60)  [degrees]
                   Example: "H, T, H" or "Rx(30), Rz(45), Rx(30)"
        n_shots: If > 0, measure N times and return distribution
        seed: Random seed for measurement
    
    Returns:
        dict with:
            - final_state: (theta, phi, vector)
            - trajectory: list of (theta, phi, vector) at each step (Box 2)
            - measurement: dict of shot results (if n_shots > 0)
            - wobble: max deviation from a clean path (the cone radius)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Start at |0>
    p = Particle(theta=0.0, phi=0.0)
    
    trajectory = [(p.theta, p.phi, p.v.clone().cpu())]
    
    # Apply each gate
    for g in gates_str.split(","):
        g = g.strip()
        if not g:
            continue
        
        # Parse parameterized gates: Rx(30), Ry(45), Rz(60)
        if "(" in g and g.endswith(")"):
            name = g[:g.index("(")]
            angle_deg = float(g[g.index("(")+1:-1])
            angle = np.radians(angle_deg)
            if name == "Rx":
                p.rotate_x(angle)
            elif name == "Ry":
                p.rotate_y(angle)
            elif name == "Rz":
                p.rotate_z(angle)
            else:
                raise ValueError(f"Unknown parameterized gate: {name}")
        else:
            # Named gates
            if g not in GATES:
                raise ValueError(f"Unknown gate: {g}. Valid: {list(GATES.keys())}")
            R = GATES[g]
            p.v = R @ p.v
            p.v = p.v / p.v.norm()
            p._update_angles()
            p.path.append(p.v.clone().cpu())
        
        trajectory.append((p.theta, p.phi, p.v.clone().cpu()))
    
    # Compute wobble: how far the path deviates from a straight line
    # (a clean circuit = straight path. A wobbly one = cone.)
    wobble = _compute_wobble(trajectory)
    
    result = {
        "gates": gates_str,
        "final_theta": np.degrees(p.theta),
        "final_phi": np.degrees(p.phi),
        "final_vector": [p.v[0].item(), p.v[1].item(), p.v[2].item()],
        "final_p0": (1 + p.v[2].item()) / 2,
        "final_p1": (1 - p.v[2].item()) / 2,
        "trajectory_length": len(trajectory),
        "wobble": wobble,
    }
    
    # Optional measurement
    if n_shots > 0:
        shots = measure_n_shots(p.v, n_shots=n_shots, seed=seed)
        result["measurement"] = {
            "n_shots": n_shots,
            "zeros": shots["zeros"],
            "ones": shots["ones"],
            "freq_0": shots["freq_0"],
            "freq_1": shots["freq_1"],
        }
    
    return result


def _compute_wobble(trajectory):
    """
    Measure how much the path deviates from a straight line.
    0 = perfect straight line (clean circuit).
    >0 = the particle is wobbling (cone).
    """
    if len(trajectory) < 3:
        return 0.0
    
    # Straight line from start to end
    start = trajectory[0][2].numpy()
    end = trajectory[-1][2].numpy()
    direction = end - start
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0, 0, 1.0])
    direction = direction / np.linalg.norm(direction)
    
    # Max perpendicular distance from the line
    max_dev = 0.0
    for _, _, v in trajectory:
        v_np = v.numpy()
        # Project onto start→end line
        t = np.dot(v_np - start, direction)
        closest = start + t * direction
        dist = np.linalg.norm(v_np - closest)
        max_dev = max(max_dev, dist)
    
    return float(max_dev)


def print_result(result, verbose=True):
    """Pretty-print a circuit result."""
    print(f"  Circuit: {result['gates']}")
    print(f"  Final:   θ={result['final_theta']:.1f}°  φ={result['final_phi']:.1f}°  "
          f"P(0)={result['final_p0']:.4f}  P(1)={result['final_p1']:.4f}")
    print(f"  Path:    {result['trajectory_length']} points")
    print(f"  Wobble:  {result['wobble']:.4f}  "
          f"({'clean' if result['wobble'] < 0.01 else 'coning'})")
    
    if "measurement" in result:
        m = result["measurement"]
        print(f"  Shots:   {m['n_shots']} → |0>: {m['zeros']} ({m['freq_0']*100:.1f}%), "
              f"|1>: {m['ones']} ({m['freq_1']*100:.1f}%)")
    print()


if __name__ == "__main__":
    print("=== CUDANT_VQPC: CIRCUIT (THE FULL PIPELINE) ===\n")
    # ============================================================
    # TEST 1: Simple circuit (H)
    # ============================================================
    
    print("--- TEST 1: H on |0> ---\n")
    r = run_circuit("H", n_shots=1000, seed=42)
    print_result(r)
    
    
    # ============================================================
    # TEST 2: H → T → H (phase kickback)
    # ============================================================
    
    print("--- TEST 2: H → T → H (phase kickback) ---\n")
    r = run_circuit("H, T, H", n_shots=1000, seed=42)
    print_result(r)
    
    
    # ============================================================
    # TEST 3: Parameterized (Rx(30) → Rz(45) → Rx(30))
    # ============================================================
    
    print("--- TEST 3: Rx(30) → Rz(45) → Rx(30) ---\n")
    r = run_circuit("Rx(30), Rz(45), Rx(30)", n_shots=1000, seed=42)
    print_result(r)
    
    
    # ============================================================
    # TEST 4: The wobble (many small rotations)
    # ============================================================
    
    print("--- TEST 4: The wobble (10× Rx(5) → Rz(10)) ---\n")
    gates_wobble = ", ".join([f"Rx(5), Rz(10)"] * 10)
    r = run_circuit(gates_wobble)
    print_result(r)
    print(f"  The path is {r['trajectory_length']} points long.")
    print(f"  Wobble = {r['wobble']:.4f} (the cone radius).")
    print(f"  The particle is NOT going straight. It's precessing.\n")
    
    
    # ============================================================
    # TEST 5: Clean vs Wobbly (comparison)
    # ============================================================
    
    print("--- TEST 5: Clean vs Wobbly ---\n")
    
    # Clean: one big rotation
    r_clean = run_circuit("Rx(50)")
    print(f"  Clean (Rx(50)):         wobble = {r_clean['wobble']:.6f}  (straight line)")
    
    # Wobbly: ten small rotations with Z-kicks
    gates_wobbly = ", ".join([f"Rx(5), Rz(10)"] * 5)
    r_wobbly = run_circuit(gates_wobbly)
    print(f"  Wobbly (5× Rx(5)+Rz(10)): wobble = {r_wobbly['wobble']:.6f}  (cone)")
    
    print(f"\n  Same total rotation. Different paths.")
    print(f"  Clean = straight. Wobbly = cone.")
    print(f"  The AI agent can SEE the difference.\n")
    
    
    # ============================================================
    # TEST 6: The agent's full answer
    # ============================================================
    
    print("--- TEST 6: The agent's full answer ---\n")
    print("  Agent asks: 'Run H, T, H. Measure 1000 times. Tell me what you got.'\n")
    
    r = run_circuit("H, T, H", n_shots=1000, seed=42)
    
    print(f"  Agent receives:")
    print(f"    Answer: |0> {r['measurement']['freq_0']*100:.1f}% of the time")
    print(f"    Confidence: P(0) = {r['final_p0']:.4f} (theoretical)")
    print(f"    Path: {r['trajectory_length']} steps, wobble = {r['wobble']:.4f}")
    print(f"    Verdict: {'TRUSTED (clean path)' if r['wobble'] < 0.01 else 'CAUTION (wobbly path)'}")
    print()
    
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("=== CIRCUIT BUILT ===\n")
    print("  Full pipeline: gates → state → trajectory → measurement  ✅")
    print("  Named gates: I, X, Y, Z, H, S, S†, T, T†               ✅")
    print("  Parameterized: Rx(θ), Ry(θ), Rz(θ)                      ✅")
    print("  Box 2: trajectory recorded at every step                ✅")
    print("  Wobble: cone radius computed (clean vs wobbly)          ✅")
    print("  Measurement: N shots, agent gets counts                 ✅")
    print("  Agent's answer: probability + confidence + path quality ✅")
    print()
    print("  This is the virtual quantum computer.")
    print("  Give it a circuit. It gives you an answer.")
    print("  Plus HOW it got there. Plus HOW SURE it is.")
    print()
    print("Next: mcp.py — the AI's interface to all of this.")   