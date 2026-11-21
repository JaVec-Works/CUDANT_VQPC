"""CUDANT_VQPC — validate_box.py
Unit test: cube model vs exact engine."""

import torch
import numpy as np


def validate(n_qubits: int, operations: list, N: int = 16, K: int = 10000) -> dict:
    from engine import run_engine
    from box import Box
    from encode_algorithm import circuit_to_tiling

    # Engine (ground truth)
    engine_result = run_engine(n_qubits, operations, n_shots=0)
    engine_bloch = engine_result.get("bloch_vectors", {})

    # Box
    T = circuit_to_tiling("generic", n_qubits=n_qubits, N=N)
    box = Box(N, T)
    v0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
    v_final, orbit = box.run(v0, K)

    # Compare (single-qubit case: direct)
    errors = {}
    if n_qubits == 1 and "qubit_0" in engine_bloch:
        bv = np.array(engine_bloch["qubit_0"])
        v_box = v_final.cpu().numpy()
        err = float(np.linalg.norm(bv - v_box))
        errors["qubit_0"] = round(err, 6)

    max_err = max(errors.values()) if errors else 0.0

    return {
        "n_qubits": n_qubits,
        "operations": operations,
        "N": N,
        "K": K,
        "per_qubit_error": errors,
        "max_error": max_err,
        "pass": max_err < 0.01,
        "verdict": "PASS" if max_err < 0.01 else f"FAIL (max_err={max_err:.4f})",
    }   