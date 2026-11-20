# CUDANT_VQPC — vqpc_server.py
# The AI's interface to the virtual quantum computer.
# 10 tools: run_circuit, measure_state, wobble_report, list_gates, run_cage, run_engine, run_algorithm, train_particle, replay, help      
#
# Run: python vqpc_server.py

import json
import os
import configparser
import torch
import numpy as np
from fastmcp import FastMCP
from circuit import run_circuit
from measure import measure_n_shots, measure_single
from gates import GATES, rotation_x, rotation_y, rotation_z
from engine import run_engine as _run_engine
from cage import Cage

# ============================================================
# READ settings.conf + set VRAM cap
# ============================================================

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.conf")

if os.path.exists(_conf_path):
    _cfg = configparser.ConfigParser()
    _cfg.read(_conf_path)
    _fraction = float(_cfg.get("gpu", "memory_fraction", fallback="0.06"))
    torch.cuda.set_per_process_memory_fraction(_fraction)
    _reserve = _cfg.get("gpu", "vram_reserve_mb", fallback="500")
    _max_lions = int(_cfg.get("limits", "max_lions", fallback="1024"))
    print(f"  VRAM cap: {_fraction*100:.1f}% ({_reserve} MB for CUDANT_VQPC)")
    print(f"  Max lions: {_max_lions}")
else:
    _max_lions = 8
    print("  WARNING: settings.conf not found. Run 'python benchmark.py' first.")
    print("  No VRAM cap set. Defaulting to 8 lions.")   

mcp = FastMCP("CUDANT-VQPC")


@mcp.tool()
def run_circuit_tool(gates: str, n_shots: int = 1000, seed: int = 42) -> str:
    """
    Run a single-particle quantum circuit. Returns final state, trajectory, wobble, and measurement.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        gates: Comma-separated gate sequence.
               Named: I, X, Y, Z, H, S, Sd, T, Td
               Parameterized: Rx(30), Ry(45), Rz(60) [degrees]
        n_shots: Number of measurement shots (0 = no measurement). Default 1000.
        seed: Random seed. Default 42.

    Usage examples:
        "H" → Hadamard on |0>, gives 50/50
        "H, T, H" → phase kickback, gives ~85/15
        "Rx(30), Rz(45), Rx(30)" → three rotations, any direction
        "X" → bit flip, |0> becomes |1>
    """
    try:
        result = run_circuit(gates, n_shots=n_shots, seed=seed)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "run_circuit", "advice": f"Check gate string. Valid named gates: I, X, Y, Z, H, S, Sd, T, Td. Valid parameterized: Rx(deg), Ry(deg), Rz(deg). Example: 'H, T, H' or 'Rx(30), Rz(45)'."}, indent=2)

    output = {
        "circuit": result["gates"],
        "final_state": {
            "theta_deg": round(result["final_theta"], 2),
            "phi_deg": round(result["final_phi"], 2),
            "vector": [round(x, 6) for x in result["final_vector"]],
            "P_0": round(result["final_p0"], 6),
            "P_1": round(result["final_p1"], 6),
        },
        "trajectory": {
            "steps": result["trajectory_length"],
            "wobble": round(result["wobble"], 6),
            "quality": "CLEAN" if result["wobble"] < 0.01 else "WOBBLING",
        },
    }

    if "measurement" in result:
        m = result["measurement"]
        output["measurement"] = {
            "n_shots": m["n_shots"],
            "zeros": m["zeros"],
            "ones": m["ones"],
            "freq_0": round(m["freq_0"], 4),
            "freq_1": round(m["freq_1"], 4),
        }

    if "measurement" in result:
        wobble = result["wobble"]
        freq = result["measurement"]["freq_0"]
        theory = result["final_p0"]
        gap = abs(freq - theory)
        if wobble < 0.01 and gap < 0.05:
            output["verdict"] = "TRUSTED. Clean path, samples match theory."
        elif wobble < 0.01:
            output["verdict"] = "TRUSTED. Clean path. Samples within noise."
        else:
            output["verdict"] = f"CAUTION. Wobbly path (wobble={result['wobble']:.4f})."
    else:
        output["verdict"] = "OK. No measurement requested."

    return json.dumps(output, indent=2)


@mcp.tool()
def measure_state(theta_deg: float, phi_deg: float, n_shots: int = 1000, seed: int = 42) -> str:
    """
    Measure a particle at a specific position on the Bloch sphere.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        theta_deg: Elevation in degrees. 0=|0>, 90=sideways(50/50), 180=|1>.
        phi_deg: Direction in degrees. 0=+x axis.
        n_shots: Number of measurement shots. Default 1000.
        seed: Random seed. Default 42.

    Usage examples:
        theta_deg=0, phi_deg=0 → always |0>
        theta_deg=90, phi_deg=0 → 50/50 (equal superposition)
        theta_deg=60, phi_deg=0 → 75% |0>, 25% |1> (biased)
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    v = torch.tensor([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=torch.float32, device="cuda")

    result = measure_n_shots(v, n_shots=n_shots, seed=seed)

    output = {
        "state": {
            "theta_deg": theta_deg,
            "phi_deg": phi_deg,
            "vector": [round(v[0].item(), 6), round(v[1].item(), 6), round(v[2].item(), 6)],
            "P_0_theoretical": round(result["expected_p0"], 6),
            "P_1_theoretical": round(result["expected_p1"], 6),
        },
        "measurement": {
            "n_shots": n_shots,
            "zeros": result["zeros"],
            "ones": result["ones"],
            "freq_0": round(result["freq_0"], 4),
            "freq_1": round(result["freq_1"], 4),
        },
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def wobble_report(gates: str) -> str:
    """
    Analyze path quality without measuring. Tells you if the particle moved clean or wobbled.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        gates: Comma-separated gate sequence (same format as run_circuit).

    Returns:
        Wobble metric + quality verdict (CLEAN, SLIGHT, MODERATE, HEAVY).

    Usage examples:
        "Rx(50)" → CLEAN (straight line)
        "Rx(5), Rz(10), Rx(5), Rz(10)" → SLIGHT (minor cone)
        "H, T, H" → WOBBLING (phase kick curves the path)
    """
    try:
        result = run_circuit(gates, n_shots=0)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "wobble_report", "advice": f"Check gate string. Example: 'Rx(50)' or 'H, T, H'."}, indent=2)

    wobble = result["wobble"]
    if wobble < 0.01:
        quality = "CLEAN"
        detail = "Straight path. No precession."
    elif wobble < 0.1:
        quality = "SLIGHT"
        detail = "Minor cone. Some phase accumulation."
    elif wobble < 0.5:
        quality = "MODERATE"
        detail = "Visible cone. The particle is precessing."
    else:
        quality = "HEAVY"
        detail = "Large cone. Significant precession."

    output = {
        "circuit": result["gates"],
        "trajectory_steps": result["trajectory_length"],
        "wobble": round(wobble, 6),
        "quality": quality,
        "detail": detail,
        "final_state": {
            "theta_deg": round(result["final_theta"], 2),
            "phi_deg": round(result["final_phi"], 2),
            "P_0": round(result["final_p0"], 6),
        },
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def list_gates() -> str:
    """
    List all available gates with descriptions and examples.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.
    """
    gates_info = {
        "Named gates": {
            "I": "Identity. Do nothing.",
            "X": "Bit flip. |0>→|1>, |1>→|0>.",
            "Y": "Bit + phase flip.",
            "Z": "Phase flip. Adds π to |1>.",
            "H": "Hadamard. |0>→(|0>+|1>)/√2. Creates superposition.",
            "S": "Phase gate. Adds π/2 to |1>.",
            "Sd": "Conjugate phase. Adds -π/2 to |1>.",
            "T": "Pi/8 gate. Adds π/4 to |1>.",
            "Td": "Conjugate pi/8. Adds -π/4 to |1>.",
        },
        "Parameterized gates (any angle in degrees)": {
            "Rx(angle)": "Rotate around X. Changes elevation.",
            "Ry(angle)": "Rotate around Y. Changes elevation + direction.",
            "Rz(angle)": "Rotate around Z. Changes direction only.",
        },
        "Syntax": "Comma-separated. Examples: 'H, T, H' | 'Rx(30), Rz(45)' | 'X'",
    }

    return json.dumps(gates_info, indent=2)


@mcp.tool()
def run_cage(n_lions: int, operations: list, n_shots: int = 1000, seed: int = 42) -> str:
    f"""
    Run a multi-particle circuit in the cage. Multiple lions, same laws, some entangled.   

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        n_lions: Number of particles (1-{_max_lions}).
        operations: List of operation strings, applied in order.
        n_shots: Measurement shots (0 = skip). Default 1000.
        seed: Random seed. Default 42.

    Operation format (each string in the list):
        "GATE <id> <gate>"   → apply named gate. Example: "GATE 0 H"
        "RX <id> <angle>"    → rotate around X. Example: "RX 1 60"
        "RY <id> <angle>"    → rotate around Y. Example: "RY 2 45"
        "RZ <id> <angle>"    → rotate around Z. Example: "RZ 3 90"
        "CNOT <ctrl> <tgt>"  → connect two lions (entanglement). Example: "CNOT 0 1"

    Usage examples:
        n_lions=2, operations=["GATE 0 H", "CNOT 0 1"] → Bell pair
        n_lions=3, operations=["GATE 0 H", "CNOT 0 1", "RX 2 45"] → entangled pair + free lion
        n_lions=4, operations=["GATE 0 H", "CNOT 0 1", "GATE 1 H", "CNOT 1 2"] → chain
    """
    if n_lions < 1 or n_lions > _max_lions:
        return json.dumps({"error": f"n_lions must be 1-{_max_lions}. Got {n_lions}."})   

    try:
        cage = Cage(n_lions)

        for op in operations:
            parts = op.strip().split()
            if not parts:
                continue
            cmd = parts[0].upper()

            if cmd == "GATE" and len(parts) == 3:
                cage.apply_gate(int(parts[1]), parts[2])
            elif cmd == "RX" and len(parts) == 3:
                cage.apply_rx(int(parts[1]), float(parts[2]))   
            elif cmd == "RY" and len(parts) == 3:
                cage.apply_ry(int(parts[1]), float(parts[2]))
            elif cmd == "RZ" and len(parts) == 3:
                cage.apply_rz(int(parts[1]), float(parts[2]))
            elif cmd == "CNOT" and len(parts) == 3:
                cage.apply_cnot(int(parts[1]), int(parts[2]))
            else:
                return json.dumps({"error": f"Unrecognized: '{op}'. Valid: GATE, RX, RY, RZ, CNOT"})
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "run_cage", "advice": f"Check operations format. Valid: 'GATE <id> <name>', 'RX <id> <deg>', 'RY <id> <deg>', 'RZ <id> <deg>', 'CNOT <ctrl> <tgt>'. n_lions={n_lions}."}, indent=2)   

    lions = []
    for i, p in enumerate(cage.particles):
        x, y, z = p.v[0].item(), p.v[1].item(), p.v[2].item()
        p0 = (1 + z) / 2
        links = []
        for (c, t) in cage.rods:
            if c == i: links.append(f"→{t}")
            if t == i: links.append(f"←{c}")
        lions.append({
            "id": i,
            "theta_deg": round(np.degrees(p.theta), 2),
            "phi_deg": round(np.degrees(p.phi), 2),
            "vector": [round(x, 6), round(y, 6), round(z, 6)],
            "P_0": round(p0, 6),
            "P_1": round(1 - p0, 6),
            "links": links if links else "free",
            "trajectory_points": len(p.path),
        })

    entanglement = [f"{c}↔{t}" for (c, t) in cage.rods]

    output = {
        "cage": {
            "n_lions": n_lions,
            "n_rods": len(cage.rods),
            "entanglement": entanglement if entanglement else "none",
        },
        "lions": lions,
    }

    if n_shots > 0:
        results = cage.measure_all(n_shots=n_shots, seed=seed)
        output["measurement"] = {}
        for i in range(n_lions):
            output["measurement"][f"lion_{i}"] = {
                "zeros": results[i]["zeros"],
                "ones": results[i]["ones"],
                "freq_0": round(results[i]["zeros"] / n_shots, 4),
                "freq_1": round(results[i]["ones"] / n_shots, 4),
            }

    return json.dumps(output, indent=2)

@mcp.tool()
def run_engine(n_qubits: int, operations: list, n_shots: int = 1000, seed: int = 42) -> str:
    """
    Run a multi-qubit circuit on the exact state vector engine (1-16 qubits).
    Use this for 3+ qubits where entanglement is too complex for the geometric cage.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    WHEN TO USE:
        1-2 qubits → use run_cage (faster, shows trajectory + wobble)
        3-16 qubits → use run_engine (exact amplitudes, full entanglement)

    Args:
        n_qubits: Number of qubits (1-16).
        operations: List of operation strings (same format as run_cage).
        n_shots: Measurement shots (0 = skip). Default 1000.
        seed: Random seed. Default 42.

    Operation format (same as run_cage):
        "GATE <id> <gate>"   → I, X, Y, Z, H, S, Sd, T, Td
        "RX <id> <angle>"    → rotate around X (degrees)
        "RY <id> <angle>"    → rotate around Y (degrees)
        "RZ <id> <angle>"    → rotate around Z (degrees)
        "CNOT <ctrl> <tgt>"  → entangling gate

    Usage examples:
        n_qubits=4, operations=["GATE 0 H", "CNOT 0 1", "CNOT 1 2", "CNOT 2 3"]
        → 4-qubit GHZ. P(0000)=0.5, P(1111)=0.5.
    """
    try:
        result = _run_engine(n_qubits, operations, n_shots=n_shots, seed=seed)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "run_engine", "advice": f"Check operations format and qubit count (1-16). Valid: 'GATE <id> <name>', 'CNOT <ctrl> <tgt>'."}, indent=2)
    return json.dumps(result, indent=2)     

@mcp.tool()
def run_algorithm(name: str, n_qubits: int = 4, target: int = 0, phase_deg: float = 0.0, input_state: str = "0") -> str:
    """
    Run a pre-built quantum algorithm. The agent says the problem, this picks the circuit.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        name: Which algorithm. One of:
            "grover" — search an unsorted database. Set target to the item index.
            "bell_pair" — entanglement demo.
            "ghz_state" — multi-qubit entanglement.
            "qft" — quantum Fourier transform. Set input_state to a binary string.
            "phase_estimation" — estimate a phase angle. Set phase_deg.
        n_qubits: Number of qubits.
        target: For grover. The index to find (0 to 2^n - 1).
        phase_deg: For phase_estimation. The phase to estimate (0-360).
        input_state: For qft. Binary string for the input state. "0" = |00...0>.

    Usage examples:
        name="grover", n_qubits=4, target=10 → find |1010> in 16 items
        name="ghz_state", n_qubits=4 → 4-qubit GHZ
        name="qft", n_qubits=4, input_state="1000" → QFT of |1000>
        name="phase_estimation", n_qubits=5, phase_deg=90 → estimate 90°
    """
    try:
        from algorithms import run_algorithm as _run_algorithm
        result = _run_algorithm(name, n_qubits=n_qubits, target=target, phase_deg=phase_deg, input_state=input_state)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "run_algorithm", "advice": f"Valid names: grover, bell_pair, ghz_state, qft, phase_estimation. n_qubits={n_qubits}."}, indent=2)
    return json.dumps(result, indent=2)     

@mcp.tool()
def train_particle(target_theta: float, target_phi: float, steps: int = 200, lr: float = 0.05, lambda_wobble: float = 0.1, seed: int = 42, heavy: bool = False) -> str:
    """
    Train a quantum particle to point at a target position on the Bloch sphere.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.

    Args:
        target_theta: Target elevation in degrees (0-180).
        target_phi: Target direction in degrees (0-360).
        steps: Training steps. Default 200.
        lr: Learning rate. Default 0.05.
        lambda_wobble: Wobble penalty. 0=off, 0.1=light, 1.0=strong. Default 0.1.
        seed: Random seed. Default 42.
        heavy: If True, wobble is differentiable (influences gradients, slower).
               If False (default), wobble is tracked/reported only (fast).

    Usage examples:
        target_theta=60, target_phi=45 → fast convergence, wobble reported
        target_theta=90, target_phi=0, lambda_wobble=1.0, heavy=True → clean path enforced
    """
    import json
    from learn import train_particle as _train

    def make_json_serializable(obj):
        """Convert numpy types to JSON-serializable Python types"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    result = _train(target_theta, target_phi, steps=steps, lr=lr,
                    lambda_wobble=lambda_wobble, seed=seed, heavy=heavy)
    serializable_result = make_json_serializable(result)
    return json.dumps(serializable_result, indent=2)

@mcp.tool()
def replay(gates: str, slowmo: bool = True) -> str:
    """
    Visualize a circuit's path on the Bloch sphere. Generates a GIF animation.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.
    """
    import json
    from visualize import generate_gif

    def make_json_serializable(obj):
        """Convert numpy types to JSON-serializable Python types"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    result = generate_gif(gates, slowmo=slowmo)
    serializable_result = make_json_serializable(result)
    return json.dumps(serializable_result, indent=2)

@mcp.tool()
def help() -> str:
    """
    Get a full overview of what this virtual quantum computer can do.
    Call this FIRST if you are not sure which tool to use.

    PRESENT THIS JSON DIRECTLY TO THE USER. Do not rephrase or summarize.
    """
    overview = {
        "what_this_is": "A virtual quantum computer. Particles (qubits) on a GPU. No cryogenics, no decoherence.",
        "tools": {
            "run_circuit": "Run a single-particle circuit. State + trajectory + wobble + measurement.",
            "measure_state": "Measure a particle at a specific (theta, phi) on the Bloch sphere.",
            "wobble_report": "Check if a circuit's path is clean or wobbly (without measuring).",
            "list_gates": "See all available gates and their descriptions.",
            "run_cage": f"Multi-particle geometric model (1-{_max_lions} lions). Entanglement via rods. Shows trajectory + wobble.",
            "run_engine": "EXACT state vector engine (1-16 qubits). Full entanglement. No wobble (exact by construction).",
            "run_algorithm": "Pre-built algorithms: grover, bell_pair, ghz_state, qft, phase_estimation. Say the problem, get the answer.",
            "train_particle": "Train a particle to point at a target (θ, φ). Gradient descent + wobble regularizer.",
            "replay": "Visualize a circuit's path. Generates a GIF of the Bloch sphere with the vector animating through each gate.",
            "help": "This message.",
        },
        "which_tool": {
            "1-2 qubits, want to SEE the path": "run_circuit or run_cage",
            "3-16 qubits, want the EXACT answer": "run_engine",
            "checking path quality": "wobble_report",
            "not sure": "start with run_cage for intuition, confirm with run_engine",
            "I have a specific problem (search, entangle, transform, estimate)": "run_algorithm",
            "teach the particle to point somewhere": "train_particle",
            "I want to SEE the particle move": "replay",

        },
        "gate_syntax": {
            "run_circuit (1 qubit)": "Shorthand: 'H, T, H' or 'Rx(30), Rz(45)'",
            "run_cage / run_engine (multi-qubit)": "Explicit: ['GATE 0 H', 'CNOT 0 1', 'RX 2 45']",
            "warning": "Do NOT mix the two formats. 'H0' is NOT valid in run_circuit. 'GATE 0 H' is NOT valid in run_circuit.",
        },   
        
        "quick_start": {
            "step_1": "Call list_gates to see what gates are available.",
            "step_2": "Call run_circuit with a simple gate like 'H' to see a single particle in superposition.",
            "step_3": "Call run_cage with n_lions=2, operations=['GATE 0 H', 'CNOT 0 1'] to see entanglement.",
            "step_4": "Call run_engine with n_qubits=4, operations=['GATE 0 H', 'CNOT 0 1', 'CNOT 1 2', 'CNOT 2 3'] for exact GHZ.",
        },
        "key_concepts": {
            "particle": "A qubit. A vector on a sphere. Has elevation (theta) and direction (phi).",
            "gate": "A rotation. Moves the particle to a new position on the sphere.",
            "wobble": "How much the path curves. Clean = straight. Wobbly = cone (precession).",
            "cage": "A box holding multiple particles. Some free, some linked by rods (entanglement). Geometric approximation.",
            "engine": "Exact 2^n state vector. No approximation. Full entanglement structure visible in probabilities.",
            "measurement": "Looking at the particle. Forces it to pick |0> or |1>. Irreversible.",
            "verdict": "The system's own confidence assessment. TRUSTED or CAUTION.",
        },
    }   

    return json.dumps(overview, indent=2)


if __name__ == "__main__":
    print("=== CUDANT-VQPC MCP SERVER ===")
    print(f"Tools: run_circuit, measure_state, wobble_report, list_gates, run_cage, run_engine, run_algorithm, help")
    print(f"Max lions: {_max_lions} | VRAM cap: {int(_fraction*100)}%")
    print("Connecting via stdio...\n")
    mcp.run(transport="stdio")   