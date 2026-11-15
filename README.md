# CUDANT_VQPC

**A virtual quantum computer for AI. Runs on a consumer GPU. No cryogenics. No decoherence. No $10 billion.**

The GPU is already a quantum computer. It rotates vertices, projects them, and reports where they land. We changed the interpretation from "pixels on a screen" to "amplitudes in Hilbert space." Same hardware. Same math. Same speed.

> **Full install instructions for any level:** see INSTALL.md

## Quick setup (experienced)

```
git clone <repo-url>
cd CUDANT_VQPC
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python circuit.py    # watch the full pipeline run
```

## What it does

- A quantum **particle** (Bloch vector) that obeys 5 quantum laws
    
- Full Bloch sphere: any angle, any direction, any elevation (θ + φ)
    
- Gates are rotations — the GPU's native job
    
- **Trajectory recording** (Box 2): the painted path the particle takes
    
- **Wobble detection**: clean path vs. cone (precession)
    
- **Measurement**: single-shot collapse + N-shot sampling
    
- **Verdict**: the system tells the agent how much to trust its own answer
    
- Exposed to AI agents via MCP (4 tools)
    

## What makes it different

|   |   |   |   |
|---|---|---|---|
||CUDANT_VQPC|Real quantum computer|Classical simulator|
|Hardware|$500 GPU|$10B facility|HPC cluster|
|Temperature|Room|15 mK|Room|
|Decoherence|None|Constant|N/A|
|Trajectory|Yes (Box 2)|No (you only get samples)|No|
|Wobble detection|Yes|No|No|
|Self-verification|Verdict (TRUSTED / CAUTION)|No|No|
|Agent interface|MCP (4 tools)|None|CLI|
|Qubits|1 (single particle, full sphere)|100s (noisy)|16–32 (exact)|

## The 5 Laws

The particle doesn't "simulate" quantum mechanics. It **obeys** 5 rules. Quantum behavior emerges.

|   |   |
|---|---|
|Law|What it means|
|1. Superposition|The particle is a vector, not a point. It can be in multiple states at once.|
|2. Unitary Evolution|Every move is a smooth, reversible rotation. The particle never teleports.|
|3. Entanglement|Two particles can become one object. You can't describe them separately.|
|4. Measurement|Looking snaps the vector to a wall. Irreversible. The only non-unitary operation.|
|5. Interference|Paths add or cancel. The engine behind Grover, phase estimation, and QFT.|

## Files

|   |   |
|---|---|
|File|What it is|
|`particle.py`|The particle. Full Bloch sphere (θ+φ), trajectory, measurement.|
|`laws.py`|The 5 quantum laws. The physics engine.|
|`gates.py`|All gates: I, X, Y, Z, H, S, S†, T, T† + Rx(θ), Ry(θ), Rz(θ).|
|`measure.py`|Single-shot collapse + N-shot sampling.|
|`circuit.py`|Full pipeline: gates → state → trajectory → wobble → measurement.|
|`vqpc_server.py`|MCP server — 4 tools for AI agents.|
|`requirements.txt`|3 dependencies: torch, numpy, fastmcp.|

## MCP Tools (`vqpc_server.py`)

|   |   |
|---|---|
|Tool|What it does|
|`run_circuit`|Run a gate sequence. Returns state + trajectory + wobble + measurement + verdict.|
|`measure_state`|Measure a particle at a specific (θ, φ) position. Returns distribution.|
|`wobble_report`|Analyze path quality without measuring. Returns wobble metric + verdict.|
|`list_gates`|Full gate reference with descriptions.|

## The agent loop

```
Agent: "Run H, T, H. Measure 1000 times."
  → run_circuit("H, T, H", n_shots=1000)
  → P(0) = 0.8536, wobble = 1.36, verdict = "CAUTION (wobbly path)"
  → Measurement: 849× |0>, 151× |1>
Agent: "The answer is |0> 85% of the time, but the path is wobbly.
        The phase kick from T is curving the trajectory. Expected."
```

## Architecture

```
AI Agent (LLM)
    ↓ MCP (stdio)
vqpc_server.py  ← 4 tools
    ↓
circuit.py  ← full pipeline
    ↓
particle.py  ← the lion (Bloch vector, θ+φ)
gates.py  ← rotations (the GPU's native job)
measure.py  ← collapse + sampling
    ↓
NVIDIA GPU (consumer, 8 GB)
```

## The two boxes

|   |   |   |
|---|---|---|
|Box|Data|What it tells the agent|
|1 (Information)|Final state vector|"The answer is 85%|0>"|
|2 (Trajectory)|Path of the vector at each gate step|"Here's HOW it got there. And here's where it wobbled."|

Box 2 is what no other simulator gives you. The agent can diagnose _why_ an answer is what it is.

## Requirements

- Python 3.12
    
- PyTorch (CUDA)
    
- numpy
    
- fastmcp
    
- NVIDIA GPU (any, 4 GB+ is fine for single-particle)
    

## Related

- **CUDANT** (private) — 16-qubit exact state vector simulator + CUDANT-DL (learnable layer)
    
- **CUDANT-128** (public) — 128-qubit extension via CUDA-Q MPS with fidelity gate
    
- CUDANT_VQPC is the **geometry-first** approach: the particle obeys laws, behavior emerges