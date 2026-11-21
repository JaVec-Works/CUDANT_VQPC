# CUDANT_VQPC

**A virtual quantum computer for AI. Runs on a laptop GPU. No cryogenics. No decoherence. No $10 billion.**

The GPU is already a quantum computer. It rotates vertices, projects them, and reports where they land. We changed the interpretation from "pixels on a screen" to "amplitudes in Hilbert space." Same hardware. Same math. Same speed.

> **Full install instructions for any level:** see INSTALL.md

---

## Quick setup (experienced)

```bash
git clone <repo-url>
cd CUDANT_VQPC
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
python benchmark.py           # generates settings.conf (VRAM cap, max lions)
python circuit.py             # watch the full pipeline run
````

---

## What it does

- A quantum **particle** (Bloch vector) that obeys 5 quantum laws
    
- Full Bloch sphere: any angle, any direction, any elevation (θ + φ)
    
- Gates are rotations — the GPU's native job
    
- **Trajectory recording** (Box 2): the painted path the particle takes
    
- **Wobble detection**: clean path vs. cone (precession)
    
- **Measurement**: single-shot collapse + N-shot sampling
    
- **Exact 16-qubit engine**: full state vector, real entanglement, 512 KB VRAM
    
- **Pre-built algorithms**: Grover search, Bell pairs, GHZ states, QFT, Phase Estimation
    
- **Differentiable learning layer**: gate angles as `nn.Parameter`, trained by gradient descent
    
- **Animated visualizer**: Bloch sphere GIFs with slowmo interpolation
    
- **Verdict**: the system tells the agent how much to trust its own answer
    
- **Vector-tiled cube model**: the tiling IS the program, the orbit IS the answer. No gates, no state vector. ~25 FLOPs per step. ← NEW
    
- **Orbit store**: FAISS (128-dim similarity) + SQLite (metadata) + GPU brute force (3D tick queries). No server, no daemon. ← NEW
    
- Exposed to AI agents via MCP (**13 tools**) ← UPDATED from 10
    

---

## What makes it different


|                   | CUDANT_VQPC                                              | Real quantum computer     | Classical simulator |       |
| ----------------- | -------------------------------------------------------- | ------------------------- | ------------------- | ----- |
| Hardware          | Laptop GPU (8 GB)                                        | $10B facility             | HPC cluster         |       |
| Temperature       | Room                                                     | 15 mK                     | Room                |       |
| Decoherence       | None                                                     | Constant                  | N/A                 |       |
| Trajectory        | Yes (Box 2)                                              | No (you only get samples) | No                  |       |
| Wobble detection  | Yes                                                      | No                        | No                  |       |
| Self-verification | Verdict (TRUSTED / CAUTION)                              | No                        | No                  |       |
| Agent interface   | MCP (13 tools)                                           | None                      | CLI                 |       |
| Qubits            | 1–16 (exact) + 1–1024 (geometric) + unlimited (cube, N²) | 100s (noisy)              | 16–32 (exact)       |       |
| Learning          | Differentiable (nn.Parameter)                            | No                        | Limited             |       |
| Visualization     | Animated GIF (replay)                                    | No                        | Static plots        |       |
| Orbit search      | GPU brute force + FAISS (in-process)                     | No                        | No                  | ← NEW |
| Complex numbers   | Not needed in cube model                                 | Yes                       | Yes                 | ← NEW |

---

## The 5 Laws

The particle doesn't "simulate" quantum mechanics. It **obeys** 5 rules. Quantum behavior emerges.


| Law                  | What it means                                                                     |
| -------------------- | --------------------------------------------------------------------------------- |
| 1. Superposition     | The particle is a vector, not a point. It can be in multiple states at once.      |
| 2. Unitary Evolution | Every move is a smooth, reversible rotation. The particle never teleports.        |
| 3. Entanglement      | Two particles can become one object. You can't describe them separately.          |
| 4. Measurement       | Looking snaps the vector to a wall. Irreversible. The only non-unitary operation. |
| 5. Interference      | Paths add or cancel. The engine behind Grover, phase estimation, and QFT.         |

**Axis convention:** All angles follow `AXIS.md`. θ is elevation from |0⟩, φ is azimuth from +X. Ry tilts, Rx tilts-and-turns, Rz spins. The full convention (matrices, gotchas, the universal recipe) lives in **`AXIS.md`** — that file is the law.

---

## The Three Backends ← UPDATED from "Two"


|                 | Cage (geometric)               | Engine (exact)                    | Box (cube) ← NEW                                   |
| --------------- | ------------------------------ | --------------------------------- | -------------------------------------------------- |
| Model           | N Bloch vectors on S²          | 2ⁿ complex amplitudes             | Direction vector bouncing in tiled cube            |
| Qubits          | 1–1024                         | 1–16                              | Unlimited (N² tiles per face)                      |
| VRAM            | ~12 KB (1024 lions)            | 512 KB (16 qubits)                | 18 KB (N=16 tiling)                                |
| Entanglement    | "Rod" (snap to +X)             | Full state vector                 | Shared tile hit (geometric coincidence)            |
| Trajectory      | Yes                            | No                                | Yes (full orbit, every tick)                       |
| Wobble          | Yes                            | No (exact by construction)        | Yes (attractor deviation)                          |
| Measurement     | Yes (collapse)                 | Yes (sampling)                    | No (orbit IS the answer)                           |
| Complex numbers | No                             | Yes                               | No                                                 |
| Use             | 1–2 qubits, "show me the path" | 3–16 qubits, "give me the answer" | Production compute, orbit search, storable results |

The agent decides which to use. The `help` tool explains the decision rule.

---

## The Cube Model ← NEW SECTION

```
Domain:  Unit cube [−1, 1]³, faces tiled N×N.
State:   Direction vector v ∈ S² (the ball never moves, only rotates).
Step:    Ray-box intersect → tile lookup → reflect. ~25 FLOPs.
Program: The tiling T (6×N×N×3 floats). 18 KB for N=16.
Answer:  The orbit {h₁, h₂, ..., h_K}. A table of xyz over time.
Zoom:    N (tiles per face). 4, 8, 16, 32, 64...
```

Simple answer: N=4, K=100. Complex answer: N=4096, K=1M, 100K colors.  
The kernel never changes. You just turn the dials.

**Correlation (entanglement in the cube):** Two trajectories hitting the same tile at the same tick are correlated. No explicit link. No rod. Geometric coincidence.

**Convergence (interference in the cube):** Multiple initial directions that funnel to the same attractor reinforce each other. The attractor IS the answer. No complex amplitudes. No phase. Just geometry.

---

## Store Architecture ← NEW SECTION


| Backend            | Data                                   | Scale       | Speed (RTX 4000) |
| ------------------ | -------------------------------------- | ----------- | ---------------- |
| GPU brute force    | Raw ticks (x, y, z)                    | 100M points | ~40 ms query     |
| FAISS (in-process) | Orbit embeddings (128-dim)             | 1M orbits   | ~50 ms search    |
| SQLite             | Metadata (tiling, v0, K, N, timestamp) | Unlimited   | < 1 ms           |

No server. No daemon. No network. Same GPU, same process.

Qdrant path: saved for H100+ scale. `requirements-future.txt`.

---

## Files


| File                      | What it is                                                              |
| ------------------------- | ----------------------------------------------------------------------- |
| `particle.py`             | The particle. Full Bloch sphere (θ+φ), trajectory, measurement.         |
| `laws.py`                 | The 5 quantum laws. The physics engine.                                 |
| `gates.py`                | All gates: I, X, Y, Z, H, S, S†, T, T† + Rx(θ), Ry(θ), Rz(θ).           |
| `measure.py`              | Single-shot collapse + N-shot sampling.                                 |
| `circuit.py`              | Full pipeline: gates → state → trajectory → wobble → measurement.       |
| `cage.py`                 | N particles, rods (CNOT), entanglement, correlated measurement.         |
| `engine.py`               | Exact 2ⁿ state vector (1–16 qubits) + `bloch_vector()` extraction.      |
| `algorithms.py`           | Grover, Bell, GHZ, QFT, Phase Estimation.                               |
| `learn.py`                | LearnableParticle + Lion16Layer (differentiable, wobble regularizer).   |
| `visualize.py`            | Bloch sphere GIF generator (replay tool).                               |
| `box.py`                  | Vector-tiled cube. Tiling IS the program. Orbit IS the answer. ← NEW    |
| `tiling.py`               | Tiling generation: flat, random, gradient, grover. Save/load. ← NEW     |
| `encode_algorithm.py`     | Gate sequence → tiling generator. ← NEW                                 |
| `validate_box.py`         | Unit test: box vs engine. PASS/FAIL. ← NEW                              |
| `store.py`                | FAISS (128-dim) + SQLite (metadata) + GPU brute force (3D ticks). ← NEW |
| `vqpc_server.py`          | MCP server — 13 tools for AI agents.                                    |
| `benchmark.py`            | Spec sheet + `settings.conf` generation.                                |
| `settings.conf`           | VRAM cap, max lions (auto-generated).                                   |
| `AXIS.md`                 | Axis convention. The law.                                               |
| `requirements.txt`        | 5 dependencies: torch, numpy, fastmcp, matplotlib, faiss-cpu.           |
| `requirements-future.txt` | Qdrant (H100+ scale). ← NEW                                             |

---

## MCP Tools (`vqpc_server.py`)


| #   | Tool               | What it does                                                                       |
| --- | ------------------ | ---------------------------------------------------------------------------------- |
| 1   | `run_circuit`      | Run a gate sequence. Returns state + trajectory + wobble + measurement + verdict.  |
| 2   | `measure_state`    | Measure a particle at a specific (θ, φ). Returns distribution.                     |
| 3   | `wobble_report`    | Analyze path quality without measuring. Returns wobble + verdict.                  |
| 4   | `list_gates`       | Full gate reference with descriptions.                                             |
| 5   | `run_cage`         | Multi-particle geometric model. Entanglement via rods.                             |
| 6   | `run_engine`       | Exact state vector (1–16 qubits). Full entanglement.                               |
| 7   | `run_algorithm`    | Pre-built: Grover, Bell, GHZ, QFT, Phase Estimation.                               |
| 8   | `train_particle`   | Train a particle to a target (θ, φ). Gradient descent + wobble.                    |
| 9   | `replay`           | Visualize a circuit's path. Generates a Bloch sphere GIF.                          |
| 10  | `help`             | Tool selection guide. The agent's first call.                                      |
| 11  | `run_box`          | Run trajectories in the vector-tiled cube. Tiling = program, orbit = answer. ← NEW |
| 12  | `encode_algorithm` | Bake a quantum algorithm into a tiling. ← NEW                                      |
| 13  | `validate_box`     | Compare cube model output against the exact engine. PASS/FAIL. ← NEW               |

### Gate syntax


| Context                           | Format             | Example                                      |
| --------------------------------- | ------------------ | -------------------------------------------- |
| `run_circuit` (1 qubit)           | Shorthand          | `"H, T, H"` or `"Rx(30), Rz(45)"`            |
| `run_cage` / `run_engine` (multi) | Explicit           | `["GATE 0 H", "CNOT 0 1", "RX 2 45"]`        |
| `run_box` (cube)                  | Tiling + direction | `tiling="random", v0=[0,0,1], N=16, K=10000` |

**Do NOT mix formats.** `"H0"` is not valid. `"GATE 0 H"` is not valid in `run_circuit`.

---

## The agent loop

```
Agent: "Run H, T, H. Measure 1000 times."
  → run_circuit("H, T, H", n_shots=1000)
  → P(0) = 0.8536, wobble = 1.36, verdict = "CAUTION (wobbly path)"
  → Measurement: 849× |0>, 151× |1>
Agent: "The answer is |0> 85% of the time, but the path is wobbly.
        The phase kick from T is curving the trajectory. Expected."

Agent: "Find |1010> in 16 items."
  → run_algorithm("grover", n_qubits=4, target=10)
  → P(target) = 0.961, iterations = 3, verdict = "FOUND"
  → Bloch vectors show the 1010 pattern in z-components.

Agent: "Show me the path for Rx(50)."
  → replay("Rx(50)", slowmo=True)
  → replay.gif: 10 frames. Straight line. CLEAN.

Agent: "Run a random tiling, 4 colors, 1000 bounces." ← NEW
  → run_box(tiling="random", N=16, K=1000, n_colors=4)
  → 4 orbits stored. Qdrant points: 40. Final directions reported.

Agent: "Is the box correct?" ← NEW
  → validate_box(n_qubits=1, operations=["GATE 0 H"], N=16, K=1000)
  → max_error = 0.0032, verdict = "PASS"
```

---

## Architecture

```
AI Agent (LLM)
    ↓ MCP (stdio)
vqpc_server.py  ← 13 tools
    ↓
┌─────────────────────────────────────────────────────────┐
│  SPHERE MODEL                                           │
│  cage.py (geometric, 1–1024 lions)          ← 1-2 qubits, see the path
│  engine.py (exact, 1–16 qubits)             ← 3-16 qubits, get the answer
│  algorithms.py (Grover, Bell, GHZ, QFT, PE) ← named problems
│  learn.py (differentiable, wobble)          ← the particle learns
│  visualize.py (Bloch sphere GIF)            ← show it
│                                                     │
│  CUBE MODEL ← NEW                                    │
│  box.py (vector-tiled billiard)             ← production compute
│  tiling.py (program generation)             ← the algorithm as geometry
│  encode_algorithm.py (gate seq → tiling)    ← bridge between models
│  validate_box.py (box vs engine)            ← unit test
│                                                     │
│  STORE ← NEW                                        │
│  store.py (FAISS + SQLite + GPU brute force) ← orbit persistence
└─────────────────────────────────────────────────────────┘
    ↓
particle.py / gates.py / measure.py  ← the physics
    ↓
NVIDIA GPU (Quadro RTX 4000, 8 GB, 500 MB cap)
```

---

## The two boxes


| Box             | Data                                 | What it tells the agent                                 |
| --------------- | ------------------------------------ | ------------------------------------------------------- |
| 1 (Information) | Final state vector                   | "The answer is 85% \|0⟩"                                |
| 2 (Trajectory)  | Path of the vector at each gate step | "Here's HOW it got there. And here's where it wobbled." |

Box 2 is what no other simulator gives you. The agent can diagnose _why_ an answer is what it is.

The cube model extends this: the orbit IS Box 2, but for unlimited steps, stored and searchable. ← NEW

---

## Requirements

- Python 3.12
    
- PyTorch (CUDA)
    
- numpy
    
- fastmcp
    
- matplotlib
    
- faiss-cpu ← NEW
    
- NVIDIA GPU (any, 4 GB+ is fine)
    

---

## Related

- **CUDANT** (private) — original 16-qubit engine + CUDANT-DL (learnable layer) + MPS 32
    
- **CUDANT-128** (public) — 128-qubit extension via CUDA-Q MPS with fidelity gate
    
- CUDANT_VQPC (public) — is the **merged product**: geometry + exact engine + cube model + algorithms + learning + visualization + orbit store, all agent-facing
    

---

## Status

- [x] Sphere model (10 tools)
    
- [x] Cube model (box, tiling, encode_algorithm, validate_box)
    
- [x] Store (FAISS + SQLite + GPU brute force)
    
- [x] Server (13 tools, additive)
    
- [x] `test_store.py` — all 5 tests passing
    
- [ ] Live LLM test
    
- [ ] Architecture consolidation